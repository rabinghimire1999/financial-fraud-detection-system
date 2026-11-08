from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DEFAULT_HASH_BUCKETS = 8
DEFAULT_K_NEIGHBORS = 8


@dataclass
class PreparedGraphBatches:
    cache_dir: Path
    train_batch_files: list[Path]
    test_batch_files: list[Path]
    scaler_file: Path
    projector_file: Path
    metadata_file: Path
    target_column: str
    feature_columns: list[str]
    train_size: int
    test_size: int
    n_components: int
    k_neighbors: int
    chunksize: int
    graph_feature_dim: int
    dataset_id: str


@dataclass
class GraphBatch:
    X: np.ndarray
    y: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    num_nodes: int
    batch_file: Path


def resolve_data_path(*candidates: str | Path) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    joined = ", ".join(str(Path(candidate)) for candidate in candidates)
    raise FileNotFoundError(f"None of the data paths exist: {joined}")


def _csv_compression(path: Path) -> str | None:
    return "zip" if path.suffix.lower() == ".zip" else None


def _read_head(data_path: Path, nrows: int = 1024) -> pd.DataFrame:
    return pd.read_csv(
        data_path,
        compression=_csv_compression(data_path),
        nrows=nrows,
    )


def _read_chunks(data_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        data_path,
        compression=_csv_compression(data_path),
        chunksize=chunksize,
    )


def detect_target_column(columns: Iterable[str]) -> str:
    for candidate in ("Class", "Fraud_Flag"):
        if candidate in columns:
            return candidate
    available = ", ".join(columns)
    raise ValueError(f"Could not detect target column. Available columns: {available}")


def _hash_categorical_column(series: pd.Series, buckets: int) -> dict[str, np.ndarray]:
    text = series.fillna("__nan__").astype(str)
    hashed = pd.util.hash_pandas_object(text, index=False).to_numpy(dtype=np.uint64)
    bucket_ids = np.mod(hashed, buckets).astype(np.int64)

    features: dict[str, np.ndarray] = {}
    for bucket in range(buckets):
        features[f"{series.name}__hash_{bucket}"] = (bucket_ids == bucket).astype(np.float32)
    return features


def build_feature_frame(
    chunk: pd.DataFrame,
    target_column: str,
    *,
    categorical_hash_buckets: int = DEFAULT_HASH_BUCKETS,
) -> pd.DataFrame:
    features = chunk.copy()

    if "Transaction_Date" in features.columns:
        timestamp = pd.to_datetime(features["Transaction_Date"], errors="coerce")
        hour = timestamp.dt.hour.fillna(0).astype(np.float32)
        day = timestamp.dt.dayofweek.fillna(0).astype(np.float32)

        features["hour_sin"] = np.sin((2.0 * np.pi * hour) / 24.0).astype(np.float32)
        features["hour_cos"] = np.cos((2.0 * np.pi * hour) / 24.0).astype(np.float32)
        features["day_sin"] = np.sin((2.0 * np.pi * day) / 7.0).astype(np.float32)
        features["day_cos"] = np.cos((2.0 * np.pi * day) / 7.0).astype(np.float32)

    drop_columns = {
        target_column,
        "id",
        "Transaction_ID",
        "Customer_ID",
        "Merchant_ID",
        "Transaction_Date",
    }
    features = features.drop(columns=[column for column in drop_columns if column in features.columns])

    dense_features: dict[str, np.ndarray] = {}
    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            dense_features[column] = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float32,
                copy=False,
            )
            continue

        dense_features.update(_hash_categorical_column(series, categorical_hash_buckets))

    if not dense_features:
        raise ValueError("No usable features were extracted from the dataset.")

    return pd.DataFrame(dense_features, index=features.index)


def _transform_chunk(
    chunk: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    *,
    categorical_hash_buckets: int = DEFAULT_HASH_BUCKETS,
) -> tuple[np.ndarray, np.ndarray]:
    feature_frame = build_feature_frame(
        chunk,
        target_column,
        categorical_hash_buckets=categorical_hash_buckets,
    )
    feature_frame = feature_frame.reindex(columns=feature_columns, fill_value=0.0)

    X_chunk = feature_frame.to_numpy(dtype=np.float32, copy=False)
    y_chunk = pd.to_numeric(chunk[target_column], errors="coerce").fillna(0).to_numpy(
        dtype=np.int8,
        copy=False,
    )
    return X_chunk, y_chunk


def _split_indices(y: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y), dtype=np.int32)
    stratify = None

    if len(y) >= 4:
        value_counts = pd.Series(y).value_counts()
        if len(value_counts) > 1 and int(value_counts.min()) >= 2:
            stratify = y

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_idx.astype(np.int32, copy=False), test_idx.astype(np.int32, copy=False)


def build_knn_graph_features(
    X_scaled: np.ndarray,
    *,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    n_samples = len(X_scaled)

    if n_samples == 0:
        empty_features = np.empty((0, 0), dtype=np.float32)
        empty_edges = np.empty((2, 0), dtype=np.int32)
        empty_weights = np.empty((0,), dtype=np.float32)
        return empty_features, empty_edges, empty_weights

    if n_samples == 1:
        zeros = np.zeros_like(X_scaled)
        summary = np.zeros((1, 3), dtype=np.float32)
        graph_features = np.concatenate([X_scaled, zeros, zeros, summary], axis=1).astype(
            np.float32,
            copy=False,
        )
        empty_edges = np.empty((2, 0), dtype=np.int32)
        empty_weights = np.empty((0,), dtype=np.float32)
        return graph_features, empty_edges, empty_weights

    effective_k = min(k_neighbors + 1, n_samples)
    knn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean", n_jobs=-1)
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(X_scaled)

    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:].astype(np.float32, copy=False)

    if neighbor_indices.shape[1] == 0:
        zeros = np.zeros_like(X_scaled)
        summary = np.zeros((n_samples, 3), dtype=np.float32)
        graph_features = np.concatenate([X_scaled, zeros, zeros, summary], axis=1).astype(
            np.float32,
            copy=False,
        )
        empty_edges = np.empty((2, 0), dtype=np.int32)
        empty_weights = np.empty((0,), dtype=np.float32)
        return graph_features, empty_edges, empty_weights

    neighbors = X_scaled[neighbor_indices]
    neighbor_mean = neighbors.mean(axis=1).astype(np.float32, copy=False)
    neighbor_delta = (X_scaled - neighbor_mean).astype(np.float32, copy=False)

    mean_distance = neighbor_distances.mean(axis=1, keepdims=True).astype(np.float32, copy=False)
    min_distance = neighbor_distances.min(axis=1, keepdims=True).astype(np.float32, copy=False)
    degree = np.full((n_samples, 1), neighbor_indices.shape[1], dtype=np.float32)

    graph_features = np.concatenate(
        [
            X_scaled,
            neighbor_mean,
            neighbor_delta,
            mean_distance,
            min_distance,
            degree,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    src = np.repeat(np.arange(n_samples, dtype=np.int32), neighbor_indices.shape[1])
    dst = neighbor_indices.reshape(-1).astype(np.int32, copy=False)
    edge_index = np.vstack([src, dst]).astype(np.int32, copy=False)
    edge_weight = (1.0 / (1.0 + neighbor_distances.reshape(-1))).astype(np.float32, copy=False)

    return graph_features, edge_index, edge_weight


def _cache_paths(cache_dir: Path, stem: str) -> tuple[Path, Path, Path, Path]:
    scaler_file = cache_dir / f"{stem}_scaler.joblib"
    projector_file = cache_dir / f"{stem}_graph_ipca.joblib"
    metadata_file = cache_dir / f"{stem}_graph_metadata.json"
    dataset_index_file = cache_dir / f"{stem}_graph_index.json"
    return scaler_file, projector_file, metadata_file, dataset_index_file


def _load_prepared_graph_batches(cache_dir: Path, stem: str) -> PreparedGraphBatches:
    scaler_file, projector_file, metadata_file, dataset_index_file = _cache_paths(cache_dir, stem)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    index = json.loads(dataset_index_file.read_text(encoding="utf-8"))

    train_batch_files = [cache_dir / "train_batches" / name for name in index["train_batch_files"]]
    test_batch_files = [cache_dir / "test_batches" / name for name in index["test_batch_files"]]

    return PreparedGraphBatches(
        cache_dir=cache_dir,
        train_batch_files=train_batch_files,
        test_batch_files=test_batch_files,
        scaler_file=scaler_file,
        projector_file=projector_file,
        metadata_file=metadata_file,
        target_column=metadata["target_column"],
        feature_columns=metadata["feature_columns"],
        train_size=int(metadata["train_size"]),
        test_size=int(metadata["test_size"]),
        n_components=int(metadata["n_components"]),
        k_neighbors=int(metadata["k_neighbors"]),
        chunksize=int(metadata["chunksize"]),
        graph_feature_dim=int(metadata["graph_feature_dim"]),
        dataset_id=metadata["dataset_id"],
    )


def prepare_graph_batches(
    data_path: str | Path,
    cache_dir: str | Path,
    *,
    n_components: int = 4,
    chunksize: int = 25_000,
    test_size: float = 0.2,
    random_state: int = 42,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    categorical_hash_buckets: int = DEFAULT_HASH_BUCKETS,
    force_rebuild: bool = False,
) -> PreparedGraphBatches:
    data_path = Path(data_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_dir = cache_dir / "train_batches"
    test_dir = cache_dir / "test_batches"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    stem = data_path.stem.replace(".csv", "")
    scaler_file, projector_file, metadata_file, dataset_index_file = _cache_paths(cache_dir, stem)

    if (
        not force_rebuild
        and scaler_file.exists()
        and projector_file.exists()
        and metadata_file.exists()
        and dataset_index_file.exists()
    ):
        return _load_prepared_graph_batches(cache_dir, stem)

    for batch_file in train_dir.glob("*.npz"):
        batch_file.unlink()
    for batch_file in test_dir.glob("*.npz"):
        batch_file.unlink()

    head = _read_head(data_path)
    target_column = detect_target_column(head.columns)
    feature_columns = build_feature_frame(
        head,
        target_column,
        categorical_hash_buckets=categorical_hash_buckets,
    ).columns.tolist()

    scaler = StandardScaler(copy=False)
    for chunk in _read_chunks(data_path, chunksize):
        X_chunk, _ = _transform_chunk(
            chunk,
            target_column,
            feature_columns,
            categorical_hash_buckets=categorical_hash_buckets,
        )
        scaler.partial_fit(X_chunk)
        del chunk, X_chunk
        gc.collect()

    projector = IncrementalPCA(
        n_components=n_components,
        batch_size=max(n_components * 64, min(chunksize, 8_192)),
    )
    graph_feature_dim = 0

    for chunk in _read_chunks(data_path, chunksize):
        X_chunk, _ = _transform_chunk(
            chunk,
            target_column,
            feature_columns,
            categorical_hash_buckets=categorical_hash_buckets,
        )
        X_scaled = scaler.transform(X_chunk).astype(np.float32, copy=False)
        graph_features, _, _ = build_knn_graph_features(X_scaled, k_neighbors=k_neighbors)
        projector.partial_fit(graph_features)
        graph_feature_dim = graph_features.shape[1]
        del chunk, X_chunk, X_scaled, graph_features
        gc.collect()

    train_batch_files: list[Path] = []
    test_batch_files: list[Path] = []
    train_size_total = 0
    test_size_total = 0

    for chunk_id, chunk in enumerate(_read_chunks(data_path, chunksize)):
        X_chunk, y_chunk = _transform_chunk(
            chunk,
            target_column,
            feature_columns,
            categorical_hash_buckets=categorical_hash_buckets,
        )
        X_scaled = scaler.transform(X_chunk).astype(np.float32, copy=False)
        train_idx, test_idx = _split_indices(y_chunk, test_size, random_state + chunk_id)

        for split_name, split_idx in (("train", train_idx), ("test", test_idx)):
            if len(split_idx) == 0:
                continue

            X_split = X_scaled[split_idx]
            y_split = y_chunk[split_idx].astype(np.int8, copy=False)

            graph_features, edge_index, edge_weight = build_knn_graph_features(
                X_split,
                k_neighbors=k_neighbors,
            )
            X_reduced = projector.transform(graph_features).astype(np.float32, copy=False)

            out_dir = train_dir if split_name == "train" else test_dir
            out_file = out_dir / f"batch_{chunk_id:05d}.npz"
            np.savez_compressed(
                out_file,
                X=X_reduced,
                y=y_split,
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=np.asarray([len(y_split)], dtype=np.int32),
            )

            if split_name == "train":
                train_batch_files.append(out_file)
                train_size_total += len(y_split)
            else:
                test_batch_files.append(out_file)
                test_size_total += len(y_split)

            del X_split, y_split, graph_features, edge_index, edge_weight, X_reduced
            gc.collect()

        del chunk, X_chunk, y_chunk, X_scaled, train_idx, test_idx
        gc.collect()

    dataset_id = (
        f"{data_path.resolve()}::{data_path.stat().st_size}::{data_path.stat().st_mtime_ns}"
        f"::{chunksize}::{k_neighbors}::{categorical_hash_buckets}::{n_components}"
    )

    metadata = {
        "data_path": str(data_path),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "train_size": train_size_total,
        "test_size": test_size_total,
        "n_components": n_components,
        "k_neighbors": k_neighbors,
        "chunksize": chunksize,
        "categorical_hash_buckets": categorical_hash_buckets,
        "graph_feature_dim": graph_feature_dim,
        "dataset_id": dataset_id,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dataset_index_file.write_text(
        json.dumps(
            {
                "train_batch_files": [path.name for path in train_batch_files],
                "test_batch_files": [path.name for path in test_batch_files],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    joblib.dump(scaler, scaler_file)
    joblib.dump(projector, projector_file)

    return PreparedGraphBatches(
        cache_dir=cache_dir,
        train_batch_files=train_batch_files,
        test_batch_files=test_batch_files,
        scaler_file=scaler_file,
        projector_file=projector_file,
        metadata_file=metadata_file,
        target_column=target_column,
        feature_columns=feature_columns,
        train_size=train_size_total,
        test_size=test_size_total,
        n_components=n_components,
        k_neighbors=k_neighbors,
        chunksize=chunksize,
        graph_feature_dim=graph_feature_dim,
        dataset_id=dataset_id,
    )


def load_graph_batch(batch_file: str | Path) -> GraphBatch:
    batch_file = Path(batch_file)
    with np.load(batch_file) as arrays:
        X = arrays["X"].astype(np.float32, copy=False)
        y = arrays["y"].astype(np.int8, copy=False)
        edge_index = arrays["edge_index"].astype(np.int32, copy=False)
        edge_weight = arrays["edge_weight"].astype(np.float32, copy=False)
        num_nodes = int(arrays["num_nodes"][0]) if "num_nodes" in arrays else int(len(y))

    return GraphBatch(
        X=X,
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        batch_file=batch_file,
    )


def create_qgcnn(num_qubits: int = 4) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    qc = QuantumCircuit(num_qubits)

    x = [Parameter(f"x{i}") for i in range(num_qubits)]
    theta = [Parameter(f"theta{i}") for i in range(6)]

    for qubit in range(num_qubits):
        qc.ry(x[qubit], qubit)

    qc.cx(0, 1)
    qc.ry(theta[0], 1)
    qc.cx(2, 3)
    qc.ry(theta[1], 3)

    qc.cx(0, 1)
    qc.rz(theta[2], 1)
    qc.cx(2, 3)
    qc.rz(theta[3], 3)

    qc.cx(1, 3)
    qc.ry(theta[4], 3)
    qc.cx(1, 3)
    qc.rz(theta[5], 3)

    return qc, x, theta


def create_qcnn(num_qubits: int = 4) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    return create_qgcnn(num_qubits=num_qubits)


def build_qgcnn_classifier(
    qc: QuantumCircuit,
    input_params: list[Parameter],
    weight_params: list[Parameter],
    *,
    maxiter_per_batch: int = 8,
    initial_point: np.ndarray | None = None,
    callback=None,
) -> NeuralNetworkClassifier:
    estimator = Estimator(
        backend_options={"method": "statevector"},
        approximation=True,
    )
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    # COBYLA avoids the SciPy SLSQP gradient-shape failure seen in Colab.
    optimizer = COBYLA(maxiter=maxiter_per_batch)
    return NeuralNetworkClassifier(
        qnn,
        optimizer=optimizer,
        warm_start=True,
        initial_point=initial_point,
        callback=callback,
    )


def predict_scores(classifier: NeuralNetworkClassifier, X: np.ndarray) -> np.ndarray:
    raw = classifier.neural_network.forward(X, classifier.weights)
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    return np.clip((raw + 1.0) / 2.0, 0.0, 1.0)


def score_qgcnn_on_batches(
    classifier: NeuralNetworkClassifier,
    batch_files: list[Path],
    *,
    max_samples: int | None = None,
) -> float:
    total = 0
    correct = 0

    for batch_file in batch_files:
        batch = load_graph_batch(batch_file)
        X_batch = batch.X
        y_batch = batch.y

        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None and total + len(y_batch) > max_samples:
            keep = max_samples - total
            X_batch = X_batch[:keep]
            y_batch = y_batch[:keep]

        predictions = classifier.predict(X_batch).astype(np.int8, copy=False)
        correct += int((predictions == y_batch).sum())
        total += len(y_batch)

    return float(correct / total) if total else 0.0


def collect_predictions_from_batches(
    classifier: NeuralNetworkClassifier,
    batch_files: list[Path],
    *,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    collected = 0

    for batch_file in batch_files:
        batch = load_graph_batch(batch_file)
        X_batch = batch.X
        y_batch = batch.y

        if max_samples is not None and collected >= max_samples:
            break

        if max_samples is not None and collected + len(y_batch) > max_samples:
            keep = max_samples - collected
            X_batch = X_batch[:keep]
            y_batch = y_batch[:keep]

        y_true_parts.append(y_batch.astype(np.int8, copy=False))
        score_parts.append(predict_scores(classifier, X_batch))
        collected += len(y_batch)

    if not y_true_parts:
        return np.empty((0,), dtype=np.int8), np.empty((0,), dtype=np.float32)

    return (
        np.concatenate(y_true_parts, axis=0).astype(np.int8, copy=False),
        np.concatenate(score_parts, axis=0).astype(np.float32, copy=False),
    )


def train_qgcnn_in_batches(
    prepared: PreparedGraphBatches,
    qc: QuantumCircuit,
    input_params: list[Parameter],
    weight_params: list[Parameter],
    *,
    checkpoint_dir: str | Path,
    epochs: int = 2,
    maxiter_per_batch: int = 6,
    random_state: int = 42,
    resume: bool = True,
    eval_max_samples: int | None = 20_000,
) -> tuple[NeuralNetworkClassifier, list[dict[str, float | int | str | None]]]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    weights_path = checkpoint_dir / "latest_weights.npy"
    best_weights_path = checkpoint_dir / "best_weights.npy"
    model_path = checkpoint_dir / "latest_model.dill"
    history_path = checkpoint_dir / "history.json"
    state_path = checkpoint_dir / "state.json"

    history: list[dict[str, float | int | str | None]] = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    callback_losses: list[float] = []

    def callback(weights: np.ndarray, loss: float) -> None:
        callback_losses.append(float(loss))
        print(f"Optimizer step {len(callback_losses):04d} | Loss: {loss:.6f}")

    initial_point = None
    if resume and weights_path.exists():
        initial_point = np.load(weights_path)

    classifier = build_qgcnn_classifier(
        qc,
        input_params,
        weight_params,
        maxiter_per_batch=maxiter_per_batch,
        initial_point=initial_point,
        callback=callback,
    )

    start_epoch = 0
    start_position = 0
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("dataset_id") == prepared.dataset_id:
            start_epoch = int(state.get("next_epoch", 0))
            start_position = int(state.get("next_batch_position", 0))

    best_eval_accuracy = max(
        (
            float(record.get("eval_accuracy", 0.0))
            for record in history
            if record.get("eval_accuracy") is not None
        ),
        default=float("-inf"),
    )

    num_batches = len(prepared.train_batch_files)
    if start_epoch >= epochs:
        return classifier, history

    for epoch in range(start_epoch, epochs):
        order = np.random.default_rng(random_state + epoch).permutation(num_batches)
        batch_position_start = start_position if epoch == start_epoch else 0

        for batch_position in range(batch_position_start, num_batches):
            batch_index = int(order[batch_position])
            batch = load_graph_batch(prepared.train_batch_files[batch_index])

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Graph batch {batch_position + 1}/{num_batches} | "
                f"Nodes {batch.num_nodes:,} | "
                f"Edges {batch.edge_index.shape[1]:,}"
            )

            if len(np.unique(batch.y)) < 2:
                record = {
                    "epoch": epoch + 1,
                    "batch_position": batch_position + 1,
                    "batch_file": batch.batch_file.name,
                    "batch_size": int(batch.num_nodes),
                    "edge_count": int(batch.edge_index.shape[1]),
                    "train_probe_accuracy": None,
                    "eval_accuracy": None,
                    "last_loss": None,
                    "status": "skipped_single_class_batch",
                }
            else:
                classifier.fit(batch.X, batch.y)
                probe_size = min(batch.num_nodes, 4096)
                predictions = classifier.predict(batch.X[:probe_size]).astype(np.int8, copy=False)
                train_probe_accuracy = float((predictions == batch.y[:probe_size]).mean())
                eval_accuracy = score_qgcnn_on_batches(
                    classifier,
                    prepared.test_batch_files,
                    max_samples=eval_max_samples,
                )

                record = {
                    "epoch": epoch + 1,
                    "batch_position": batch_position + 1,
                    "batch_file": batch.batch_file.name,
                    "batch_size": int(batch.num_nodes),
                    "edge_count": int(batch.edge_index.shape[1]),
                    "train_probe_accuracy": train_probe_accuracy,
                    "eval_accuracy": eval_accuracy,
                    "last_loss": float(callback_losses[-1]) if callback_losses else None,
                    "status": "trained",
                }

                np.save(weights_path, classifier.weights)
                try:
                    classifier.save(str(model_path))
                except Exception:
                    pass

                if eval_accuracy >= best_eval_accuracy:
                    best_eval_accuracy = float(eval_accuracy)
                    np.save(best_weights_path, classifier.weights)

            history.append(record)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            next_epoch = epoch
            next_batch_position = batch_position + 1
            if next_batch_position >= num_batches:
                next_epoch = epoch + 1
                next_batch_position = 0

            state_path.write_text(
                json.dumps(
                    {
                        "dataset_id": prepared.dataset_id,
                        "next_epoch": next_epoch,
                        "next_batch_position": next_batch_position,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            del batch, record
            gc.collect()

        start_position = 0

    return classifier, history
