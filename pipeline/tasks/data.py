"""Data validation and loading tasks."""

import logging
import os

import numpy as np
import pandas as pd
from prefect import task

from pipeline.config import (
    EXPECTED_GNN_TEST_FILES,
    EXPECTED_GNN_TRAINING_FILES,
    EXPECTED_RAW_COLUMNS,
)

logger = logging.getLogger(__name__)


@task(name="validate-raw-input")
def validate_raw_input(csv_path: str) -> dict:
    """Validate that the raw CSV exists and has expected columns.

    Returns metadata dict with row_count, fraud_ratio, null_counts.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, nrows=0)
    missing = set(EXPECTED_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Read full file for metadata (just the fraud column for efficiency)
    df_full = pd.read_csv(csv_path, usecols=["Is Fraud?"])
    row_count = len(df_full)
    fraud_count = (df_full["Is Fraud?"] == "Yes").sum()
    fraud_ratio = fraud_count / row_count if row_count > 0 else 0.0

    # Null counts per column (sample first 10k rows for speed)
    df_sample = pd.read_csv(csv_path, nrows=10000)
    null_counts = df_sample.isnull().sum().to_dict()

    metadata = {
        "row_count": row_count,
        "fraud_count": int(fraud_count),
        "fraud_ratio": float(fraud_ratio),
        "null_counts": null_counts,
    }
    logger.info("Input validation passed: %d rows, %.2f%% fraud", row_count, fraud_ratio * 100)
    return metadata


@task(name="validate-gnn-outputs")
def validate_gnn_outputs(gnn_dir: str) -> dict:
    """Validate that preprocessing produced all expected GNN files."""
    missing_training = []
    for f in EXPECTED_GNN_TRAINING_FILES:
        if not os.path.exists(os.path.join(gnn_dir, f)):
            missing_training.append(f)

    missing_test = []
    for f in EXPECTED_GNN_TEST_FILES:
        if not os.path.exists(os.path.join(gnn_dir, f)):
            missing_test.append(f)

    all_missing = missing_training + missing_test
    if all_missing:
        raise FileNotFoundError(f"Missing GNN output files: {all_missing}")

    result = {"training_files_ok": True, "test_files_ok": True}
    logger.info("GNN output validation passed")
    return result


@task(name="load-test-data")
def load_test_data(test_data_dir: str) -> dict:
    """Load heterogeneous graph test data for inference.

    Adapted from preprocess_TabFormer_lp.load_hetero_graph().
    Returns dict with node features, edge indices, edge attrs, labels, and feature masks.
    """
    nodes_dir = os.path.join(test_data_dir, "nodes")
    edges_dir = os.path.join(test_data_dir, "edges")
    out = {}

    # Load node features and feature masks
    if os.path.isdir(nodes_dir):
        for fname in sorted(os.listdir(nodes_dir)):
            if fname.endswith(".csv") and not fname.endswith("_feature_mask.csv"):
                node_name = fname[:-len(".csv")]
                node_df = pd.read_csv(os.path.join(nodes_dir, fname))
                out[f"x_{node_name}"] = node_df.to_numpy(dtype=np.float32)

                mask_path = os.path.join(nodes_dir, f"{node_name}_feature_mask.csv")
                if os.path.exists(mask_path):
                    mask = pd.read_csv(mask_path, header=None).to_numpy(dtype=np.int32).ravel()
                else:
                    mask = np.zeros(node_df.shape[1], dtype=np.int32)
                out[f"feature_mask_{node_name}"] = mask

    # Load edges: base, attrs, labels, feature masks
    base_edges = {}
    edge_attrs = {}
    edge_labels = {}
    edge_feature_masks = {}

    if os.path.isdir(edges_dir):
        for fname in sorted(os.listdir(edges_dir)):
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(edges_dir, fname)
            if fname.endswith("_attr.csv"):
                edge_name = fname[:-len("_attr.csv")]
                edge_attrs[edge_name] = pd.read_csv(path)
            elif fname.endswith("_label.csv"):
                edge_name = fname[:-len("_label.csv")]
                edge_labels[edge_name] = pd.read_csv(path)
            elif fname.endswith("_feature_mask.csv"):
                edge_name = fname[:-len("_feature_mask.csv")]
                edge_feature_masks[edge_name] = pd.read_csv(path, header=None)
            else:
                edge_name = fname[:-len(".csv")]
                base_edges[edge_name] = pd.read_csv(path)

    for edge_name, df in base_edges.items():
        out[f"edge_index_{edge_name}"] = df.to_numpy(dtype=np.int64).T
        if edge_name in edge_attrs:
            out[f"edge_attr_{edge_name}"] = edge_attrs[edge_name].to_numpy(dtype=np.float32)
        if edge_name in edge_feature_masks:
            out[f"edge_feature_mask_{edge_name}"] = (
                edge_feature_masks[edge_name].to_numpy(dtype=np.int32).ravel()
            )
        elif edge_name in edge_attrs:
            out[f"edge_feature_mask_{edge_name}"] = np.zeros(
                edge_attrs[edge_name].shape[1], dtype=np.int32
            )

    for label_edge_name, label_df in edge_labels.items():
        out[f"edge_label_{label_edge_name}"] = label_df

    return out
