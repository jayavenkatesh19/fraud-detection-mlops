"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with minimal synthetic TabFormer CSV."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    n_rows = 100
    rng = np.random.RandomState(42)

    df = pd.DataFrame({
        "User": rng.randint(0, 5, n_rows),
        "Card": rng.randint(0, 3, n_rows),
        "Year": rng.choice([2017, 2018, 2019], n_rows),
        "Month": rng.randint(1, 13, n_rows),
        "Day": rng.randint(1, 29, n_rows),
        "Time": [f"{rng.randint(0, 24)}:{rng.randint(0, 60):02d}" for _ in range(n_rows)],
        "Amount": [f"${rng.uniform(1, 5000):.2f}" for _ in range(n_rows)],
        "Use Chip": rng.choice(["Chip Transaction", "Online Transaction", "Swipe Transaction"], n_rows),
        "Merchant Name": rng.choice(["merchant_a", "merchant_b", "merchant_c"], n_rows),
        "Merchant City": rng.choice(["CityA", "CityB"], n_rows),
        "Merchant State": rng.choice(["CA", "NY", None], n_rows),
        "Zip": rng.choice([90210.0, 10001.0, None], n_rows),
        "MCC": rng.choice([5411, 5912, 7011], n_rows),
        "Errors?": rng.choice(["Bad PIN", None, "Insufficient Balance"], n_rows),
        "Is Fraud?": rng.choice(["Yes", "No"], n_rows, p=[0.1, 0.9]),
    })

    df.to_csv(raw_dir / "card_transaction.v1.csv", index=False)
    return tmp_path


@pytest.fixture
def tmp_gnn_dir(tmp_path):
    """Create a temporary GNN output directory with minimal graph data."""
    gnn_dir = tmp_path / "gnn"
    nodes_dir = gnn_dir / "nodes"
    edges_dir = gnn_dir / "edges"
    test_nodes = gnn_dir / "test_gnn" / "nodes"
    test_edges = gnn_dir / "test_gnn" / "edges"

    for d in [nodes_dir, edges_dir, test_nodes, test_edges]:
        d.mkdir(parents=True)

    pd.DataFrame(np.random.randn(5, 3).astype(np.float32), columns=["f0", "f1", "f2"]).to_csv(
        nodes_dir / "user.csv", index=False
    )
    pd.DataFrame(np.random.randn(4, 3).astype(np.float32), columns=["f0", "f1", "f2"]).to_csv(
        nodes_dir / "merchant.csv", index=False
    )

    pd.DataFrame({"src": [0, 1, 2, 3, 4], "dst": [0, 1, 2, 3, 0]}).to_csv(
        edges_dir / "user_to_merchant.csv", index=False
    )
    pd.DataFrame(np.random.randn(5, 3).astype(np.float32), columns=["a0", "a1", "a2"]).to_csv(
        edges_dir / "user_to_merchant_attr.csv", index=False
    )
    pd.DataFrame({"Fraud": [0, 0, 1, 0, 0]}).to_csv(
        edges_dir / "user_to_merchant_label.csv", index=False
    )

    pd.DataFrame(np.random.randn(3, 3).astype(np.float32), columns=["f0", "f1", "f2"]).to_csv(
        test_nodes / "user.csv", index=False
    )
    pd.DataFrame(np.random.randn(2, 3).astype(np.float32), columns=["f0", "f1", "f2"]).to_csv(
        test_nodes / "merchant.csv", index=False
    )
    np.savetxt(test_nodes / "user_feature_mask.csv", [0, 1, 2], delimiter=",", fmt="%d")
    np.savetxt(test_nodes / "merchant_feature_mask.csv", [3, 4, 5], delimiter=",", fmt="%d")

    pd.DataFrame({"src": [0, 1, 2], "dst": [0, 1, 0]}).to_csv(
        test_edges / "user_to_merchant.csv", index=False
    )
    pd.DataFrame(np.random.randn(3, 3).astype(np.float32), columns=["a0", "a1", "a2"]).to_csv(
        test_edges / "user_to_merchant_attr.csv", index=False
    )
    pd.DataFrame({"Fraud": [0, 1, 0]}).to_csv(
        test_edges / "user_to_merchant_label.csv", index=False
    )
    np.savetxt(test_edges / "user_to_merchant_feature_mask.csv", [6, 7, 8], delimiter=",", fmt="%d")

    return gnn_dir


@pytest.fixture
def tmp_model_repo(tmp_path):
    """Create a minimal Triton model repository structure."""
    model_dir = tmp_path / "prediction_and_shapley"
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True)

    (model_dir / "config.pbtxt").write_text('name: "prediction_and_shapley"\nbackend: "python"\n')
    (version_dir / "model.py").write_text("# placeholder model")
    (version_dir / "meta.json").write_text('{"version": 1}')
    (version_dir / "state_dict_gnn_model.pth").write_bytes(b"fake_weights")
    (version_dir / "embedding_based_xgboost.json").write_text('{"learner": {}}')

    return tmp_path
