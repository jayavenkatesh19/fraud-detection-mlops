"""Tests for data validation tasks."""

import pandas as pd
import pytest

from pipeline.tasks.data import validate_raw_input, validate_gnn_outputs, load_test_data


class TestValidateRawInput:
    def test_valid_csv(self, tmp_data_dir):
        result = validate_raw_input.fn(str(tmp_data_dir / "raw" / "card_transaction.v1.csv"))
        assert result["row_count"] == 100
        assert "fraud_ratio" in result
        assert result["null_counts"] is not None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_raw_input.fn(str(tmp_path / "nonexistent.csv"))

    def test_missing_columns_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"wrong_col": [1, 2, 3]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_raw_input.fn(str(bad_csv))


class TestValidateGnnOutputs:
    def test_valid_outputs(self, tmp_gnn_dir):
        result = validate_gnn_outputs.fn(str(tmp_gnn_dir))
        assert result["training_files_ok"] is True
        assert result["test_files_ok"] is True

    def test_missing_files_raises(self, tmp_path):
        (tmp_path / "gnn").mkdir()
        with pytest.raises(FileNotFoundError, match="Missing GNN"):
            validate_gnn_outputs.fn(str(tmp_path / "gnn"))


class TestLoadTestData:
    def test_loads_graph_data(self, tmp_gnn_dir):
        data = load_test_data.fn(str(tmp_gnn_dir / "test_gnn"))
        assert "x_user" in data
        assert "x_merchant" in data
        assert "edge_index_user_to_merchant" in data
        assert "edge_attr_user_to_merchant" in data
        assert "edge_label_user_to_merchant" in data
