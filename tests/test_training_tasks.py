"""Tests for training tasks — config generation and container commands."""

import json
import os
import shutil

import pytest

from pipeline.config import DEFAULT_GNN_PARAMS, DEFAULT_XGB_PARAMS
from pipeline.tasks.training import generate_training_config, build_training_command, validate_model_outputs


class TestGenerateTrainingConfig:
    def test_default_config(self, tmp_path):
        config_path = generate_training_config.fn(
            data_dir="/data/gnn",
            output_dir="/trained_models",
            gnn_params=DEFAULT_GNN_PARAMS,
            xgb_params=DEFAULT_XGB_PARAMS,
            config_dir=str(tmp_path),
        )
        assert os.path.exists(config_path)
        with open(config_path) as f:
            config = json.load(f)
        assert config["paths"]["data_dir"] == "/data"
        assert config["paths"]["output_dir"] == "/trained_models"
        assert config["models"][0]["kind"] == "GNN_XGBoost"
        assert config["models"][0]["hyperparameters"]["gnn"]["hidden_channels"] == 32
        assert config["models"][0]["hyperparameters"]["xgb"]["max_depth"] == 6

    def test_custom_params(self, tmp_path):
        custom_gnn = {**DEFAULT_GNN_PARAMS, "hidden_channels": 64, "num_epochs": 20}
        config_path = generate_training_config.fn(
            data_dir="/data/gnn",
            output_dir="/trained_models",
            gnn_params=custom_gnn,
            xgb_params=DEFAULT_XGB_PARAMS,
            config_dir=str(tmp_path),
        )
        with open(config_path) as f:
            config = json.load(f)
        assert config["models"][0]["hyperparameters"]["gnn"]["hidden_channels"] == 64
        assert config["models"][0]["hyperparameters"]["gnn"]["num_epochs"] == 20


class TestBuildTrainingCommand:
    def test_command_structure(self):
        cmd = build_training_command.fn(
            data_dir="/host/data/gnn",
            output_dir="/host/trained_models",
            config_path="/host/config.json",
            training_image="nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0",
            gpu_device="0",
        )
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--gpus" in cmd
        cmd_str = " ".join(cmd)
        assert "/host/data/gnn:/data" in cmd_str
        assert "/host/trained_models:/trained_models" in cmd_str
        assert "/host/config.json:/app/config.json" in cmd_str


class TestValidateModelOutputs:
    def test_valid_model_repo(self, tmp_model_repo):
        parent = tmp_model_repo.parent / "check"
        repo_dir = parent / "python_backend_model_repository"
        repo_dir.mkdir(parents=True)
        shutil.copytree(
            tmp_model_repo / "prediction_and_shapley",
            repo_dir / "prediction_and_shapley",
        )
        validate_model_outputs.fn(str(parent))

    def test_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Missing model"):
            validate_model_outputs.fn(str(tmp_path))
