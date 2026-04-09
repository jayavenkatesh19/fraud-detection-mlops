"""Tests for MLflow utility tasks."""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.tasks.mlflow_utils import (
    get_or_create_experiment,
    log_training_run,
    log_evaluation_metrics,
    get_champion_metrics,
    register_champion,
)


class TestGetOrCreateExperiment:
    @patch("pipeline.tasks.mlflow_utils.mlflow")
    def test_creates_experiment(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "123"
        result = get_or_create_experiment.fn("test-experiment")
        assert result == "123"
        mock_mlflow.create_experiment.assert_called_once_with("test-experiment")

    @patch("pipeline.tasks.mlflow_utils.mlflow")
    def test_returns_existing(self, mock_mlflow):
        mock_exp = MagicMock()
        mock_exp.experiment_id = "456"
        mock_mlflow.get_experiment_by_name.return_value = mock_exp
        result = get_or_create_experiment.fn("existing-experiment")
        assert result == "456"


class TestLogEvaluationMetrics:
    @patch("pipeline.tasks.mlflow_utils.mlflow")
    def test_logs_challenger_metrics(self, mock_mlflow):
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        challenger = {"f1_score": 0.85, "precision": 0.80, "recall": 0.90, "accuracy": 0.88}
        champion = {"f1_score": 0.82, "precision": 0.78, "recall": 0.87, "accuracy": 0.85}
        log_evaluation_metrics.fn("run-123", challenger, champion)

        # Verify log_metrics was called (at least for challenger and deltas)
        assert mock_mlflow.log_metrics.call_count >= 2

    @patch("pipeline.tasks.mlflow_utils.mlflow")
    def test_logs_without_champion(self, mock_mlflow):
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        challenger = {"f1_score": 0.85}
        log_evaluation_metrics.fn("run-123", challenger, None)
        # Only challenger metrics logged, no deltas
        mock_mlflow.log_metrics.assert_called_once()


class TestGetChampionMetrics:
    @patch("pipeline.tasks.mlflow_utils.MlflowClient")
    def test_returns_none_when_no_champion(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []
        mock_client_cls.return_value = mock_client
        result = get_champion_metrics.fn("prediction_and_shapley")
        assert result is None

    @patch("pipeline.tasks.mlflow_utils.MlflowClient")
    def test_returns_metrics_for_champion(self, mock_client_cls):
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.run_id = "champion-run"
        mock_version.aliases = ["champion"]
        mock_client.search_model_versions.return_value = [mock_version]

        mock_run = MagicMock()
        mock_run.data.metrics = {"f1_score": 0.82, "precision": 0.78}
        mock_client.get_run.return_value = mock_run

        mock_client_cls.return_value = mock_client
        result = get_champion_metrics.fn("prediction_and_shapley")
        assert result["f1_score"] == 0.82
