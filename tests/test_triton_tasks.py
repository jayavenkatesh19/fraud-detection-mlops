"""Tests for Triton version management and health check tasks."""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.tasks.triton import (
    get_current_version,
    stage_challenger_version,
    health_check,
    cleanup_version_artifacts,
)


class TestGetCurrentVersion:
    def test_single_version(self, tmp_model_repo):
        result = get_current_version.fn(str(tmp_model_repo), "prediction_and_shapley")
        assert result == 1

    def test_multiple_versions(self, tmp_model_repo):
        v2 = tmp_model_repo / "prediction_and_shapley" / "2"
        v2.mkdir()
        (v2 / "model.py").write_text("# v2")
        result = get_current_version.fn(str(tmp_model_repo), "prediction_and_shapley")
        assert result == 2

    def test_no_model_returns_zero(self, tmp_path):
        result = get_current_version.fn(str(tmp_path), "nonexistent_model")
        assert result == 0


class TestStageChallenger:
    def test_stages_as_next_version(self, tmp_model_repo):
        source = tmp_model_repo / "prediction_and_shapley" / "1"
        champion_v, challenger_v = stage_challenger_version.fn(
            str(source), str(tmp_model_repo), "prediction_and_shapley"
        )
        assert champion_v == 1
        assert challenger_v == 2
        assert (tmp_model_repo / "prediction_and_shapley" / "2" / "model.py").exists()

    def test_first_deployment(self, tmp_path):
        """When no model exists, champion_version=0, challenger_version=1."""
        source = tmp_path / "source_artifacts"
        source.mkdir()
        (source / "model.py").write_text("# model")

        champion_v, challenger_v = stage_challenger_version.fn(
            str(source), str(tmp_path), "prediction_and_shapley"
        )
        assert champion_v == 0
        assert challenger_v == 1


class TestHealthCheck:
    @patch("pipeline.tasks.triton.httpclient.InferenceServerClient")
    def test_healthy_server(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.is_model_ready.return_value = True
        mock_client_cls.return_value = mock_client
        assert health_check.fn("localhost:8000", "prediction_and_shapley", 1) is True
        # Verify model_version is passed as string
        mock_client.is_model_ready.assert_called_once_with("prediction_and_shapley", model_version="1")

    @patch("pipeline.tasks.triton.httpclient.InferenceServerClient")
    def test_unhealthy_server(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.is_model_ready.return_value = False
        mock_client_cls.return_value = mock_client
        assert health_check.fn("localhost:8000", "prediction_and_shapley", 1) is False

    @patch("pipeline.tasks.triton.httpclient.InferenceServerClient")
    def test_connection_error_returns_false(self, mock_client_cls):
        mock_client_cls.side_effect = ConnectionError("refused")
        assert health_check.fn("localhost:8000", "prediction_and_shapley", 1) is False


class TestCleanupVersionArtifacts:
    def test_removes_version_dir(self, tmp_model_repo):
        v2 = tmp_model_repo / "prediction_and_shapley" / "2"
        v2.mkdir()
        (v2 / "model.py").write_text("# v2")
        cleanup_version_artifacts.fn(str(tmp_model_repo), "prediction_and_shapley", 2)
        assert not v2.exists()
        assert (tmp_model_repo / "prediction_and_shapley" / "1").exists()

    def test_no_op_for_missing_version(self, tmp_model_repo):
        """Should not raise if version doesn't exist."""
        cleanup_version_artifacts.fn(str(tmp_model_repo), "prediction_and_shapley", 99)
