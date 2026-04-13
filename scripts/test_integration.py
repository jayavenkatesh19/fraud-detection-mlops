#!/usr/bin/env python3
"""Integration test — run train, evaluate, deploy flows against live MLflow + Prefect + Triton."""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("integration-test")

# Ensure pipeline is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set env vars before importing pipeline config
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5050")
os.environ.setdefault("PREFECT_API_URL", "http://localhost:4200/api")
os.environ.setdefault("TRITON_HTTP_URL", "localhost:8000")
os.environ.setdefault("TRITON_GRPC_URL", "localhost:8001")
os.environ.setdefault("DATA_ROOT", "/home/ubuntu/fraud-detection-mlops/data/TabFormer")
os.environ.setdefault("MODEL_OUTPUT_DIR", "/home/ubuntu/fraud-detection-mlops/data/trained_models")
os.environ.setdefault("TRITON_MODEL_REPO", "/home/ubuntu/fraud-detection-mlops/data/models")


def test_connectivity():
    """Test connectivity to all three services."""
    import mlflow
    import tritonclient.http as httpclient

    # MLflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    exp = mlflow.get_experiment_by_name("fraud-detection-lp")
    logger.info("MLflow experiment: %s", exp.experiment_id if exp else "will create")
    logger.info("MLflow: OK")

    # Prefect — just check the API is reachable
    import requests
    resp = requests.get(f"{os.environ['PREFECT_API_URL']}/health")
    assert resp.status_code == 200, f"Prefect health check failed: {resp.status_code}"
    logger.info("Prefect: OK")

    # Triton
    client = httpclient.InferenceServerClient(url=os.environ["TRITON_HTTP_URL"])
    assert client.is_server_ready(), "Triton not ready"
    logger.info("Triton: OK (server ready)")


def test_train_flow():
    """Run the train flow — this invokes the training container, logs to MLflow."""
    from pipeline.flows.train import train_flow

    logger.info("=== Running train_flow ===")
    run_id = train_flow(
        data_dir=os.path.join(os.environ["DATA_ROOT"], "gnn"),
        output_dir=os.environ["MODEL_OUTPUT_DIR"],
    )
    logger.info("train_flow completed — MLflow run_id: %s", run_id)
    return run_id


def test_evaluate_flow(run_id: str):
    """Run the evaluate flow — stages challenger, scores via Triton, decides promotion."""
    from pipeline.flows.evaluate import evaluate_flow

    logger.info("=== Running evaluate_flow ===")
    result = evaluate_flow(
        challenger_run_id=run_id,
        challenger_artifacts_path=os.environ["MODEL_OUTPUT_DIR"],
        test_data_dir=os.path.join(os.environ["DATA_ROOT"], "gnn", "test_gnn"),
        triton_url=os.environ["TRITON_HTTP_URL"],
        model_name="prediction_and_shapley",
        model_repo_path=os.environ["TRITON_MODEL_REPO"],
    )
    logger.info("evaluate_flow completed — should_promote: %s, reason: %s",
                result["should_promote"], result["reason"])
    return result


def test_deploy_flow(eval_result: dict):
    """Run the deploy flow — promotes or rejects based on eval result."""
    from pipeline.flows.deploy import deploy_flow

    logger.info("=== Running deploy_flow ===")
    result = deploy_flow(
        eval_result=eval_result,
        triton_url=os.environ["TRITON_HTTP_URL"],
        model_name="prediction_and_shapley",
        model_repo_path=os.environ["TRITON_MODEL_REPO"],
    )
    logger.info("deploy_flow completed — action: %s", result["action"])
    return result


def test_full_pipeline():
    """Run the full pipeline orchestrator (skip preprocess since data exists)."""
    from pipeline.flows.full_pipeline import full_pipeline_flow

    logger.info("=== Running full_pipeline_flow ===")
    result = full_pipeline_flow(skip_preprocess=True)
    logger.info("full_pipeline_flow completed — status: %s", result["status"])
    logger.info("  train_run_id: %s", result["train_run_id"])
    logger.info("  eval decision: %s", result["eval_result"]["reason"])
    logger.info("  deploy action: %s", result["deploy_result"]["action"])
    return result


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    if stage in ("connectivity", "all"):
        test_connectivity()

    if stage == "full":
        test_full_pipeline()
    elif stage in ("train", "all"):
        run_id = test_train_flow()

        if stage in ("evaluate", "all"):
            eval_result = test_evaluate_flow(run_id)
        else:
            eval_result = None

        if stage in ("deploy", "all") and eval_result:
            test_deploy_flow(eval_result)
    elif stage == "evaluate":
        run_id = sys.argv[2] if len(sys.argv) > 2 else "manual-test"
        test_evaluate_flow(run_id)
    elif stage == "deploy":
        pass  # deploy needs eval_result, can't run standalone easily

    logger.info("=== Integration test complete ===")


if __name__ == "__main__":
    main()
