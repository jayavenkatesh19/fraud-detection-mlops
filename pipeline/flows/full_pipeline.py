"""Full pipeline orchestrator — chains preprocess → train → evaluate → deploy."""

import logging
import os

from prefect import flow
from prefect.deployments.flow_runs import run_deployment

from pipeline.config import (
    DATA_ROOT,
    DEFAULT_FRAUD_RATIO,
    DEFAULT_GNN_PARAMS,
    DEFAULT_UNDER_SAMPLE,
    DEFAULT_XGB_PARAMS,
    MIN_IMPROVEMENT,
    MODEL_OUTPUT_DIR,
    RAW_CSV_PATH,
    TRITON_HTTP_URL,
    TRITON_MODEL_REPO,
)

logger = logging.getLogger(__name__)


@flow(name="full-pipeline", log_prints=True)
def full_pipeline_flow(
    raw_csv_path: str = RAW_CSV_PATH,
    data_output_path: str = DATA_ROOT,
    model_output_dir: str = MODEL_OUTPUT_DIR,
    triton_url: str = TRITON_HTTP_URL,
    model_repo_path: str = TRITON_MODEL_REPO,
    fraud_ratio: float = DEFAULT_FRAUD_RATIO,
    under_sample: bool = DEFAULT_UNDER_SAMPLE,
    gnn_params: dict | None = None,
    xgb_params: dict | None = None,
    min_improvement: float = MIN_IMPROVEMENT,
) -> dict:
    """End-to-end fraud detection pipeline.

    Runs on the user's PC (default pool) and dispatches compute stages
    to the GPU pool via run_deployment().
    """
    gnn_params = gnn_params or DEFAULT_GNN_PARAMS
    xgb_params = xgb_params or DEFAULT_XGB_PARAMS

    logger.info("Starting full pipeline")

    # Stage 1: Preprocess (gpu pool)
    preprocess_run = run_deployment(
        name="preprocess/preprocess",
        parameters={
            "raw_csv_path": raw_csv_path,
            "output_base_path": data_output_path,
            "fraud_ratio": fraud_ratio,
            "under_sample": under_sample,
        },
        timeout=0,
    )
    logger.info("Preprocessing complete")

    # Stage 2: Train (gpu pool)
    gnn_data_dir = os.path.join(data_output_path, "gnn")
    train_run = run_deployment(
        name="train/train",
        parameters={
            "data_dir": gnn_data_dir,
            "output_dir": model_output_dir,
            "gnn_params": gnn_params,
            "xgb_params": xgb_params,
        },
        timeout=0,
    )
    logger.info("Training complete")

    # Stage 3: Evaluate (gpu pool)
    test_data_dir = os.path.join(data_output_path, "gnn", "test_gnn")
    eval_run = run_deployment(
        name="evaluate/evaluate",
        parameters={
            "challenger_run_id": "latest",
            "challenger_artifacts_path": model_output_dir,
            "test_data_dir": test_data_dir,
            "triton_url": triton_url,
            "model_repo_path": model_repo_path,
            "min_improvement": min_improvement,
        },
        timeout=0,
    )
    logger.info("Evaluation complete")

    # Stage 4: Deploy (gpu pool)
    deploy_run = run_deployment(
        name="deploy/deploy",
        parameters={
            "eval_result": {},
            "triton_url": triton_url,
            "model_repo_path": model_repo_path,
        },
        timeout=0,
    )
    logger.info("Deploy complete")

    return {"status": "complete"}


if __name__ == "__main__":
    full_pipeline_flow()
