"""Deploy flow — promote or rollback Triton model versions."""

import logging

from prefect import flow

from pipeline.config import TRITON_HTTP_URL, TRITON_MODEL_NAME, TRITON_MODEL_REPO
from pipeline.tasks.mlflow_utils import register_champion
from pipeline.tasks.triton import (
    cleanup_version_artifacts,
    health_check,
    reload_model,
)

logger = logging.getLogger(__name__)


@flow(name="deploy", log_prints=True)
def deploy_flow(
    eval_result: dict,
    triton_url: str = TRITON_HTTP_URL,
    model_name: str = TRITON_MODEL_NAME,
    model_repo_path: str = TRITON_MODEL_REPO,
) -> dict:
    """Deploy or rollback based on evaluation result.

    If eval says promote:
    1. Challenger already loaded in Triton (from evaluate stage reload)
    2. Remove old champion version artifacts
    3. Reload model in Triton (now only challenger version remains)
    4. Health check the new champion
    5. Register in MLflow as champion

    If eval says reject:
    1. Remove challenger artifacts
    2. Reload model (restores champion-only state)
    """
    should_promote = eval_result["should_promote"]
    champion_version = eval_result["champion_version"]
    challenger_version = eval_result["challenger_version"]
    challenger_run_id = eval_result["challenger_run_id"]

    if should_promote:
        logger.info("Promoting challenger version %d as new champion", challenger_version)

        # Remove old champion artifacts and reload
        if champion_version > 0:
            cleanup_version_artifacts(model_repo_path, model_name, champion_version)
            reload_model(triton_url, model_name)

        # Health check new champion
        healthy = health_check(triton_url, model_name, challenger_version)
        if not healthy:
            logger.error("Health check failed for version %d", challenger_version)
            raise RuntimeError(
                f"Deploy failed: version {challenger_version} not healthy after promotion."
            )

        # Register in MLflow
        register_champion(challenger_run_id, model_name)

        logger.info("Deploy complete: version %d is now champion", challenger_version)
        return {"action": "promoted", "version": challenger_version, "run_id": challenger_run_id}

    else:
        logger.info("Rejecting challenger version %d", challenger_version)

        # Remove challenger artifacts and reload to restore champion-only state
        cleanup_version_artifacts(model_repo_path, model_name, challenger_version)
        if champion_version > 0:
            reload_model(triton_url, model_name)

        logger.info(
            "Rejected: version %d cleaned up, champion version %d unchanged",
            challenger_version, champion_version,
        )
        return {"action": "rejected", "version": challenger_version, "reason": eval_result.get("reason", "")}


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1:
        eval_result = json.loads(sys.argv[1])
        deploy_flow(eval_result=eval_result)
