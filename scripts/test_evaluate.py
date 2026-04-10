"""Test evaluate flow: stage challenger, reload Triton, score both versions."""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.tasks.triton import (
    get_current_version,
    stage_challenger_version,
    health_check,
)

MODEL_REPO = "data/trained_models/python_backend_model_repository"
MODEL_NAME = "prediction_and_shapley"
TRITON_URL = "localhost:8000"

# Step 1: Check current version
current = get_current_version.fn(MODEL_REPO, MODEL_NAME)
print(f"Current version in repo: {current}")

# Step 2: Stage challenger (copy version 1 as version 2 to simulate a new training run)
source_artifacts = os.path.join(MODEL_REPO, MODEL_NAME, "1")
champion_v, challenger_v = stage_challenger_version.fn(source_artifacts, MODEL_REPO, MODEL_NAME)
print(f"Staged: champion=v{champion_v}, challenger=v{challenger_v}")

# Step 3: Verify version directory exists
v2_dir = os.path.join(MODEL_REPO, MODEL_NAME, str(challenger_v))
print(f"Version {challenger_v} dir exists: {os.path.exists(v2_dir)}")
print(f"Files: {os.listdir(v2_dir)}")

# Step 4: Reload Triton to pick up new version
import tritonclient.http as httpclient
client = httpclient.InferenceServerClient(url=TRITON_URL)
print("\nUnloading model...")
client.unload_model(MODEL_NAME)

print("Loading model (both versions)...")
client.load_model(MODEL_NAME)

import time
for i in range(30):
    try:
        if client.is_model_ready(MODEL_NAME):
            break
    except Exception:
        pass
    time.sleep(2)

# Step 5: Check both versions are available
v1_ready = health_check.fn(TRITON_URL, MODEL_NAME, champion_v)
v2_ready = health_check.fn(TRITON_URL, MODEL_NAME, challenger_v)
print(f"\nVersion {champion_v} ready: {v1_ready}")
print(f"Version {challenger_v} ready: {v2_ready}")

# Step 6: Score both versions
if v1_ready and v2_ready:
    import numpy as np
    from pipeline.tasks.data import load_test_data
    from sklearn.metrics import f1_score

    test_data = load_test_data.fn("data/TabFormer/gnn/test_gnn")
    labels = test_data["edge_label_user_to_merchant"]
    if hasattr(labels, "values"):
        labels = labels.values
    labels = np.asarray(labels, dtype=np.int32).ravel()
    inference_data = {k: v for k, v in test_data.items() if not k.startswith("edge_label_")}

    from pipeline.tasks.triton import score_model_version
    print("\nScoring version 1 (champion)...")
    m1 = score_model_version.fn(TRITON_URL, MODEL_NAME, champion_v, inference_data, labels)
    print(f"  F1: {m1['f1_score']:.4f}")

    print(f"Scoring version 2 (challenger)...")
    m2 = score_model_version.fn(TRITON_URL, MODEL_NAME, challenger_v, inference_data, labels)
    print(f"  F1: {m2['f1_score']:.4f}")

    print(f"\nDelta: {m2['f1_score'] - m1['f1_score']:.4f}")
else:
    print("Not all versions ready, skipping scoring")

# Step 7: Cleanup - remove version 2
from pipeline.tasks.triton import cleanup_version_artifacts
cleanup_version_artifacts.fn(MODEL_REPO, MODEL_NAME, challenger_v)
print(f"\nCleaned up version {challenger_v}")

# Reload to restore single-version state
client.unload_model(MODEL_NAME)
client.load_model(MODEL_NAME)
time.sleep(5)
print(f"Model ready after cleanup: {client.is_model_ready(MODEL_NAME)}")
print("Done!")
