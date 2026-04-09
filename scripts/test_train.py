"""Quick test: run training container with preprocessed data."""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.tasks.training import generate_training_config, build_training_command
from pipeline.config import DEFAULT_GNN_PARAMS, DEFAULT_XGB_PARAMS

data_dir = os.path.abspath("data/TabFormer/gnn")
output_dir = os.path.abspath("data/trained_models")
os.makedirs(output_dir, exist_ok=True)

# Generate config
config_path = generate_training_config.fn(
    data_dir=data_dir,
    output_dir=output_dir,
    gnn_params=DEFAULT_GNN_PARAMS,
    xgb_params=DEFAULT_XGB_PARAMS,
    config_dir=".",
)

print(f"Config written to: {config_path}")
with open(config_path) as f:
    print(json.dumps(json.load(f), indent=2))

# Build and run the training command
config_path = os.path.abspath(config_path)
cmd = build_training_command.fn(
    data_dir=data_dir,
    output_dir=output_dir,
    config_path=config_path,
)

print(f"\nRunning: {' '.join(cmd)}")
result = subprocess.run(cmd, text=True)

if result.returncode == 0:
    print("\nTraining complete!")
    # Check output
    model_repo = os.path.join(output_dir, "python_backend_model_repository")
    if os.path.exists(model_repo):
        for root, dirs, files in os.walk(model_repo):
            for f in files:
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                print(f"  {os.path.relpath(path, output_dir)} ({size} bytes)")
    else:
        print(f"WARNING: {model_repo} not found")
else:
    print(f"\nTraining failed with exit code {result.returncode}")
