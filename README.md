# Fraud Detection MLOps Pipeline

A production MLOps pipeline for the [NVIDIA Financial Fraud Detection AI Blueprint](https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection), built with Prefect, MLflow, and Triton Inference Server.

Takes the blueprint's notebook-based GNN + XGBoost fraud detection workflow and wraps it in automated orchestration, experiment tracking, champion/challenger model evaluation, and zero-downtime deployment with automatic rollback.

## What This Does

The blueprint provides a Jupyter notebook that manually walks through: download data, preprocess, train in a Docker container, serve on Triton, run inference. This pipeline automates that entire flow:

```
[1. Preprocess] --> [2. Train] --> [3. Evaluate] --> [4. Deploy]
     cuDF             NGC            Triton          Triton
     graph          container       versioning      promotion
   formation        black box      champion vs       or
   + validation    + MLflow log    challenger       rollback
```

**Preprocess**: Validates raw TabFormer CSV, runs cuDF-based graph formation (bipartite User-Merchant graph with edge attributes), validates outputs, logs metadata to MLflow.

**Train**: Generates training config, runs the `financial-fraud-training:2.0.0` NGC container as a black box, validates model outputs, logs hyperparameters and artifacts to MLflow.

**Evaluate**: Stages the new model as the next version in Triton's model repository, reloads Triton, scores both the challenger and current champion on held-out test data via version-specific inference, compares F1 scores, and decides whether to promote.

**Deploy**: If the challenger wins, removes the old champion version, health-checks the new one, and registers it in MLflow's model registry with the `champion` alias. If it loses, cleans up the challenger artifacts and keeps the current champion.

## Model Variant

This pipeline uses the **Link Prediction** variant of the fraud detection blueprint:

- **Graph**: Bipartite (User <-> Merchant)
- **Edges**: Represent transactions, with attributes (encoded features) and labels (fraud/not-fraud)
- **Task**: Edge classification ("Is this user-merchant transaction fraudulent?")
- **Training config**: `kind: "GNN_XGBoost"` (SAGEConv GNN + XGBoost ensemble)

## Architecture

### Work Pools

| Pool | Where it runs | What runs there |
|------|--------------|-----------------|
| `gpu` | Cloud GPU instance | preprocess, train, evaluate, deploy |
| `default` | Your laptop | full_pipeline orchestrator only |

The orchestrator on your laptop dispatches all compute to the GPU pool via `run_deployment()`. In local-only mode, both pools run on the same machine.

### Infrastructure

| Service | Role | Port |
|---------|------|------|
| **Prefect** | Orchestration, scheduling, work pools, UI | 4200 |
| **MLflow** | Experiment tracking, model registry, artifact storage | 5000 |
| **Triton** | GPU inference, model versioning, model control API | 8000/8001/8002 |

### Champion/Challenger via Triton Native Versioning

Triton's model repository supports multiple version directories (`1/`, `2/`, `3/`...). With `version_policy: { all: {} }` in `config.pbtxt`, all versions are served simultaneously. The evaluate flow exploits this:

1. Stage challenger artifacts as version N+1
2. Reload model (Triton picks up both versions)
3. Send test data to `model_version="N"` and `model_version="N+1"` separately
4. Compare metrics, decide promotion
5. Remove the loser's version directory, reload again

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **Docker** with NVIDIA Container Toolkit
- **NGC API Key** from https://ngc.nvidia.com/setup/api-key (needed to pull the training container)
- **Python 3.10+** with conda (for cuDF)
- **TabFormer dataset** (266MB, downloaded separately; see below)

## Setup

### 1. Clone and configure

```bash
git clone <this-repo>
cd fraud-detection-mlops
cp .env.example .env
# Edit .env: set NGC_API_KEY and adjust paths for your environment
```

### 2. Install Python dependencies

```bash
# Create a conda environment (cuDF requires conda)
conda create -n fraud-mlops python=3.12 -y
conda activate fraud-mlops

# Install RAPIDS cuDF
conda install -c rapidsai -c nvidia -c conda-forge cudf=25.08 python=3.12 -y

# Install pipeline dependencies
pip install -r requirements.txt
```

### 3. Download the TabFormer dataset

```bash
./scripts/download_data.sh /path/to/data/TabFormer
```

This will prompt you to download `transactions.tgz` from [IBM Box](https://ibm.ent.box.com/v/tabformer-data/folder/130747715605) and extract it. The result should be:

```
/path/to/data/TabFormer/
└── raw/
    └── card_transaction.v1.csv    # 24M rows, 15 columns
```

### 4. Start the infrastructure

```bash
# MLflow + Prefect (no GPU needed)
docker compose up -d

# With Triton (needs GPU)
docker compose --profile gpu up -d
```

Verify:
- Prefect UI: http://localhost:4200
- MLflow UI: http://localhost:5000

### 5. Pull the training container

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker pull nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0
```

### 6. Create Prefect work pools

```bash
prefect work-pool create gpu --type process
prefect work-pool create default --type process
```

## Running the Pipeline

### Full automated pipeline

```bash
# Register and serve all deployments
python -m pipeline.deployments
```

This starts serving 5 deployments:
- `preprocess` (gpu pool, nightly at midnight)
- `train` (gpu pool, on-demand)
- `evaluate` (gpu pool, on-demand)
- `deploy` (gpu pool, on-demand)
- `full-pipeline` (default pool, nightly at 2am)

Trigger manually from the Prefect UI, or:

```bash
# Start a GPU worker (on the GPU machine)
prefect worker start --pool gpu

# Start a default worker (on your laptop)
prefect worker start --pool default
```

### Run individual stages

Each flow can be run standalone for testing or ad-hoc experiments:

```bash
# Preprocess only
python -m pipeline.flows.preprocess

# Train with default hyperparameters
python -m pipeline.flows.train

# Train with custom hyperparameters
python -m pipeline.flows.train \
    --gnn-params '{"hidden_channels": 64, "num_epochs": 20}' \
    --xgb-params '{"max_depth": 8}'

# Evaluate a specific MLflow run
python -m pipeline.flows.evaluate --challenger-run-id <mlflow_run_id>

# Deploy (pass eval result JSON)
python -m pipeline.flows.deploy --eval-result '{"should_promote": true, ...}'
```

## Project Structure

```
fraud-detection-mlops/
├── pipeline/
│   ├── config.py                 # All defaults, paths, thresholds in one place
│   ├── deployments.py            # Prefect deployment registration + schedules
│   ├── flows/
│   │   ├── preprocess.py         # Stage 1: validate + cuDF preprocessing + MLflow
│   │   ├── train.py              # Stage 2: config gen + container + MLflow
│   │   ├── evaluate.py           # Stage 3: Triton versioning + champion/challenger
│   │   ├── deploy.py             # Stage 4: promote or rollback
│   │   └── full_pipeline.py      # Orchestrator: chains stages via run_deployment()
│   └── tasks/
│       ├── data.py               # Input validation, output validation, data loading
│       ├── training.py           # Config generation, Docker commands, output checks
│       ├── mlflow_utils.py       # Experiment mgmt, metrics, model registry aliases
│       └── triton.py             # Version staging, model reload, health checks, scoring
├── scripts/
│   ├── preprocess_tabformer.py   # Adapted from blueprint (parameterized, logging)
│   └── download_data.sh          # TabFormer dataset download helper
├── tests/                        # 32 unit tests, all passing without GPU
│   ├── conftest.py               # Synthetic data fixtures
│   ├── test_data_tasks.py
│   ├── test_training_tasks.py
│   ├── test_mlflow_tasks.py
│   ├── test_triton_tasks.py
│   └── test_scoring.py
├── triton/
│   └── Dockerfile                # Custom Triton image (PyTorch, XGBoost GPU, Captum)
├── docker-compose.yml            # MLflow + Prefect + Triton (GPU profile)
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Configuration

All configuration lives in `pipeline/config.py` and can be overridden via environment variables (loaded from `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_ROOT` | `/data/TabFormer` | Base directory for TabFormer data |
| `MODEL_OUTPUT_DIR` | `/data/trained_models` | Where training artifacts are written |
| `TRITON_MODEL_REPO` | `/models` | Triton model repository root |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `PREFECT_API_URL` | `http://localhost:4200/api` | Prefect server URL |
| `TRITON_HTTP_URL` | `localhost:8000` | Triton HTTP endpoint |
| `NGC_API_KEY` | (required) | NGC API key for training container |

### Default Hyperparameters

**GNN**: SAGEConv, 32 hidden channels, 2 hops, dropout 0.1, 8 epochs, batch size 4096

**XGBoost**: max depth 6, learning rate 0.2, 3 parallel trees, 512 boost rounds

Override per-run via flow parameters or the Prefect UI.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

All 32 tests run without GPU or Docker. They use synthetic data fixtures and mock external services (MLflow, Triton).

## Design Decisions

1. **Training container is a black box.** We orchestrate around it (config gen, container lifecycle, artifact collection) and never modify its internals.
2. **cuDF preserved.** This is a RAPIDS deployment example; preprocessing uses GPU DataFrames throughout.
3. **Triton native versioning** for champion/challenger. No separate evaluation server needed.
4. **`--model-control-mode=explicit`** means Triton loads nothing at startup; we control everything via the model control API.
5. **MLflow aliases** (`champion`) instead of deprecated stage-based promotion.
6. **Deferred cuDF import.** The preprocessing flow imports cuDF only when it runs, so all other flows work on CPU-only machines.

## Related

- [NVIDIA Financial Fraud Detection AI Blueprint](https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection) (source notebook this pipeline is built from)
- [IBM TabFormer](https://github.com/IBM/TabFormer) (synthetic credit card transaction dataset)
- [Prefect 3.x docs](https://docs.prefect.io)
- [MLflow docs](https://mlflow.org/docs/latest)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
