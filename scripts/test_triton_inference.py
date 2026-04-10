"""Test Triton inference with real preprocessed test data."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.tasks.data import load_test_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tritonclient.http as httpclient
from tritonclient.http import InferInput, InferRequestedOutput

# Load test data
test_dir = "data/TabFormer/gnn/test_gnn"
print(f"Loading test data from {test_dir}...")
test_data = load_test_data.fn(test_dir)

labels = test_data["edge_label_user_to_merchant"]
if hasattr(labels, "values"):
    labels = labels.values
labels = np.asarray(labels, dtype=np.int32).ravel()

# Build inference request
inference_data = {k: v for k, v in test_data.items() if not k.startswith("edge_label_")}

client = httpclient.InferenceServerClient(url="localhost:8000")

print(f"Server ready: {client.is_server_ready()}")
print(f"Model ready: {client.is_model_ready('prediction_and_shapley')}")

inputs = []
for key, value in inference_data.items():
    if key.startswith("x_"):
        dtype = "FP32"
    elif key.startswith("feature_mask_"):
        dtype = "INT32"
    elif key.startswith("edge_feature_mask_"):
        dtype = "INT32"
    elif key.startswith("edge_index_"):
        dtype = "INT64"
    elif key.startswith("edge_attr_"):
        dtype = "FP32"
    else:
        continue
    inp = InferInput(key, list(value.shape), datatype=dtype)
    inp.set_data_from_numpy(value)
    inputs.append(inp)

# Add COMPUTE_SHAP = False
shap_flag = np.array([False], dtype=np.bool_)
inp = InferInput("COMPUTE_SHAP", [1], datatype="BOOL")
inp.set_data_from_numpy(shap_flag)
inputs.append(inp)

outputs = [InferRequestedOutput("PREDICTION")]

print("Sending inference request...")
response = client.infer("prediction_and_shapley", inputs=inputs, outputs=outputs)
predictions = response.as_numpy("PREDICTION")

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions sample: {predictions[:10].ravel()}")

# Compute metrics
y_pred = (predictions > 0.5).astype(int).ravel()
y_true = labels.ravel()

print(f"\nMetrics:")
print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
print(f"\nTest data: {len(y_true)} samples, {y_true.sum()} fraud")
