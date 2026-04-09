"""Quick test: run preprocessing on real TabFormer data."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess_tabformer import preprocess_data

metadata, user_mask, mx_mask, tx_mask = preprocess_data(
    raw_csv_path="data/TabFormer/raw/card_transaction.v1.csv",
    output_base_path="data/TabFormer",
)

print("Preprocessing complete!")
print(f"Metadata: {metadata}")
print(f"User mask keys: {list(user_mask.keys())}")
print(f"Merchant mask keys: {list(mx_mask.keys())}")
print(f"Transaction mask keys: {list(tx_mask.keys())}")
