import sys
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from invoice_flagging.data_preprocessing import (
    load_invoice_data,
    apply_labels
)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

def main():
    db_path = BASE_DIR / "Data" / "inventory.db"
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    print("Loading data from database...")
    try:
        df = load_invoice_data(db_path)
        print(f"Total rows retrieved: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Drop missing values in our features
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    
    # We want a clean, representative sample of background data (e.g. 1000 rows)
    sample_size = min(1000, len(df))
    print(f"Sampling {sample_size} records for background dataset...")
    df_sample = df.sample(n=sample_size, random_state=42)
    
    X_background = df_sample[FEATURES]
    
    model_dir = BASE_DIR / "invoice_flagging" / "models"
    model_dir.mkdir(exist_ok=True)
    
    output_path = model_dir / "background_data.pkl"
    joblib.dump(X_background, output_path)
    print(f"Successfully saved background data to: {output_path}")

if __name__ == "__main__":
    main()
