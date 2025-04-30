import os
import yaml
import requests
import pandas as pd
from pathlib import Path

def load_data():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    raw_path = Path(params["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download data
    df = pd.read_csv(params["data"]["source_url"])
    df.to_csv(raw_path, index=False)
    print(f"Data saved to {raw_path}")

if __name__ == "__main__":
    load_data()