import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import mlflow

def preprocess():
    # Initialize MLflow
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("Spam Classification")
    
    with mlflow.start_run(run_name="preprocessing"):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        # Load raw data
        df = pd.read_csv(params["data"]["raw_path"])
        
        # Rename columns based on params.yaml
        df = df.rename(columns={
            "v2": params["features"]["text_column"],  # Original text column is "v2"
            "v1": "original_label"  # Temporary name for label column
        })
        
        # Create numerical labels
        df[params["features"]["target_column"]] = df["original_label"].map({
            "ham": 0, 
            "spam": 1
        })
        
        # Select final columns (FIXED LINE)
        df = df[[
            params["features"]["text_column"],
            params["features"]["target_column"]
        ]]  # Added missing closing bracket
        
        # Split data
        train_df, test_df = train_test_split(
            df,
            test_size=params["data"]["test_size"],
            random_state=params["data"]["random_state"]
        )
        
        # Save processed data
        processed_dir = Path(params["data"]["processed_path"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Log parameters
        mlflow.log_params({
            "dataset_size": len(df),
            "train_size": len(train_df),
            "test_size": len(test_df)
        })

if __name__ == "__main__":
    preprocess()