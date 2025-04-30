import yaml
import mlflow
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import json
def evaluate():
    # Initialize MLflow tracking with same settings as training
    mlflow.set_tracking_uri("file:///mlruns")  # Add this line
    mlflow.set_experiment("Spam Classification")  # Add this line
    
    with mlflow.start_run(run_name="evaluation"):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        # Load artifacts
        model = joblib.load("models/model.pkl")
        tfidf = joblib.load("models/tfidf_vectorizer.pkl")
        
        # Load test data
        test_path = Path(params["data"]["processed_path"]) / "test.csv"
        test_df = pd.read_csv(test_path)
        
        # Prepare features
        X_test = tfidf.transform(test_df[params["features"]["text_column"]])
        y_test = test_df[params["features"]["target_column"]]
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save metrics
        metrics_dir = Path("reports")
        metrics_dir.mkdir(exist_ok=True)
        # pd.DataFrame([metrics]).to_csv(metrics_dir / "evaluation_metrics.csv", index=False)
        # To JSON
        with open(metrics_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f)
        print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    evaluate()