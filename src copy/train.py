import yaml
import mlflow
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # ADD THIS IMPORT
from pathlib import Path
import json

def train_model():
    # Initialize MLflow tracking
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("Spam Classification")
    with mlflow.start_run(run_name="training") as run:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        # Load training data
        train_path = Path(params["data"]["processed_path"]) / "train.csv"
        train_df = pd.read_csv(train_path)
        
        # Feature engineering
        tfidf = TfidfVectorizer(
            max_features=params["features"]["max_features"],
            ngram_range=tuple(params["features"]["ngram_range"])
        )
        X_train = tfidf.fit_transform(train_df[params["features"]["text_column"]])
        y_train = train_df[params["features"]["target_column"]]
        
        # Train model
        model = LogisticRegression(**params["model"]["hyperparameters"])
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)  # Now works
        
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Save training metrics
        training_metrics = {
            "train_accuracy": train_accuracy
        }
        with open(reports_dir / "training_metrics.json", "w") as f:
            json.dump(training_metrics, f)
        
        # Log metrics to MLflow
        mlflow.log_metrics(training_metrics)

        # Save artifacts
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        joblib.dump(tfidf, model_dir / "tfidf_vectorizer.pkl")
        joblib.dump(model, model_dir / "model.pkl")
        
        # Log to MLflow
        mlflow.log_params(params["model"]["hyperparameters"])
        mlflow.log_params(params["features"])
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model training complete. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()