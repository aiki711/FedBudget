# mlflow_config.py
import mlflow
from pathlib import Path

def setup_mlflow():
    mlruns_path = Path(__file__).resolve().parents[1] / "mlruns"
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    print(f"✅ MLflow URI set to: file:{mlruns_path}")