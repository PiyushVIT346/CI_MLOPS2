import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd

# Set up DagsHub authentication
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN not found")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "PiyushVIT346"
repo_name = "mlflow_exp_dagshub"
repo_url = f"{dagshub_url}/{repo_owner}/{repo_name}"

# Set MLflow Tracking URI
os.environ["MLFLOW_TRACKING_URI"] = repo_url

model_name = "Best Model"

class TestModelLoading(unittest.TestCase):
    """Unit test class to verify MLflow model loading from the staging stage"""

    def test_model_in_staging(self):
        """Test to verify if the model is in the staging stage"""
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        self.assertGreater(len(versions), 0, "Model not in staging stage")

    def test_model_loading(self):
        """Test to verify if the model can be loaded"""
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not versions:
            self.skipTest("No model found in staging stage, skipping model loading test.")

        latest_version = versions[0].version
        run_id = versions[0].run_id
        logged_model = f"runs:/{run_id}/{model_name}"

        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Failed to load model: {str(e)}")

        self.assertIsNotNone(loaded_model, "Model loading failed")
        print(f"Model successfully loaded: {logged_model}")

if __name__ == "__main__":
    unittest.main()
