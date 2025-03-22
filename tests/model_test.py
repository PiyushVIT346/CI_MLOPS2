import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd

# Set up DagsHub authentication
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN not found in environment variables.")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Define MLflow Tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "PiyushVIT346"
repo_name = "mlflow_exp_dagshub"
repo_url = f"{dagshub_url}/{repo_owner}/{repo_name}"

os.environ["MLFLOW_TRACKING_URI"] = repo_url

# Model name
model_name = "Best Model"


class TestModelLoading(unittest.TestCase):
    """Unit test class to verify MLflow model loading from the staging stage"""

    def setUp(self):
        """Set up MLflow client"""
        self.client = MlflowClient(tracking_uri=repo_url)

    def test_model_in_staging(self):
        """Test if the model exists in the Staging stage"""
        versions = self.client.get_latest_versions(model_name, stages=["Staging"])
        self.assertGreater(len(versions), 0, "No model found in Staging stage.")

    def test_model_loading(self):
        """Test if the model can be successfully loaded from Staging"""
        versions = self.client.get_latest_versions(model_name, stages=["Staging"])

        if not versions:
            self.skipTest("No model found in Staging stage, skipping model loading test.")

        latest_version = versions[0].version
        run_id = versions[0].run_id
        logged_model = f"runs:/{run_id}/{model_name}"

        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Failed to load model: {str(e)}")

        self.assertIsNotNone(loaded_model, "Model loading failed.")
        print(f"✅ Model successfully loaded: {logged_model}")

    def test_model_performance(self):
        """Test if the loaded model performs well on test data"""
        versions = self.client.get_latest_versions(model_name, stages=["Staging"])

        if not versions:
            self.skipTest("No model found in Staging stage, skipping model performance test.")

        latest_version = versions[0].run_id
        logged_model = f"runs:/{latest_version}/{model_name}"

        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Failed to load model: {str(e)}")

        test_data_path = "./data/processed/test_processed.csv"

        if not os.path.exists(test_data_path):
            self.fail(f"Test data file '{test_data_path}' not found.")

        test_data = pd.read_csv(test_data_path)

        if "Potability" not in test_data.columns:
            self.fail("Column 'Potability' missing in test data.")

        x_test = test_data.drop(columns=["Potability"])
        y_test = test_data["Potability"]

        predictions = loaded_model.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="binary", zero_division=1)
        recall = recall_score(y_test, predictions, average="binary", zero_division=1)
        f1 = f1_score(y_test, predictions, average="binary", zero_division=1)

        print(f"📊 Model Performance: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

        # Set thresholds based on model expectations
        self.assertGreaterEqual(accuracy, 0.3, "Accuracy is below the threshold (0.3).")
        self.assertGreaterEqual(precision, 0.3, "Precision is below the threshold (0.3).")
        self.assertGreaterEqual(recall, 0.3, "Recall is below the threshold (0.3).")
        self.assertGreaterEqual(f1, 0.3, "F1 score is below the threshold (0.3).")


if __name__ == "__main__":
    unittest.main()
