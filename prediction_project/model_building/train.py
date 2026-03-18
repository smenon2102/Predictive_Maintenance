from datasets import load_dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import pandas as pd
import joblib
import os

data_repo_id = "avatar2102/engine-predictive-maintenance"
model_repo_id = "avatar2102/engine-predictive-maintenance-model"
token = os.getenv("PREDICTIVE_GIT_TOKEN")

if token is None:
    raise ValueError("PREDICTIVE_GIT_TOKEN environment variable not set")

# Load train and test datasets from Hugging Face
train_df = load_dataset(data_repo_id, data_files="train.csv")["train"].to_pandas()
test_df = load_dataset(data_repo_id, data_files="test.csv")["train"].to_pandas()

print("Train and test datasets loaded successfully from Hugging Face.")

# Split features and target
X_train = train_df.drop("engine_condition", axis=1)
y_train = train_df["engine_condition"]

X_test = test_df.drop("engine_condition", axis=1)
y_test = test_df["engine_condition"]

print("Feature-target split completed.")

# Final AdaBoost model using already tuned best parameters
final_model = AdaBoostClassifier(
    n_estimators=150,
    learning_rate=0.05,
    random_state=42
)

# Train model
final_model.fit(X_train, y_train)
print("Final AdaBoost model trained successfully.")

# Predict on test data
y_pred = final_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Model Evaluation Completed.")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(cm)

# Save experiment log
log_df = pd.DataFrame([{
    "model": "AdaBoost",
    "n_estimators": 150,
    "learning_rate": 0.05,
    "cv_f1_score": 0.7742989393943112,
    "test_accuracy": accuracy,
    "test_precision": precision,
    "test_recall": recall,
    "test_f1_score": f1,
    "confusion_matrix": str(cm.tolist())
}])

log_df.to_csv("prediction_project/model_building/final_adaboost_model_log.csv", index=False)
print("Experiment log saved successfully.")

# Save model locally
joblib.dump(final_model, "prediction_project/model_building/adaboost_final_model.joblib")
print("Model saved locally as joblib.")

# Create model repo if needed
api = HfApi(token=token)

try:
    api.repo_info(repo_id=model_repo_id, repo_type="model")
    print(f"Model repo '{model_repo_id}' already exists. Using it.")
except (RepositoryNotFoundError, HfHubHTTPError):
    print(f"Model repo '{model_repo_id}' not found. Creating new repo...")
    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
    print(f"Model repo '{model_repo_id}' created.")

# Upload model_building folder to Hugging Face Model Hub
api.upload_folder(
    folder_path="prediction_project/model_building",
    repo_id=model_repo_id,
    repo_type="model",
    commit_message="Upload final AdaBoost model and experiment log"
)

print("Model uploaded successfully to Hugging Face Model Hub.")
