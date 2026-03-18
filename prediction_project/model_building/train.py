import os
import pandas as pd
import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

DATASET_REPO_ID = "avatar2102/engine-predictive-maintenance"
MODEL_REPO_ID = "avatar2102/engine-predictive-maintenance-model"

token = os.getenv("PREDICTIVE_GIT_TOKEN")
if token is None:
    raise ValueError("PREDICTIVE_GIT_TOKEN environment variable not set")

api = HfApi(token=token)

# Load train/test data from Hugging Face dataset repo
train_path = f"hf://datasets/{DATASET_REPO_ID}/train.csv"
test_path = f"hf://datasets/{DATASET_REPO_ID}/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train and test datasets loaded successfully from Hugging Face.")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Split features and target
X_train = train_df.drop("engine_condition", axis=1)
y_train = train_df["engine_condition"]

X_test = test_df.drop("engine_condition", axis=1)
y_test = test_df["engine_condition"]

print("Feature-target split completed.")

# Final AdaBoost model using tuned parameters from interim phase
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

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Model evaluation completed.")
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

# Create model repo if missing
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
    print(f"Model repo '{MODEL_REPO_ID}' already exists. Using it.")
except (RepositoryNotFoundError, HfHubHTTPError):
    print(f"Model repo '{MODEL_REPO_ID}' not found. Creating new repo...")
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
    print(f"Model repo '{MODEL_REPO_ID}' created.")

# Upload model_building folder to HF Model Hub
api.upload_folder(
    folder_path="prediction_project/model_building",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    commit_message="Upload final AdaBoost model and experiment log"
)

print("Model uploaded successfully to Hugging Face Model Hub.")
