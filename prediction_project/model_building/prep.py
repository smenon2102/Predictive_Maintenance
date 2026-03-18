%%writefile prediction_project/model_building/prep.py
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
import pandas as pd
import os

repo_id = "avatar2102/engine-predictive-maintenance"
token = os.getenv("PREDICTIVE_GIT_TOKEN")

if token is None:
    raise ValueError("PREDICTIVE_GIT_TOKEN environment variable not set")

api = HfApi(token=token)

# Load dataset directly from Hugging Face using pandas
data_path = f"hf://datasets/{repo_id}/engine_data.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully from Hugging Face.")
print("Original shape:", df.shape)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Basic cleaning
df = df.drop_duplicates()
df = df.dropna()

print("Data cleaning completed.")
print("Shape after cleaning:", df.shape)

# Split into train and test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["engine_condition"]
)

# Save locally
train_df.to_csv("prediction_project/data/train.csv", index=False)
test_df.to_csv("prediction_project/data/test.csv", index=False)

print("Train and test datasets saved locally.")

# Upload updated data folder back to Hugging Face
api.upload_folder(
    folder_path="prediction_project/data",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload processed train and test datasets"
)

print("Train and test datasets uploaded successfully to Hugging Face.")
