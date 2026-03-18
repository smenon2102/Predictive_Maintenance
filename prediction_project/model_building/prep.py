from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import pandas as pd
import os

repo_id = "avatar2102/engine-predictive-maintenance"
token = os.getenv("PREDICTIVE_GIT_TOKEN")

if token is None:
    raise ValueError("PREDICTIVE_GIT_TOKEN environment variable not set")

# Load dataset from Hugging Face
dataset = load_dataset(repo_id)
df = dataset["train"].to_pandas()

print("Dataset loaded successfully from Hugging Face.")
print("Shape:", df.shape)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Basic cleaning
df = df.drop_duplicates()
df = df.dropna()

print("Data cleaning completed.")
print("Shape after cleaning:", df.shape)

# Train-test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["engine_condition"]
)

# Save locally
train_path = "prediction_project/data/train.csv"
test_path = "prediction_project/data/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Train and test datasets saved locally.")

# Upload processed files back to Hugging Face
api = HfApi(token=token)

api.upload_folder(
    folder_path="prediction_project/data",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload processed train and test datasets"
)

print("Train and test datasets uploaded successfully to Hugging Face.")
