from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

repo_id = "avatar2102/engine-predictive-maintenance"
repo_type = "dataset"

token = os.getenv("PREDICTIVE_GIT_TOKEN")
if token is None:
    raise ValueError("PREDICTIVE_GIT_TOKEN environment variable not set")

api = HfApi(token=token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except (RepositoryNotFoundError, HfHubHTTPError):
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    print(f"Dataset repo '{repo_id}' created.")

api.upload_folder(
    folder_path="prediction_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Add engine predictive maintenance dataset"
)

print("Dataset uploaded successfully.")
