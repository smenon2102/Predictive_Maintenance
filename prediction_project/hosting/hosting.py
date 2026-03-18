from huggingface_hub import HfApi
import os

SPACE_REPO_ID = "avatar2102/engine-predictive-maintenance-app"
REPO_TYPE = "space"

token = os.getenv("PREDICTIVE_GIT_TOKEN")
if token is None:
    raise ValueError("PREDICTIVE_GIT_TOKEN environment variable not set")

api = HfApi(token=token)

api.upload_folder(
    folder_path="prediction_project/deployment",
    repo_id=SPACE_REPO_ID,
    repo_type=REPO_TYPE,
    path_in_repo="",
    commit_message="Upload deployment files to Hugging Face Space"
)

print(f"Deployment files uploaded successfully to HF Space: {SPACE_REPO_ID}")
