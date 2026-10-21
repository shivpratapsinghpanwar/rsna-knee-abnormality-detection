"""Upload D:\\kaggle_c\\hf_upload to a private Hugging Face model repo (resumable).

usage: upload_hf.py <repo_name>
"""
import sys

from huggingface_hub import HfApi

api = HfApi()
user = api.whoami()["name"]
repo = f"{user}/{sys.argv[1]}"
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
# upload_large_folder resumes after an interruption and splits work into many small commits.
api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=r"D:\kaggle_c\hf_upload",
                        num_workers=4)
files = api.list_repo_files(repo)
print(f"UPLOAD_DONE https://huggingface.co/{repo}  ({len(files)} files, private)")
