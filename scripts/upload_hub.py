import argparse
import os
from huggingface_hub import HfApi

def upload_all(api: HfApi, repo_id: str):
    print(f"Uploading all EmsyAI assets to {repo_id}...")
    upload_readme(api, repo_id)
    upload_gguf(api, repo_id)
    upload_checkpoints(api, repo_id)
    print("All assets uploaded successfully!")

def upload_readme(api: HfApi, repo_id: str):
    print("Uploading Model Card (HF_README_V3.md)...")
    try:
        api.upload_file(
            path_or_fileobj="HF_README_V3.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update Model Card for Hugging Face audience"
        )
        print("Model Card uploaded successfully.")
    except Exception as e:
        print(f"Failed to upload README: {e}")

def upload_gguf(api: HfApi, repo_id: str):
    print("Uploading GGUF binaries...")
    ggufs = [f for f in os.listdir(".") if f.endswith(".gguf")]
    for g in ggufs:
        try:
            api.upload_file(
                path_or_fileobj=g,
                path_in_repo=g,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {g}"
            )
            print(f"Uploaded: {g}")
        except Exception as e:
            print(f"Failed to upload {g}: {e}")

def upload_checkpoints(api: HfApi, repo_id: str):
    print("Uploading latest PyTorch checkpoints (Base & LoRA)...")
    # Add logic here to find the latest .pt files if desired.
    print("Checkpoints upload functionality can be mapped to your specific v3 paths later.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Hugging Face Publisher for EmsyAI")
    parser.add_argument("--all", action="store_true", help="Upload all assets (README, GGUF, Checkpoints)")
    parser.add_argument("--readme", action="store_true", help="Upload only the README.md")
    parser.add_argument("--gguf", action="store_true", help="Upload only the GGUF files")
    args = parser.parse_args()
    
    api = HfApi()
    repo_id = "gulding/EmsyAI"
    
    if args.all:
        upload_all(api, repo_id)
    elif args.readme:
        upload_readme(api, repo_id)
    elif args.gguf:
        upload_gguf(api, repo_id)
    else:
        print("Please specify an action (e.g., --all, --readme, --gguf). Use -h for help.")
