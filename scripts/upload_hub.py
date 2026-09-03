import argparse
import os
from huggingface_hub import HfApi

def upload_all(api: HfApi, repo_id: str):
    print(f"Uploading all EmsyAI assets to {repo_id}...")
    upload_readme(api, repo_id)
    upload_gguf(api, repo_id)
    print("All assets uploaded successfully!")

def upload_readme(api: HfApi, repo_id: str):
    print("Uploading README and images...")
    try:
        # Dynamically inject the YAML frontmatter for Hugging Face
        yaml_frontmatter = """---
license: mit
pipeline_tag: text-generation
language:
- en
tags:
- gguf
- llama.cpp
- ollama
- from-scratch
- educational
- pytorch
- lora
---

"""
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
            
        with open(".HF_README_TEMP.md", "w", encoding="utf-8") as f:
            f.write(yaml_frontmatter + readme_content)

        api.upload_file(
            path_or_fileobj=".HF_README_TEMP.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update V4 Model Card"
        )
        os.remove(".HF_README_TEMP.md")
        print("README.md uploaded successfully.")
        
        # Upload the training curve so it renders in the README on HF
        if os.path.exists("v4_training_curve.png"):
            api.upload_file(
                path_or_fileobj="v4_training_curve.png",
                path_in_repo="v4_training_curve.png",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add V4 training curve"
            )
            print("v4_training_curve.png uploaded successfully.")
    except Exception as e:
        print(f"Failed to upload documentation: {e}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Hugging Face Publisher for EmsyAI")
    parser.add_argument("--all", action="store_true", help="Upload all assets (README, Image, GGUF)")
    parser.add_argument("--readme", action="store_true", help="Upload only the README.md and images")
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
