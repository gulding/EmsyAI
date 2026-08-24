from huggingface_hub import HfApi, login
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, help="HuggingFace token (or run huggingface-cli login first)")
    args = parser.parse_args()
    
    if args.token:
        login(token=args.token)
        
    api = HfApi()
    repo_id = "gulding/EmsyAI"
    
    print(f"Uploading checkpoints to {repo_id}...")
    
    # Upload checkpoints
    if os.path.exists("checkpoints"):
        api.upload_folder(
            folder_path="checkpoints",
            path_in_repo="checkpoints",
            repo_id=repo_id,
            repo_type="model"
        )
        print("Uploaded checkpoints!")
        
    # Upload tokenizer
    if os.path.exists("dataset/tokenizer.json"):
        api.upload_file(
            path_or_fileobj="dataset/tokenizer.json",
            path_in_repo="tokenizer.json",
            repo_id=repo_id,
            repo_type="model"
        )
        print("Uploaded tokenizer.json!")

    print("Successfully pushed to HuggingFace!")

if __name__ == "__main__":
    main()
