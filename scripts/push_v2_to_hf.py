from huggingface_hub import HfApi
import os

def push_v2():
    api = HfApi()
    repo_id = "gulding/EmsyAI"
    
    print("Uploading tokenizer_v2.json...")
    if os.path.exists("dataset/tokenizer_v2.json"):
        api.upload_file(
            path_or_fileobj="dataset/tokenizer_v2.json",
            path_in_repo="tokenizer_v2.json",
            repo_id=repo_id,
            repo_type="model"
        )
        print("Uploaded tokenizer!")
        
    print("Uploading EmsyAI-120M Base Checkpoint...")
    if os.path.exists("checkpoints_v2/model_step_5000.pt"):
        api.upload_file(
            path_or_fileobj="checkpoints_v2/model_step_5000.pt",
            path_in_repo="base_model_120M_step_5000.pt",
            repo_id=repo_id,
            repo_type="model"
        )
        print("Uploaded base model!")

if __name__ == "__main__":
    push_v2()
