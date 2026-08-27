import os
from huggingface_hub import HfApi

def clean_and_update_repo():
    api = HfApi()
    repo_id = "gulding/EmsyAI"
    
    print("Fetching current files in repo...")
    try:
        current_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"Failed to fetch repo files: {e}")
        return
        
    print(f"Found {len(current_files)} files in repo.")
    
    files_to_delete = [
        # Intermediate base checkpoints
        "checkpoints/model_step_500.pt",
        "checkpoints/model_step_1000.pt",
        "checkpoints/model_step_1500.pt",
        "checkpoints/model_step_2000.pt",
        "checkpoints/model_step_2500.pt",
        "checkpoints/model_step_3000.pt",
        "checkpoints/model_step_3500.pt",
        "checkpoints/model_step_4000.pt",
        "checkpoints/model_step_4500.pt",
        "checkpoints/model_step_5000.pt", # Old v1 base model
        
        # Intermediate LoRA checkpoints
        "checkpoints/lora/instruct_lora_step_1000.pt",
        "checkpoints/lora/instruct_lora_step_2000.pt",
        "checkpoints/lora/instruct_lora_step_3000.pt",
        "checkpoints/lora/instruct_lora_step_4000.pt",
        "checkpoints/lora/instruct_lora_step_5000.pt",
        "checkpoints/lora/instruct_lora_step_6000.pt",
        "checkpoints/lora/instruct_lora_step_7000.pt",
        "checkpoints/lora/instruct_lora_step_8000.pt",
        "checkpoints/lora/instruct_lora_step_9000.pt",
        
        # Old v1 artifacts
        "emsyai-f32.gguf",
        "tokenizer.json",
    ]
    
    deleted_count = 0
    for f in files_to_delete:
        if f in current_files:
            try:
                api.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="model")
                print(f"Deleted: {f}")
                deleted_count += 1
            except Exception as e:
                print(f"Could not delete {f}: {e}")
                
    print(f"\nCleanup complete. Deleted {deleted_count} files.")
    
    print("\nUploading final v2 assets...")
    
    # Upload tokenizer_v2.json as tokenizer.json
    try:
        api.upload_file(
            path_or_fileobj="dataset/tokenizer_v2.json",
            path_in_repo="tokenizer.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update tokenizer to v2 16k vocab"
        )
        print("Uploaded: tokenizer.json (from dataset/tokenizer_v2.json)")
    except Exception as e:
        print(f"Failed to upload tokenizer.json: {e}")
        
    # Upload the new GGUF
    gguf_path = "emsyai-120m-instruct-f32.gguf"
    if os.path.exists(gguf_path):
        try:
            api.upload_file(
                path_or_fileobj=gguf_path,
                path_in_repo="emsyai-v2-instruct-f32.gguf",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload v2 88M GGUF model with tokenizer fix"
            )
            print("Uploaded: emsyai-v2-instruct-f32.gguf")
        except Exception as e:
            print(f"Failed to upload GGUF: {e}")
    else:
        print(f"Warning: Local GGUF file {gguf_path} not found!")

    # Ensure Modelfile is updated
    try:
        with open("Modelfile", "r", encoding="utf-8") as f:
            content = f.read()
        
        if "emsyai-120m-instruct-f32.gguf" in content:
            new_content = content.replace("emsyai-120m-instruct-f32.gguf", "emsyai-v2-instruct-f32.gguf")
            with open("Modelfile", "w", encoding="utf-8") as f:
                f.write(new_content)
            
            api.upload_file(
                path_or_fileobj="Modelfile",
                path_in_repo="Modelfile",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update Modelfile to point to v2 GGUF"
            )
            print("Updated and uploaded Modelfile.")
    except Exception as e:
        print(f"Failed to update Modelfile: {e}")
        
    print("\nRepository restructure is complete!")

if __name__ == "__main__":
    clean_and_update_repo()
