import os
from huggingface_hub import HfApi

def push_readme():
    api = HfApi()
    repo_id = "gulding/EmsyAI"
    
    # Read the actual local README.md
    with open("README.md", "r", encoding="utf-8") as f:
        repo_readme = f.read()
    
    # Prepend the HuggingFace YAML metadata
    hf_metadata = """---
license: mit
language:
- en
tags:
- from-scratch
- pytorch
- educational
- lora
- code-generation
---

"""
    hf_readme_content = hf_metadata + repo_readme
    
    # Save it to a temporary file
    with open("HF_README.md", "w", encoding="utf-8") as f:
        f.write(hf_readme_content)
        
    print(f"Uploading README.md to {repo_id}...")
    api.upload_file(
        path_or_fileobj="HF_README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    print("Successfully uploaded the Model Card to HuggingFace!")

if __name__ == "__main__":
    push_readme()
