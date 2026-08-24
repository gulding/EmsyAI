from huggingface_hub import HfApi
import os

def push_gguf():
    api = HfApi()
    repo_id = "gulding/EmsyAI"
    
    # Upload GGUF
    if os.path.exists("emsyai-f32.gguf"):
        print(f"Uploading emsyai-f32.gguf to {repo_id}...")
        api.upload_file(
            path_or_fileobj="emsyai-f32.gguf",
            path_in_repo="emsyai-f32.gguf",
            repo_id=repo_id,
            repo_type="model"
        )
        print("Uploaded GGUF!")
        
    # Upload Modelfile for Ollama
    modelfile_content = """FROM ./emsyai-f32.gguf
TEMPLATE \"\"\"[USER]
{{ .Prompt }}
[MODEL]
\"\"\"
PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2"""
    
    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write(modelfile_content)
        
    print("Uploading Modelfile...")
    api.upload_file(
        path_or_fileobj="Modelfile",
        path_in_repo="Modelfile",
        repo_id=repo_id,
        repo_type="model"
    )
    print("Uploaded Modelfile! Your model is now natively compatible with Ollama!")

if __name__ == "__main__":
    push_gguf()
