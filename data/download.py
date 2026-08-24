import os
import requests
import zipfile
import io
from pathlib import Path
from tqdm import tqdm

def download_cpython_dataset(target_dir="dataset"):
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    out_file = os.path.join(target_dir, "train.txt")
    
    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping download.")
        return
        
    print("Downloading CPython repository (this may take a moment)...")
    url = "https://github.com/python/cpython/archive/refs/heads/main.zip"
    response = requests.get(url)
    response.raise_for_status()
    
    print("Extracting Python files...")
    python_code = []
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        py_files = [f for f in z.namelist() if f.endswith('.py')]
        for file_path in tqdm(py_files, desc="Processing files"):
            try:
                content = z.read(file_path).decode('utf-8')
                python_code.append(content)
            except Exception:
                continue
                
    full_text = "\n\n".join(python_code)
    size_mb = len(full_text.encode('utf-8')) / (1024 * 1024)
    print(f"Collected {size_mb:.2f} MB of Python code.")
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"Dataset saved to {out_file}")

if __name__ == "__main__":
    download_cpython_dataset()
