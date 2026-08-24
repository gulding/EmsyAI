import os
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from emsyai.tokenizer import BPETokenizer

# Global tokenizer for multiprocessing
g_tok = None

def init_worker():
    global g_tok
    g_tok = BPETokenizer(16000)
    g_tok.load("dataset/tokenizer_v2.json")

def tokenize_chunk(text):
    ids = g_tok.encode(text, allowed_special={"<|eos|>"})
    ids.append(g_tok.special_tokens["<|eos|>"])
    return ids

def get_cosmopedia(num_samples=75000):
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    count = 0
    for row in ds:
        if count >= num_samples: break
        yield row["text"]
        count += 1

def get_python_codes(num_samples=75000):
    ds = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    count = 0
    for row in ds:
        if count >= num_samples: break
        yield row["instruction"] + "\n" + row["output"]
        count += 1

def get_synthetic_edits(num_samples=25000):
    # A lightweight dataset with code instructions to act as synthetic edit pairs
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    count = 0
    for row in ds:
        if count >= num_samples: break
        text = f"[USER]\n{row['instruction']}\n{row['input']}\n[MODEL]\n```python\n{row['output']}\n```"
        yield text
        count += 1

def build_dataset():
    out_bin = "dataset/train_v3.bin"
    out_txt = "dataset/smollm_corpus_v3.txt"
    
    print("Gathering texts...")
    texts = []
    
    for text in tqdm(get_cosmopedia(75000), total=75000, desc="Cosmopedia"):
        texts.append(text)
        
    for text in tqdm(get_python_codes(75000), total=75000, desc="Python Codes"):
        texts.append(text)
        
    for text in tqdm(get_synthetic_edits(25000), total=18800, desc="Synthetic Edits"):
        texts.append(text)
        
    print(f"Writing {len(texts)} documents to text file...")
    with open(out_txt, "w", encoding="utf-8") as f_txt:
        for t in texts:
            f_txt.write(t + "\n<|eos|>\n")
            
    print("Tokenizing (this may take a while, using multiprocessing)...")
    all_tokens = []
    
    # We use multiprocessing to parallelize the slow Python BPE tokenizer
    with Pool(initializer=init_worker, processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(tokenize_chunk, texts, chunksize=100), total=len(texts)))
        
    for r in results:
        all_tokens.extend(r)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    print("Saving to binary...")
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(out_bin)
    print(f"Saved {out_bin} ({os.path.getsize(out_bin)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    build_dataset()
