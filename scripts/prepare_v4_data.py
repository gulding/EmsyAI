import os
import numpy as np
import multiprocessing as mp
from datasets import load_dataset
from emsyai.tokenizer import BPETokenizer
from tqdm import tqdm

# Globals for worker processes
_tokenizer = None
_contaminants = None
N_GRAM_SIZE = 13

def get_ngrams(tokens, n=13):
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def init_worker():
    global _tokenizer, _contaminants
    _tokenizer = BPETokenizer()
    _tokenizer.load("dataset/tokenizer_v2.json")
    
    # Load HumanEval once per worker to build the registry
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    _contaminants = set()
    for task in humaneval:
        full_text = task["prompt"] + task["canonical_solution"]
        _contaminants.update(get_ngrams(_tokenizer.encode(full_text), n=N_GRAM_SIZE))

def process_document(text):
    global _tokenizer, _contaminants
    
    tokens = _tokenizer.encode(text)
    
    # Check for contamination
    doc_ngrams = get_ngrams(tokens, n=N_GRAM_SIZE)
    if doc_ngrams.intersection(_contaminants):
        # Document is contaminated! Return empty to discard it.
        return b"", 0
        
    # Document is clean, add EOS and convert to bytes
    tokens.append(_tokenizer.vocab.get("<|eos|>", 2))
    arr = np.array(tokens, dtype=np.uint16)
    return arr.tobytes(), len(tokens)

def main():
    print("Loading Dataset Stream...")
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    TARGET_TOKENS = 500_000_000
    out_file = "dataset/train_v4.bin"
    
    print(f"Targeting {TARGET_TOKENS:,} tokens.")
    print(f"Using {mp.cpu_count()} CPU cores for tokenization...")
    
    tokens_written = 0
    
    # Open file in binary append/write mode
    with open(out_file, "wb") as f, mp.Pool(mp.cpu_count(), initializer=init_worker) as pool:
        pbar = tqdm(total=TARGET_TOKENS, desc="Tokenizing", unit="tok")
        
        # Buffer to hold text before sending to pool
        batch_size = 2000
        text_buffer = []
        
        for doc in dataset:
            text_buffer.append(doc["text"])
            
            if len(text_buffer) >= batch_size:
                # Map across processes
                results = pool.map(process_document, text_buffer)
                
                for byte_data, tok_count in results:
                    f.write(byte_data)
                    tokens_written += tok_count
                    pbar.update(tok_count)
                    
                text_buffer = []
                
                if tokens_written >= TARGET_TOKENS:
                    break
        
        # Process remaining buffer
        if text_buffer and tokens_written < TARGET_TOKENS:
            results = pool.map(process_document, text_buffer)
            for byte_data, tok_count in results:
                f.write(byte_data)
                tokens_written += tok_count
                pbar.update(tok_count)
                if tokens_written >= TARGET_TOKENS:
                    break
                    
        pbar.close()

    print(f"\nDone! Wrote {tokens_written:,} tokens to {out_file}")
    print(f"File size: {os.path.getsize(out_file) / (1024*1024*1024):.2f} GB")

if __name__ == "__main__":
    main()
