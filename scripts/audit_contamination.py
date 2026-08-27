import os
import numpy as np
from datasets import load_dataset
from emsyai.tokenizer import BPETokenizer
from tqdm import tqdm

def get_ngrams(tokens, n=13):
    """Extract all n-grams from a list of tokens."""
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def main():
    print("Loading BPETokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load("dataset/tokenizer_v2.json")

    print("Fetching HumanEval dataset...")
    # Load HumanEval to get the canonical solutions (requires namespace in modern HF library)
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    
    print("Tokenizing HumanEval and building N-gram registry (N=13)...")
    N = 13 # 13 tokens is roughly 30-40 characters, a highly reliable exact-match threshold
    contaminant_ngrams = set()
    
    for task in humaneval:
        # Combine prompt and solution to form the full target string
        full_text = task["prompt"] + task["canonical_solution"]
        tokens = tokenizer.encode(full_text)
        contaminant_ngrams.update(get_ngrams(tokens, n=N))
        
    print(f"Tracking {len(contaminant_ngrams):,} unique {N}-gram sequences from HumanEval.")

    bin_path = "dataset/train_v4.bin"
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        return

    print(f"\nScanning {bin_path} for contamination...")
    # Load the 500M tokens via memory mapping
    mmap = np.memmap(bin_path, dtype=np.uint16, mode='r')
    total_tokens = len(mmap)
    
    # We scan in chunks to avoid converting the entire 500M array to tuples at once
    CHUNK_SIZE = 10_000_000
    overlap_count = 0
    
    for i in tqdm(range(0, total_tokens, CHUNK_SIZE)):
        end_idx = min(i + CHUNK_SIZE + N, total_tokens) # overlap chunks by N to not miss boundaries
        chunk = mmap[i:end_idx]
        
        # Fast n-gram extraction for the chunk
        chunk_list = chunk.tolist()
        chunk_ngrams = set(tuple(chunk_list[j:j+N]) for j in range(len(chunk_list) - N + 1))
        
        # Intersection
        overlaps = chunk_ngrams.intersection(contaminant_ngrams)
        if overlaps:
            overlap_count += len(overlaps)
            print(f"\n[WARNING] Found {len(overlaps)} overlapping N-grams in chunk starting at {i:,}!")
            # Print a sample of what leaked
            sample_ngram = list(overlaps)[0]
            print(f"Leaked text snippet: {repr(tokenizer.decode(list(sample_ngram)))}")
            
    if overlap_count == 0:
        print("\n[PASSED] Zero HumanEval contamination detected in the training dataset!")
    else:
        print(f"\n[FAILED] Found {overlap_count} overlapping sequences. Dataset is contaminated.")

if __name__ == "__main__":
    main()
