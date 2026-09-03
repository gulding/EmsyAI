import json
import os
from datasets import load_dataset
from emsyai.tokenizer import BPETokenizer

def get_ngrams(tokens, n=13):
    """Extract all n-grams from a list of tokens."""
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def main():
    print("Loading BPETokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load("dataset/tokenizer_v2.json")

    print("Fetching HumanEval dataset...")
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    
    N = 13
    contaminant_ngrams = set()
    
    for task in humaneval:
        full_text = task["prompt"] + task["canonical_solution"]
        tokens = tokenizer.encode(full_text)
        contaminant_ngrams.update(get_ngrams(tokens, n=N))
        
    print(f"Tracking {len(contaminant_ngrams):,} unique {N}-gram sequences from HumanEval.")

    instruct_path = "dataset/instruct.jsonl"
    if not os.path.exists(instruct_path):
        print(f"Error: {instruct_path} not found.")
        return

    print(f"\nScanning {instruct_path} for contamination...")
    overlap_count = 0
    
    with open(instruct_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            full_text = item["instruction"] + "\n" + item["output"]
            tokens = tokenizer.encode(full_text)
            
            if len(tokens) < N:
                continue
                
            example_ngrams = get_ngrams(tokens, n=N)
            overlaps = example_ngrams.intersection(contaminant_ngrams)
            
            if overlaps:
                overlap_count += len(overlaps)
                print(f"\n[WARNING] Found {len(overlaps)} overlapping N-grams in instruction #{i}!")
                sample_ngram = list(overlaps)[0]
                print(f"Leaked text snippet: {repr(tokenizer.decode(list(sample_ngram)))}")
            
    if overlap_count == 0:
        print("\n[PASSED] Zero HumanEval contamination detected in the instruction dataset!")
    else:
        print(f"\n[FAILED] Found {overlap_count} overlapping sequences. Instruction dataset is contaminated.")

if __name__ == "__main__":
    main()
