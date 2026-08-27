import re
from tqdm import tqdm

EXERCISM_PROMPTS = [
    "Given a year, report if it is a leap year",
    "Bob is a lackadaisical teenager",
    "Calculate the moment when someone has lived for 10^9 seconds",
    "Determine if a sentence is a pangram",
    "Calculate how old someone would be on",
    "Determine if a word or phrase is an isogram",
    "Convert a number to a string, the contents of which depend on the number's factors",
    "Given an age in seconds, calculate how old someone would be on",
    "One for you, one for me",
    "Manage robot factory settings"
]

def check_file(filename):
    print(f"\nScanning {filename} for Exercism contamination...")
    hits = {prompt: 0 for prompt in EXERCISM_PROMPTS}
    total_lines = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            total_lines += 1
            lower_line = line.lower()
            for prompt in EXERCISM_PROMPTS:
                if prompt.lower() in lower_line:
                    hits[prompt] += 1
                    
    total_hits = sum(hits.values())
    print(f"\nResults for {filename} (Total lines: {total_lines}):")
    if total_hits == 0:
        print("CLEAN! No direct Exercism prompts found.")
    else:
        print(f"CONTAMINATED! Found {total_hits} matches.")
        for prompt, count in hits.items():
            if count > 0:
                print(f"  - '{prompt}': {count} hits")

if __name__ == "__main__":
    import os
    if os.path.exists("dataset/smollm_corpus_v3.txt"):
        check_file("dataset/smollm_corpus_v3.txt")
    if os.path.exists("dataset/instruct.jsonl"):
        check_file("dataset/instruct.jsonl")
