import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def stream_and_save(dataset_name, config_name, split, max_examples, output_file, text_column="text"):
    print(f"Streaming {max_examples} examples from {dataset_name} ({config_name})...")
    
    # We use streaming=True so we don't have to download the multi-terabyte dataset
    dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # We write it to a text file for our BPE tokenizer to train on, 
    # and eventually for our pretraining data prep script.
    with open(output_file, "a", encoding="utf-8") as f:
        count = 0
        for item in tqdm(dataset, total=max_examples):
            if count >= max_examples:
                break
                
            text = item[text_column]
            if text:
                f.write(text + "\n<|eos|>\n")
                count += 1
                
    print(f"Saved {count} examples to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    # 50,000 examples of Cosmopedia (high quality textbook English)
    parser.add_argument("--num_english", type=int, default=50000)
    # 50,000 examples of Python Edu (high quality Python code)
    parser.add_argument("--num_python", type=int, default=50000)
    args = parser.parse_args()

    output_file = "dataset/smollm_corpus_v2.txt"
    
    # Clear the file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print("=== Phase 6: Data Engine ===")
    
    # 1. Download English Reasoning Data (Cosmopedia v2)
    stream_and_save(
        dataset_name="HuggingFaceTB/smollm-corpus", 
        config_name="cosmopedia-v2", 
        split="train", 
        max_examples=args.num_english, 
        output_file=output_file,
        text_column="text"
    )
    
    # 2. Download Python Data (python-codes-25k)
    stream_and_save(
        dataset_name="flytech/python-codes-25k", 
        config_name=None, 
        split="train", 
        max_examples=args.num_python, 
        output_file=output_file,
        text_column="text"
    )
    
    print("\nPhase 6 Data Download Complete!")
    print(f"Next Step: Retrain the tokenizer to 16K vocab on {output_file}")

if __name__ == "__main__":
    main()
