import json
from datasets import load_dataset
import os

def main():
    print("Downloading CodeAlpaca-20k from HuggingFace...")
    # This dataset contains ~20,000 instruction/output pairs for coding
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    
    # We will take all 20,000 examples
    # The dataset has 'instruction', 'input', and 'output' columns.
    
    formatted_data = []
    
    for row in dataset:
        instruction = row["instruction"]
        input_ctx = row["input"]
        output = row["output"]
        
        # Combine instruction and input context if it exists
        if input_ctx and input_ctx.strip():
            full_instruction = f"{instruction}\n\nContext:\n{input_ctx}"
        else:
            full_instruction = instruction
            
        formatted_data.append({
            "instruction": full_instruction,
            "output": output
        })
        
    os.makedirs("dataset", exist_ok=True)
    out_path = "dataset/instruct.jsonl"
    
    print(f"Formatting and saving {len(formatted_data)} examples to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done! You now have a real instruction tuning dataset.")

if __name__ == "__main__":
    main()
