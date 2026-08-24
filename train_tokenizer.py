import os
from emsyai.tokenizer import BPETokenizer

def main():
    dataset_path = "dataset/train.txt"
    model_path = "dataset/tokenizer.json"
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run data/download.py first.")
        return
        
    print("Loading dataset...")
    # Load first 10MB of the dataset to keep training time reasonable for pure Python
    # 10MB is about 10 million characters, enough to learn a solid 8000 vocab for Python
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read(10 * 1024 * 1024) 
        
    print(f"Loaded {len(text)} characters for tokenizer training.")
    
    tokenizer = BPETokenizer(vocab_size=8000)
    tokenizer.train(text)
    
    print(f"Saving tokenizer to {model_path}...")
    tokenizer.save(model_path)
    
    # Test it
    test_text = "def hello_world():\n    print('Hello, world!')"
    print(f"\nTesting tokenizer on: {test_text!r}")
    ids = tokenizer.encode(test_text)
    print(f"Encoded IDs: {ids}")
    decoded = tokenizer.decode(ids)
    print(f"Decoded text: {decoded!r}")
    assert test_text == decoded, "Decoded text does not match original!"
    print("Test passed! Tokenizer is working correctly.")

if __name__ == "__main__":
    main()
