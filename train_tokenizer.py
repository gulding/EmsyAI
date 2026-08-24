import os
from emsyai.tokenizer import BPETokenizer

def main():
    dataset_path = "dataset/train.txt"
    model_path = "dataset/tokenizer.json"
    # Phase 6: Use the new hybrid dataset
    data_path = "dataset/smollm_corpus_v2.txt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data/download_v2.py first.")
        return
        
    print(f"Reading {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Scale to 16,000 vocab size as per the Phase 6 Roadmap
    vocab_size = 16000
    print(f"Text length: {len(text)} characters")
    print(f"Training BPE tokenizer (vocab size {vocab_size})...")
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)
    
    out_path = "dataset/tokenizer.json"
    tokenizer.save(out_path)
    print(f"Tokenizer saved to {out_path}")
    
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
