import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from emsyai.tokenizer import BPETokenizer

class PretokDataset(Dataset):
    """
    Dataset that loads pre-tokenized chunks.
    We pack tokens into fixed-length sequences. For example, if we have a stream of tokens,
    we just take chunks of size `seq_len + 1` (since we need the +1 for the next-token target).
    """
    def __init__(self, data_path: str, seq_len: int = 512):
        # We store tokens as uint16 since our vocab size is 8000 (< 65535)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        # Number of possible sequences we can extract
        self.num_samples = len(self.data) // (seq_len + 1)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Slice a chunk of seq_len + 1 tokens
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        
        # x is the input sequence, y is the target sequence (shifted by 1)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def prepare_dataset(text_path: str, tokenizer_path: str, out_path: str):
    """
    Reads the raw text, tokenizes it, and saves it as a binary file of uint16 tokens.
    This prevents us from having to tokenize the entire dataset every time we start training.
    """
    if os.path.exists(out_path):
        print(f"Dataset already pre-tokenized at {out_path}")
        return
        
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"Loading text from {text_path}...")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print("Tokenizing text (this might take a few minutes with our pure Python tokenizer)...")
    # For a large dataset, we process in chunks to save memory and show progress
    chunk_size = 1024 * 1024 # 1MB chunks
    tokens = []
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        ids = tokenizer.encode(chunk)
        tokens.extend(ids)
        print(f"Tokenized {min(i + chunk_size, len(text))} / {len(text)} bytes")
        
    print(f"Total tokens: {len(tokens)}")
    
    print(f"Saving to {out_path} as uint16 binary...")
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(out_path)
    print("Done!")

def get_dataloaders(
    data_path: str, 
    seq_len: int = 512, 
    batch_size: int = 4,
    val_split: float = 0.05
):
    """
    Creates PyTorch DataLoaders for training and validation.
    """
    dataset = PretokDataset(data_path, seq_len)
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # We use a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader
