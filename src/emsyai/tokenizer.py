import re
import json
from collections import defaultdict
from typing import List, Dict, Tuple

# Simple regex to split text into words, preventing merges across boundaries (like spaces and punctuation)
# This matches contractions, alphabetical sequences, numbers, or sequences of punctuation, plus isolated spaces.
TOKEN_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z_]+| ?[0-9]+| ?[^\sA-Za-z0-9_]+|\s+(?!\S)|\s+""")

class BPETokenizer:
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        
        # Special tokens
        self.special_tokens = {
            "<|pad|>": 0,
            "<|bos|>": 1,
            "<|eos|>": 2,
            "<|unk|>": 3,
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Initialize vocab with 256 byte values
        self._init_byte_vocab()
        
    def _init_byte_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Next token ID will start after special tokens and 256 bytes
        self.next_id = 256 + len(self.special_tokens)

    def train(self, text: str):
        """
        Train the BPE tokenizer on a string of text.
        """
        print("Splitting text into words...")
        words = TOKEN_PATTERN.findall(text)
        
        # Count word frequencies to speed up training
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
            
        # Convert words to lists of byte integers
        # e.g., " hello" -> [32, 104, 101, 108, 108, 111]
        splits = {tuple(word.encode('utf-8')): count for word, count in word_counts.items()}
        
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        
        print(f"Starting BPE training for {num_merges} merges...")
        for i in range(num_merges):
            # Count pair frequencies
            pair_counts = defaultdict(int)
            for split, count in splits.items():
                if len(split) < 2:
                    continue
                for j in range(len(split) - 1):
                    pair = (split[j], split[j+1])
                    pair_counts[pair] += count
            
            if not pair_counts:
                break
                
            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            
            # Record merge
            new_id = self.next_id
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.next_id += 1
            
            # Apply merge to all words
            new_splits = {}
            for split, count in splits.items():
                if len(split) < 2:
                    new_splits[split] = count
                    continue
                
                new_split = []
                j = 0
                while j < len(split):
                    if j < len(split) - 1 and (split[j], split[j+1]) == best_pair:
                        new_split.append(new_id)
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                new_splits[tuple(new_split)] = count
                
            splits = new_splits
            
            if (i + 1) % 500 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_id} (count: {pair_counts[best_pair]})")
                
        print("Training complete.")

    def encode(self, text: str, allowed_special: set = None) -> List[int]:
        """Convert string to token IDs, natively handling special tokens."""
        if allowed_special is None:
            allowed_special = set()
            
        ids = []
        # Find all special tokens in the text if allowed
        # To avoid complex regex, we can just find them and split the string
        
        # Simple approach for a limited set of special tokens:
        # If the string contains a special token we allow, we should parse it.
        # But a more robust way is to just do a string replace or split.
        
        # For our simple tokenizer, let's assume if it exactly matches a special token string, it is one.
        # But since they can be embedded in text, let's build a regex for the allowed special tokens.
        if allowed_special:
            escaped_special = [re.escape(s) for s in allowed_special if s in self.special_tokens]
            if escaped_special:
                special_pattern = re.compile("(" + "|".join(escaped_special) + ")")
                chunks = special_pattern.split(text)
            else:
                chunks = [text]
        else:
            chunks = [text]
            
        for chunk in chunks:
            if chunk in allowed_special and chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
                continue
                
            if not chunk:
                continue
                
            words = TOKEN_PATTERN.findall(chunk)
            for word in words:
                # Start with byte IDs
                split = list(word.encode('utf-8'))
                
                # Iteratively apply merges
                while len(split) >= 2:
                    min_pair = None
                    min_id = float('inf')
                    
                    for i in range(len(split) - 1):
                        pair = (split[i], split[i+1])
                        if pair in self.merges and self.merges[pair] < min_id:
                            min_id = self.merges[pair]
                            min_pair = pair
                            
                    if min_pair is None:
                        break # No more merges possible
                        
                    # Apply the specific merge sequentially across the word
                    new_split = []
                    i = 0
                    while i < len(split):
                        if i < len(split) - 1 and (split[i], split[i+1]) == min_pair:
                            new_split.append(min_id)
                            i += 2
                        else:
                            new_split.append(split[i])
                            i += 1
                    split = new_split
                    
                ids.extend(split)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to string"""
        byte_list = []
        for id in ids:
            if id in self.inverse_special_tokens:
                byte_list.extend(self.inverse_special_tokens[id].encode('utf-8'))
            elif id in self.vocab:
                byte_list.extend(self.vocab[id])
            else:
                byte_list.extend(self.inverse_special_tokens[self.special_tokens["<|unk|>"]].encode('utf-8'))
        
        return bytes(byte_list).decode('utf-8', errors='replace')
        
    def save(self, path: str):
        """Save vocabulary and merges to disk"""
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "special_tokens": self.special_tokens
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    def load(self, path: str):
        """Load vocabulary and merges from disk"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        self.merges = {}
        for k_str, v in data["merges"].items():
            p1, p2 = k_str.split(",")
            self.merges[(int(p1), int(p2))] = int(v)
            
        # Reconstruct vocab
        self._init_byte_vocab()
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for pair, new_id in sorted_merges:
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.next_id = max(self.next_id, new_id + 1)
