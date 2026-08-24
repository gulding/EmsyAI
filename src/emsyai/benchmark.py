import ast
import torch
from emsyai.chat import load_model
from emsyai.model.generate import generate

SIGNATURES = [
    "def add(a, b):",
    "def factorial(n):",
    "def is_prime(n):",
    "def reverse_string(s):",
    "def get_even_numbers(lst):",
    "class User:",
    "def parse_json(data):",
    "def connect_to_db(url):",
    "def calculate_area(radius):",
    "def find_max(numbers):",
    "def sort_dictionary_by_value(d):",
    "def flatten_list(nested_list):",
    "def check_palindrome(s):",
    "def fibonacci(n):",
    "def merge_dicts(d1, d2):",
    "class LinkedListNode:",
    "def send_email(to, subject, body):",
    "def count_words(text):",
    "def binary_search(arr, target):",
    "def random_password_generator(length):"
]

def check_syntax(code: str) -> bool:
    """
    Attempts to parse the generated string into a Python Abstract Syntax Tree (AST).
    If it parses without throwing a SyntaxError, the model generated valid Python syntax.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "checkpoints/model_step_5000.pt"
    tokenizer_path = "dataset/tokenizer.json"
    
    print(f"Loading EmsyAI for benchmarking on {device}...")
    model, tokenizer = load_model(checkpoint, tokenizer_path, device)
    
    valid_count = 0
    total = len(SIGNATURES)
    
    print("\n" + "="*50)
    print("Starting Syntax Benchmark")
    print("="*50)
    
    for i, sig in enumerate(SIGNATURES):
        print(f"\n[{i+1}/{total}] Prompt: {sig}")
        
        # We use a low temperature to make the model more deterministic and grammar-focused
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=sig + "\n",
            max_new_tokens=50,
            temperature=0.3,
            device=device
        )
        
        # Check syntax
        is_valid = check_syntax(output)
        if is_valid:
            valid_count += 1
            print("Status: [PASS] Valid Python Syntax")
        else:
            print("Status: [FAIL] Syntax Error")
            
        print("Generated Code:")
        print("-" * 40)
        print(output)
        print("-" * 40)
        
    score = (valid_count / total) * 100
    
    print("\n" + "="*50)
    print("Benchmark Results")
    print("="*50)
    print(f"Syntax Pass Rate: {valid_count}/{total} ({score:.1f}%)")
    print("\nNote: A passing score only means the code compiles in Python, not that the logic is correct!")

if __name__ == "__main__":
    run_benchmark()
