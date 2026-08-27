import os
import torch
import subprocess
import tempfile
import argparse
from emsyai.chat import load_model
from emsyai.model.generate import generate

# A mini "HumanEval" style test suite.
# The model receives the 'prompt', and its generation is concatenated with 'test'
TEST_CASES = [
    {
        "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b\"\"\"\n",
        "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('PASS')"
    },
    {
        "prompt": "def is_prime(n):\n    \"\"\"Return True if n is prime, else False\"\"\"\n",
        "test": "assert is_prime(2) == True\nassert is_prime(4) == False\nassert is_prime(17) == True\nassert is_prime(1) == False\nprint('PASS')"
    },
    {
        "prompt": "def reverse_string(s):\n    \"\"\"Return the reversed string\"\"\"\n",
        "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nprint('PASS')"
    },
    {
        "prompt": "def get_even_numbers(lst):\n    \"\"\"Return a list of only the even numbers\"\"\"\n",
        "test": "assert get_even_numbers([1, 2, 3, 4, 5]) == [2, 4]\nassert get_even_numbers([1, 3, 5]) == []\nprint('PASS')"
    },
    {
        "prompt": "def factorial(n):\n    \"\"\"Return the factorial of n (n >= 0)\"\"\"\n",
        "test": "assert factorial(0) == 1\nassert factorial(5) == 120\nprint('PASS')"
    }
]

def execute_code(code_str: str, timeout: int = 3) -> bool:
    """
    Saves the code to a temporary file and executes it via subprocess.
    Returns True if the script exits with code 0 (all asserts passed).
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code_str)
        temp_path = f.name

    try:
        # Run the temporary Python file in a sandbox
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        passed = (result.returncode == 0)
    except subprocess.TimeoutExpired:
        passed = False
    finally:
        os.unlink(temp_path)
        
    return passed

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v4/model_step_15000.pt")
    parser.add_argument("--tokenizer", type=str, default="dataset/tokenizer_v2.json")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.checkpoint):
        print(f"Warning: Checkpoint {args.checkpoint} not found. Cannot run benchmark.")
        return

    print(f"Loading EmsyAI for Functional Benchmarking on {device}...")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device, version="v4")
    
    valid_count = 0
    total = len(TEST_CASES)
    
    print("\n" + "="*60)
    print("Starting Execution-Based Benchmark (pass@1)")
    print("="*60)
    
    for i, test_case in enumerate(TEST_CASES):
        prompt_text = test_case["prompt"]
        print(f"\n[{i+1}/{total}] Testing: {prompt_text.strip().split('(')[0]}")
        
        # Format the prompt exactly how the model expects it
        formatted_prompt = f"[USER]\nWrite the function:\n{prompt_text}\n[MODEL]\n{prompt_text}"
        
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_new_tokens=100,
            temperature=0.2, # Low temp for coding tasks
            device=device
        )
        
        # Extract just the model's generated portion
        generated_code = output[len(formatted_prompt):].split("[USER]")[0].strip()
        
        # Combine the original prompt (def signature), the generated code, and the test assertions
        full_program = f"{prompt_text}\n{generated_code}\n\n# --- TESTS ---\n{test_case['test']}"
        
        # Execute it
        is_valid = execute_code(full_program)
        
        if is_valid:
            valid_count += 1
            print("Status: [PASS] Code compiled and passed all assertions.")
        else:
            print("Status: [FAIL] Code failed assertions, syntax error, or timed out.")
            print("--- Generated Code That Failed ---")
            print(generated_code)
            print("----------------------------------")
            
    score = (valid_count / total) * 100
    
    print("\n" + "="*60)
    print(f"Benchmark Results: Functional pass@1 Rate: {valid_count}/{total} ({score:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()
