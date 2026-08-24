import json

# A small synthetic dataset teaching the model to answer python coding instructions.
# In reality, this would be thousands of examples from a dataset like Alpaca or CodeAlpaca.
INSTRUCT_DATA = [
    {
        "instruction": "Write a Python function to add two numbers.",
        "output": "def add(a, b):\n    return a + b"
    },
    {
        "instruction": "Write a function that returns the factorial of n.",
        "output": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"
    },
    {
        "instruction": "Create a function to check if a string is a palindrome.",
        "output": "def is_palindrome(s):\n    return s == s[::-1]"
    },
    {
        "instruction": "How do I print 'Hello World' in Python?",
        "output": "print('Hello World')"
    },
    {
        "instruction": "Write a class representing a Rectangle.",
        "output": "class Rectangle:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height\n\n    def area(self):\n        return self.width * self.height"
    },
    {
        "instruction": "Write a function that filters even numbers from a list.",
        "output": "def get_evens(numbers):\n    return [n for n in numbers if n % 2 == 0]"
    },
    {
        "instruction": "Write a function to reverse a string.",
        "output": "def reverse_string(s):\n    return s[::-1]"
    },
    {
        "instruction": "What is the syntax for a list comprehension that squares numbers?",
        "output": "squares = [x**2 for x in range(10)]"
    }
]

# We will format our prompts using standard special tokens so the model learns the structure
# <|user|>
# Write a function...
# <|model|>
# def function...<|eos|>

def build_prompt(item: dict) -> str:
    return f"<|user|>\n{item['instruction']}\n<|model|>\n{item['output']}<|eos|>"

def main():
    print("Generating synthetic instruction dataset...")
    with open("dataset/instruct.jsonl", "w", encoding="utf-8") as f:
        for item in INSTRUCT_DATA:
            # We save the raw dict, but we will format it during loading
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(INSTRUCT_DATA)} examples to dataset/instruct.jsonl")

if __name__ == "__main__":
    main()
