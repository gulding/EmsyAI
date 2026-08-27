from huggingface_hub import HfApi
import re

api = HfApi()
repo = 'gulding/EmsyAI'

# Read the readme
with open('HF_README_V3.md', 'r', encoding='utf-8') as f:
    readme = f.read()

# Replace the strong claim with reality
old_claim = "Prompt loss masking successfully eliminated base-model \"topic drift\", resulting in a model that strictly outputs the desired python format."
new_claim = """Prompt loss masking successfully eliminated base-model "topic drift" (the model attempts to output Python blocks rather than continuing the prompt), but the underlying logic is still heavily limited by the 154M parameter count. 

Here is a raw, unedited transcript from the model demonstrating its capabilities (and limitations):

```python
[USER]
Write the function def is_prime(n):
[MODEL]
def is_prime(n):
  if n <= 1:
    return False

 for i in range(2, int(n-1)+"):
       print("Fizz")
```
*Note: As seen above, while the model attempts to generate Python, it frequently hallucinates syntax errors or drifts into unrelated coding topics (like FizzBuzz).*"""

if old_claim not in readme:
    print("Warning: Could not find exact claim to replace. Appending instead.")
    readme += "\n\n" + new_claim
else:
    readme = readme.replace(old_claim, new_claim)

with open('HF_README_V3.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("Uploading updated README...")
api.upload_file(path_or_fileobj='HF_README_V3.md', path_in_repo='README.md', repo_id=repo, repo_type='model')
print("Done!")
