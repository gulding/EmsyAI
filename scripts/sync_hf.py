from huggingface_hub import HfApi
api = HfApi()
repo = 'gulding/EmsyAI'

print("Uploading V3 GGUF...")
api.upload_file(path_or_fileobj='emsyai-v3-titan-instruct-f32.gguf', path_in_repo='emsyai-v3-titan-instruct-f32.gguf', repo_id=repo, repo_type='model')

print("Uploading stripped V3 checkpoint...")
api.upload_file(path_or_fileobj='checkpoints_v3/base_model_154M_step_5000.pt', path_in_repo='base_model_154M_step_5000.pt', repo_id=repo, repo_type='model')

print("Deleting old bloated checkpoint...")
try:
    api.delete_file('base_model_120M_step_5000.pt', repo_id=repo, repo_type='model')
except: pass

print("Deleting redundant tokenizer...")
try:
    api.delete_file('tokenizer_v2.json', repo_id=repo, repo_type='model')
except: pass

print("Done!")
