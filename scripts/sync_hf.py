import os
from huggingface_hub import HfApi

def sync():
    # HfApi will automatically use the cached token if you ran `hf auth login`
    api = HfApi()
    repo_id = 'gulding/EmsyAI'
    
    print(f'Uploading README.md to {repo_id}...')
    try:
        url = api.upload_file(
            path_or_fileobj='README.md',
            path_in_repo='README.md',
            repo_id=repo_id,
            repo_type='model'
        )
        print(f'SUCCESS! Live URL: {url}')
    except Exception as e:
        print(f'FAILED to upload: {e}')

if __name__ == '__main__':
    sync()
