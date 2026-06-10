# download_model.py
# Purpose: Download GLiClass model from Hugging Face to local F: drive
# Run: python download_model.py
# Result: Model downloaded to /mnt/f/models/gliclass-llama-1.3B (2.7 GB)

import os
from huggingface_hub import login, snapshot_download

# Remove any old bad token from environment
os.environ.pop('HF_TOKEN', None)

# Login with your Hugging Face token
login(token='hf_your_token_here')  # <-- REPLACE with your actual token

# Download model to F: drive to save WSL home space
model_path = snapshot_download(
    repo_id='knowledgator/gliclass-llama-1.3B-v1.0',
    local_dir='/mnt/f/models/gliclass-llama-1.3B'
)

print('Model downloaded to:', model_path)

# List downloaded files
for f in os.listdir('/mnt/f/models/gliclass-llama-1.3B'):
    print(f)
