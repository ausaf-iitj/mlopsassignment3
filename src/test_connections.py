# test_connections.py
# Purpose: Test all 3 service connections: HuggingFace, Kaggle, W&B
# Run: python test_connections.py
# Result: All 3 services connected and verified

import os

# ─── 1. Hugging Face ───────────────────────────────────────────────────────────
from huggingface_hub import whoami

os.environ.pop('HF_TOKEN', None)
user = whoami()
print(f'HF Connected: {user["name"]}')

# ─── 2. Kaggle ─────────────────────────────────────────────────────────────────
import kaggle

kaggle.api.authenticate()
print('Kaggle Connected!')

# ─── 3. Weights & Biases ───────────────────────────────────────────────────────
import wandb

wandb.login()
run = wandb.init(project='mlops-assignment3', name='connection-test', mode='online')
print(f'W&B Connected! Run ID: {run.id}')
run.finish()

print('All services connected successfully!')
