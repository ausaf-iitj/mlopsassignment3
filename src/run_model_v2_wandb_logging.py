# run_model.py — Version 2: With W&B Logging
# Purpose: Log classification scores to W&B dashboard
# Run: python run_model_v2_wandb_logging.py
# Docker: docker run --rm -v "/mnt/f/models/gliclass-llama-1.3B:/models/gliclass-llama-1.3B:ro" -v ~/.netrc:/root/.netrc:ro gliclass-model
# Result: Scores logged — technology=0.3827, sports=0.3613, business=0.1716

import wandb
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

# Initialise W&B run
run = wandb.init(
    project='mlops-assignment3',
    name='gliclass-inference',
    config={'model': 'gliclass-llama-1.3B', 'threshold': 0.1, 'device': 'cpu'}
)
print('W&B run ID:', run.id)

model_path = '/models/gliclass-llama-1.3B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GLiClassModel.from_pretrained(model_path, torch_dtype=torch.float32)
model = model.float()

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='multi-label',
    device='cpu'
)

text = 'Apple announces new iPhone with amazing features'
labels = ['technology', 'sports', 'politics', 'business']

result = pipeline(text, labels, threshold=0.1)

print('\n=== Classification Results ===')
scores = {}
for item in result:
    for pred in item:
        print(f"  {pred['label']}: {pred['score']:.4f}")
        scores[pred['label']] = pred['score']

# Log scores to W&B
wandb.log(scores)
print('Logged to W&B:', scores)

run.finish()
