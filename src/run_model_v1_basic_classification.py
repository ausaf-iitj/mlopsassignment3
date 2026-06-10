# run_model.py — Version 1: Basic Classification
# Purpose: First working version — load model and classify a single text
# Run: python run_model_v1_basic_classification.py
# Docker: docker run --rm -v "/mnt/f/models/gliclass-llama-1.3B:/models/gliclass-llama-1.3B:ro" gliclass-model
# Result: technology:0.3827  sports:0.3613  business:0.1716
# Key fix: model.float() — converts BFloat16 weights to Float32 to fix RuntimeError

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import torch

model_path = '/models/gliclass-llama-1.3B'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path)

print('Loading model...')
model = GLiClassModel.from_pretrained(model_path, torch_dtype=torch.float32)
model = model.float()  # Fix: BFloat16 -> Float32

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='multi-label',
    device='cpu'
)

# Test classification
text = 'Apple announces new iPhone with amazing features'
labels = ['technology', 'sports', 'politics', 'business']

result = pipeline(text, labels, threshold=0.1)

print('\n=== Classification Results ===')
for item in result:
    for pred in item:
        print(f"  {pred['label']}: {pred['score']:.4f}")
