# run_model.py — Version 4: Kaggle v1 (Fixed Threshold)
# Purpose: Classify all 3263 disaster tweets with fixed threshold 0.3
# Run: python run_model_v4_kaggle_fixed_threshold.py
# Docker: docker run --rm -v model -v /data -v ~/.netrc gliclass-model
# Kaggle score: 0.42966
# ❌ FAILED — all 3263 tweets predicted as disaster (target=1)
# BUG: threshold=0.3 too low — disaster scores mostly above 0.3 → all predictions = 1

import torch
import wandb
import pandas as pd
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

run = wandb.init(project='mlops-assignment3', name='kaggle-disaster-tweets')

tokenizer = AutoTokenizer.from_pretrained('/models/gliclass-llama-1.3B')
model = GLiClassModel.from_pretrained('/models/gliclass-llama-1.3B',
                                      torch_dtype=torch.float32)
model = model.float()

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='multi-label',
    device='cpu'
)

test_df = pd.read_csv('/data/test.csv')
labels = ['disaster', 'not disaster']
predictions = []

for i, row in test_df.iterrows():
    text = str(row['text'])
    result = pipeline(text, labels, threshold=0.1)
    scores = {pred['label']: pred['score']
              for item in result for pred in item}

    disaster_score = scores.get('disaster', 0)

    # BUG: 0.3 too low — most tweets score above this
    prediction = 1 if disaster_score > 0.3 else 0
    predictions.append(prediction)

    if i % 100 == 0:
        wandb.log({'progress': i, 'disaster_score': disaster_score})

submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
submission.to_csv('/data/submission.csv', index=False)

print(submission['target'].value_counts())
# Output: 1    3263  (ALL disaster — WRONG)

run.finish()
