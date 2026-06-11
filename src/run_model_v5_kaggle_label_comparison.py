# run_model.py — Version 5: Kaggle v2 (Label Comparison)
# Purpose: Compare disaster vs not-disaster scores directly
# Run: python run_model_v5_kaggle_label_comparison.py
# Docker: docker run --rm -v model -v /data -v ~/.netrc gliclass-model
# ❌ FAILED — all 3263 tweets predicted as not disaster (target=0)
# BUG: 'not disaster' label always scores higher → all predictions = 0

import torch
import wandb
import pandas as pd
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

run = wandb.init(project='mlops-assignment3', name='kaggle-disaster-tweets-v2')

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
    not_disaster_score = scores.get('not disaster', 0)

    # BUG: not_disaster always wins — model biased toward this label
    prediction = 1 if disaster_score > not_disaster_score else 0
    predictions.append(prediction)

    if i % 100 == 0:
        wandb.log({'progress': i, 'disaster_score': disaster_score})

submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
submission.to_csv('/data/submission.csv', index=False)

print(submission['target'].value_counts())
# Output: 0    3263  (ALL not disaster — WRONG)

run.finish()
