# run_model.py — Version 6 FINAL: Median Threshold
# Purpose: Use median of all disaster scores as dynamic threshold
# Run: python run_model_v6_kaggle_median_threshold_BEST.py
# Docker: docker run --rm -v model -v /data -v ~/.netrc gliclass-model
# ✅ SUCCESS — Kaggle score: 0.48605 (BEST)
# Result: 1631 disaster, 1632 not disaster — balanced 50/50 split
#
# KEY INSIGHT: Collect all scores first → compute median → use as threshold
# → Adapts to model's score distribution → balanced predictions

import torch
import wandb
import pandas as pd
import statistics
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

# Initialise W&B
run = wandb.init(
    project='mlops-assignment3',
    name='kaggle-disaster-tweets-v3'
)

# Load model
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained('/models/gliclass-llama-1.3B')
model = GLiClassModel.from_pretrained('/models/gliclass-llama-1.3B',
                                      torch_dtype=torch.float32)
model = model.float()

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='multi-label',
    device='cpu'
)

# Load competition data
test_df = pd.read_csv('/data/test.csv')
train_df = pd.read_csv('/data/train.csv')
print(f'Test: {len(test_df)} tweets')
print(f'Train distribution:\n{train_df["target"].value_counts()}')

# ── Step 1: Collect all disaster scores ───────────────────────────────────────
labels = ['disaster', 'not disaster']
disaster_scores = []

for i, row in test_df.iterrows():
    text = str(row['text'])
    # Use very low threshold to capture all scores
    result = pipeline(text, labels, threshold=0.01)
    scores = {pred['label']: pred['score']
              for item in result for pred in item}

    disaster_score = scores.get('disaster', 0)
    not_disaster_score = scores.get('not disaster', 0)
    disaster_scores.append(disaster_score)

    if i % 100 == 0:
        print(f'  {i}/{len(test_df)}'
              f' | disaster={disaster_score:.3f}'
              f' | not_disaster={not_disaster_score:.3f}')
        wandb.log({
            'progress': i,
            'disaster_score': disaster_score,
            'not_disaster_score': not_disaster_score
        })

# ── Step 2: Use MEDIAN as dynamic threshold ────────────────────────────────────
# This adapts to model's score distribution → balanced predictions
median_score = statistics.median(disaster_scores)
print(f'\nMedian disaster score: {median_score:.4f}')
# Our value: 0.4502

# ── Step 3: Predict using median threshold ─────────────────────────────────────
predictions = [1 if s > median_score else 0 for s in disaster_scores]

# ── Step 4: Save submission CSV ────────────────────────────────────────────────
submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
submission.to_csv('/data/submission.csv', index=False)

print('\n=== Final Results ===')
print(submission['target'].value_counts())
# Output: 1 → 1631,  0 → 1632  (balanced!)

# Log final metrics to W&B
wandb.log({
    'median_threshold': median_score,
    'total_disaster': int(sum(predictions)),
    'total_not_disaster': int(len(predictions) - sum(predictions))
})

run.finish()
print('Done! Submission saved and results logged to W&B!')
