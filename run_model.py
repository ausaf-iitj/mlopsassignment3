import torch
import wandb
import pandas as pd
import statistics
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

run = wandb.init(project='mlops-assignment3', name='kaggle-disaster-tweets-v3')

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("/models/gliclass-llama-1.3B")
model = GLiClassModel.from_pretrained("/models/gliclass-llama-1.3B",
               torch_dtype=torch.float32)
model = model.float()

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='multi-label',
    device='cpu'
)

test_df = pd.read_csv("/data/test.csv")
labels = ["disaster", "not disaster"]
disaster_scores = []

for i, row in test_df.iterrows():
    text = str(row['text'])
    result = pipeline(text, labels, threshold=0.01)
    scores = {pred['label']: pred['score'] for item in result for pred in item}
    disaster_scores.append(scores.get('disaster', 0))
    if i % 100 == 0:
        print(f"  {i}/{len(test_df)} score={disaster_scores[-1]:.3f}")
        wandb.log({'progress': i, 'disaster_score': disaster_scores[-1]})

median_score = statistics.median(disaster_scores)
predictions = [1 if s > median_score else 0 for s in disaster_scores]

submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
submission.to_csv("/data/submission.csv", index=False)
print(submission['target'].value_counts())
wandb.log({'median_threshold': median_score,
           'total_disaster': int(sum(predictions))})
run.finish()
print("Done!")
