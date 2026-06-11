# test_wandb.py
# Purpose: Standalone W&B connection test
# Run: python3 test_wandb.py
# Result: W&B run created: ID kn6ofac0, project mlops-assignment3

import wandb

wandb.login()
print('W&B Connected!')

run = wandb.init(project='mlops-assignment3', mode='online')
print('Run ID:', run.id)

run.finish()
print('Done!')
