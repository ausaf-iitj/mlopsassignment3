# check_data.py
# Purpose: Inspect Kaggle disaster tweets dataset

import pandas as pd

# Load data
train_df = pd.read_csv("/mnt/f/datasets/disaster-tweets/train.csv")
test_df  = pd.read_csv("/mnt/f/datasets/disaster-tweets/test.csv")

# Basic info
print("=== TRAIN DATA ===")
print(f"Shape: {train_df.shape}")
print(train_df.head())
print(train_df.info())
print("\nMissing values:")
print(train_df.isnull().sum())

print("\n=== CLASS DISTRIBUTION ===")
print(train_df['target'].value_counts())
print(train_df['target'].value_counts(normalize=True))

print("\n=== TEST DATA ===")
print(f"Shape: {test_df.shape}")
print(test_df.head())
print("\nMissing values:")
print(test_df.isnull().sum())