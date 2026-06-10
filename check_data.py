import pandas as pd

df = pd.read_csv('train.csv')

print('=== HEAD ===')
print(df.head())

print('\n=== TARGET VALUE COUNTS ===')
print(df['target'].value_counts())

print('\n=== MISSING VALUES ===')
print(df.isnull().sum())
