import pandas as pd

df = pd.read_parquet('test_0524.parquet')

print(df.head())

print(df.shape)

print(df.columns)

print(df.info())

print(df['prompt'][0])