import pandas as pd
import os

path = "/Users/prateekmac/Downloads/untitled folder/student_resource/dataset/test/test_source1.tsv"

df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nDtypes:")
print(df.dtypes)

print("\nMissing:")
print(df.isna().sum())

print("\nSample:")
print(df.head())