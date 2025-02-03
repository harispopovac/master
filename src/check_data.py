import pandas as pd

# Read the CSV file
df = pd.read_csv('data/quran.csv')

# Print basic information
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nTotal number of verses:", len(df)) 