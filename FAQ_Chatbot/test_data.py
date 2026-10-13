import pandas as pd

# Step 1: Read the FAQ CSV file
df = pd.read_csv("data/faq_dataset.csv")

# Step 2: Display the first 5 rows
print("First 5 FAQ records:")
print(df.head())

# Step 3: Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Step 4: Display the column names
print("\nColumn names:")
print(df.columns)

# Step 5: Check for missing values
print("\nMissing values:")
print(df.isnull().sum())