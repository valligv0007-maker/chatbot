import pandas as pd

# Step 1: Read the FAQ dataset
df = pd.read_csv("data/faq_dataset.csv")

# Step 2: Create a new column by combining question and answer
df["text"] = (
    "Question: " + df["question"] +
    "\nAnswer: " + df["answer"]
)

# Step 3: Display the first FAQ text
print("First FAQ text:")
print(df["text"].iloc[0])

# Step 4: Display the complete DataFrame
print("\nComplete DataFrame:")
print(df.head())