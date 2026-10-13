import pandas as pd
from sentence_transformers import SentenceTransformer

# Step 1: Read the FAQ dataset
df = pd.read_csv("data/faq_dataset.csv")

# Step 2: Combine question and answer
df["text"] = (
    "Question: " + df["question"] +
    "\nAnswer: " + df["answer"]
)

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert FAQ text into embeddings
embeddings = model.encode(df["text"].tolist())

# Step 5: Display the number of embeddings
print("Number of FAQ records:", len(embeddings))

# Step 6: Display the size of one embedding
print("Embedding dimensions:", len(embeddings[0]))

# Step 7: Display the first embedding
print("First embedding:")
print(embeddings[0])