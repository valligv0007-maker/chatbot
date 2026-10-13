import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Load the FAQ dataset
df = pd.read_csv("data/faq_dataset.csv")

# Step 2: Combine question and answer into one text
df["text"] = (
    "Question: " + df["question"] +
    "\nAnswer: " + df["answer"]
)

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Create embeddings for all FAQ texts
embeddings = model.encode(df["text"].tolist()).tolist()

# Step 5: Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Step 6: Create or get the FAQ collection
collection = client.get_or_create_collection(
    name="faq_collection"
)

# Step 7: Store FAQ data, embeddings and metadata in ChromaDB
collection.add(
    ids=df["id"].astype(str).tolist(),
    documents=df["text"].tolist(),
    embeddings=embeddings,
    metadatas=df[["category"]].to_dict(orient="records")
)

# Step 8: Check how many records are stored
print("FAQ records stored:", collection.count())