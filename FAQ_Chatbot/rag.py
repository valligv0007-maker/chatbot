import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# Step 1: Load environment variables
load_dotenv()

# Step 2: Create the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 3: Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Step 5: Get the FAQ collection
collection = chroma_client.get_collection(
    name="faq_collection"
)


# Step 6: Retrieve relevant FAQs
def search_faq(question, top_k=3):

    # Convert user's question into an embedding
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    return results


# Step 7: Generate an answer using the LLM
def generate_answer(question):

    # Retrieve relevant FAQs
    results = search_faq(question)

    # Get retrieved documents
    documents = results["documents"][0]

    # Combine retrieved FAQs into context
    context = "\n\n".join(documents)

    # Create the prompt
    prompt = f"""
You are an FAQ chatbot.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context,
say: "I don't have enough information to answer that."

Context:
{context}

User Question:
{question}

Answer:
"""

    # Send the prompt to the LLM
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    # Return the generated answer
    return response.output_text


# Step 8: Test the complete RAG system
if __name__ == "__main__":

    user_question = "How do I get admission?"

    answer = generate_answer(user_question)

    print("\nUser Question:")
    print(user_question)

    print("\nChatbot Answer:")
    print(answer)