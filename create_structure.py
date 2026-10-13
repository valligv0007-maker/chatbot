import os

# Step 1: Create the main project folder
os.makedirs("FAQ_Chatbot", exist_ok=True)

# Step 2: Create the data folder inside FAQ_Chatbot
os.makedirs("FAQ_Chatbot/data", exist_ok=True)

# Step 3: Create the ChromaDB folder inside FAQ_Chatbot
os.makedirs("FAQ_Chatbot/chroma_db", exist_ok=True)

# Step 4: Create the app.py file
open("FAQ_Chatbot/app.py", "w").close()

# Step 5: Create the ingest.py file
open("FAQ_Chatbot/ingest.py", "w").close()

# Step 6: Create the rag.py file
open("FAQ_Chatbot/rag.py", "w").close()

# Step 7: Create the requirements.txt file
open("FAQ_Chatbot/requirements.txt", "w").close()

# Step 8: Create the FAQ CSV file inside the data folder
open("FAQ_Chatbot/data/faq_dataset.csv", "w").close()

# Step 9: Print a success message
print("FAQ Chatbot project structure created successfully!")