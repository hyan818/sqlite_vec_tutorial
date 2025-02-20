"""
Step 1: Embed the .txt documents listed in /data and place these docs
with embeddings in the sqlite database.
"""

import os
import sqlite3

import ollama
import sqlite_vec
from sqlite_vec import serialize_float32

# Path to the database file
db_path = "my_docs.db"

# Delete the database file if it exists
if os.path.exists(db_path):
    os.remove(db_path)

# Connect to a database (or create it if it doesn't exist)
db = sqlite3.connect(db_path)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Create a vec0 virtual table to store text files and their embeddings
db.execute("""
    CREATE VIRTUAL TABLE documents USING vec0(
        embedding float[384],
        +file_name TEXT,
        +content TEXT
    )
""")


# Function to get embeddings using Ollama
def get_ollama_embedding(text):
    return ollama.embed(model="all-minilm", input=text).embeddings[0]


# Iterate over .txt files in the /data directory
for file_name in os.listdir("data"):
    file_path = os.path.join("data", file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        # Generate embedding for the content
        embedding = get_ollama_embedding(content)
        if embedding:
            # Insert file content and embedding into the vec0 table
            db.execute(
                "INSERT INTO documents (embedding, file_name, content) VALUES (?, ?, ?)",
                (serialize_float32(list(embedding)), file_name, content),
            )

# Commit changes
db.commit()

"""
Step 2: Ask queries and perform vector similarity search
to pull the relevant documents into the context of the LLM.
"""

# Perform a sample KNN query
query_text = "What is general relativity?"
query_embedding = get_ollama_embedding(query_text)
if query_embedding:
    rows = db.execute(
        """
        SELECT
            file_name,
            content,
            distance
        FROM documents
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 3
        """,
        [serialize_float32(list(query_embedding))],
    ).fetchall()

    print("Top 3 most similar documents:")
    top_contexts = []
    for row in rows:
        print(row)
        top_contexts.append(row[1])  # Append the 'content' column

    # Prepare the context for the query
    context = "\n\n".join(top_contexts)
    system_message = (
        "You are a helpful assistant. Use the following context to answer the query."
    )

    stream = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuery: {query_text}",
            },
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)

# Close the database connection
db.close()
