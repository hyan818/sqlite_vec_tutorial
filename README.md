# SQLite RAG Tutorial

**This is a great tutorial on using SQLite to build a RAG application. I've modified it to support Ollama instead of OpenAI.**

A simple project demonstrating Retrieval Augmented Generation (RAG) using SQLite, [sqlite-vec](https://github.com/asg017/sqlite-vec), and OpenAI. It embeds text files, stores them in a SQLite database, and retrieves relevant documents using vector search.

## Features

- **Lightweight:** Single-file SQLite database.
- **Vector Search:** Semantic search with sqlite-vec.
- **OpenAI Integration:** Uses OpenAI for embeddings and chat responses.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/sqlite-rag-tutorial.git
   cd sqlite-rag-tutorial
   ```

2. **Set Up a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   `bash
pip install -r requirements.txt
`
   `requirements.txt` includes:

- sqlite-vec==0.1.6
- openai==1.57.4
- python-dotenv==1.0.1

## Configuration

- Create a `.env` file in the project root:
  ```
  OPENAI_API_KEY=your_openai_api_key_here
  ```
- Place your `.txt` files in the data/ folder.

## Usage

Run the main script:

```bash
python sqlite_rag_tutorial.py
```

The script will:

1. Embed text files from the data/ directory and store them in my_docs.db.
2. Run a sample query (e.g., "What is general relativity?") to retrieve relevant documents.
3. Generate a response using OpenAI's chat API.

## Project Structure

```pgsql
.
├── data
│   └── *.txt
├── my_docs.db         # Generated SQLite database
├── requirements.txt
└── sqlite_rag_tutorial.py
```

## Acknowledgements

- [sqlite-vec](https://github.com/asg017/sqlite-vec)
- OpenAI
- Mozilla Builders Accelerator
