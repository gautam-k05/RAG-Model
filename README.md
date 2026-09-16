# PDF RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that lets you ask natural-language questions over a folder of PDF documents and get grounded, cited answers.

## What it does

- Loads and parses all PDFs from a directory (recursively)
- Splits documents into overlapping chunks for better retrieval
- Generates embeddings with `sentence-transformers` and stores them in a persistent **ChromaDB** vector store
- Retrieves the most relevant chunks for a query using cosine similarity
- Passes retrieved context to an LLM (via **Groq**) to generate a grounded answer
- Advanced pipeline adds source citations, streaming output, query history, and answer summarization

## Tech Stack

| Component | Tool |
|---|---|
| PDF parsing | LangChain (`PyPDFLoader`) |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| LLM inference | Groq API (`ChatGroq`) |

## How it works

```
PDFs → Load & Parse → Chunk → Embed → Store in ChromaDB
                                              ↓
                    Query → Embed Query → Similarity Search → Retrieve Top-K Chunks
                                              ↓
                              LLM (Groq) + Retrieved Context → Cited Answer
```

## Setup

> **Note:** Any API keys visible in the notebook's commit history have been rotated and are no longer valid. Use your own key via a `.env` file as described below.

1. Clone the repo and install dependencies:
   ```bash
   pip install langchain-community langchain-text-splitters chromadb sentence-transformers langchain-groq python-dotenv
   ```
2. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Place your PDF files in a `data/` directory.
4. Run the notebook cells in order — the pipeline will process the PDFs, build the vector store, and let you query it.

## Example

```python
answer = rag_simple("What is a word vector?", rag_retriever, llm)
```

Returns an answer with cited source PDFs and page numbers.

## Possible Improvements

- Swap notebook cells into a proper Python package with a CLI or API endpoint
- Add a lightweight UI (Streamlit/Gradio) for interactive querying
- Cache embeddings to avoid recomputation on re-runs
