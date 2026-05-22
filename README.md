# Multi-Agent Codebase RAG Assistant


Ask questions about any Python codebase and auto-generate documentation using AI.

## What it does
- Reads `.py` files and indexes them into ChromaDB
- 3 agents answer your questions (Retrieval → Reasoning → Response)
- Generates FRD documentation for any module

User gives folder path
        ↓
Read all .py files
        ↓
AST splits into functions/classes
        ↓
Store in ChromaDB
        ↓
User asks a question
        ↓
Retrieval Agent → finds chunks
        ↓
Reasoning Agent → what's relevant
        ↓
Response Agent → final answer


## How to run

1. Install packages
```bash
pip install -r requirements.txt
```

2. Add your Groq API key to `.env` file
```
GROQ_API_KEY=your_key_here
```

3. Run
```bash
streamlit run main.py
```

## Files
- `rag.py` — all agents and logic
- `main.py` — web interface

## Get free Groq API key
https://console.groq.com


Quick summary:

Import	            -            Purpose

uuid4	            -           Unique IDs

load_dotenv	    -             API keys

Path	            -            Handle file paths

UnstructuredURLLoader  -      Load website content

RecursiveCharacterTextSplitter -	Split text into chunks

Chroma	          -              Vector database

ChatGroq	     -          Connect to Groq LLM

HuggingFaceEmbeddings	-        Create embeddings

ast	          -             Analyze Python code structure

os	          -              Work with files/folders
