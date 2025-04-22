# RAG Chat with LangChain and Google Gemini

A learning project implementing a Retrieval-Augmented Generation (RAG) chatbot using LangChain, Google Gemini, and FAISS for efficient similarity search.

## Requirements

- Python >= 3.13
- Dependencies:
  - langchain-google-genai
  - faiss-cpu
  - sentence-transformers
  - python-dotenv

## Installation

```bash
git clone <repository-url>
cd lang_chain
pip install -r requirements.txt
```

## Project Structure

```
lang_chain/
├── main.py                 # Entry point and Gemini API integration
├── requirements.txt        # Project dependencies
├── app/
│   ├── chat.py            # Main RAG implementation
│   ├── data/
│   │   └── meuarquivo.txt # Knowledge base
│   ├── models/
│   │   ├── embeddings.py  # Embedding model implementation
│   │   └── vector_store.py # FAISS vector store implementation
│   └── utils/
│       └── text_processing.py # Text processing utilities
└── .env                    # Environment variables
```

## How It Works

1. **Text Processing**: Documents are loaded and split into manageable chunks
2. **Embedding Generation**: Using Sentence Transformers (all-MiniLM-L6-v2)
3. **Vector Storage**: FAISS indexes embeddings for efficient similarity search
4. **Query Processing**: User questions are embedded and matched with relevant context
5. **Response Generation**: Google Gemini generates responses based on retrieved context

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
LANGSMITH_ENDPOINT=""
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT=""
GOOGLE_API_KEY=""
```

## Usage

1. Set up your environment variables in `.env`
2. Place your text data in `meuarquivo.txt`
3. Run the chat interface:

```bash
python chat.py
```

## Next features

1. Replace FAISS to Pinecone for index database
2. Add support for PDF
3. Create an interface using streamlit 