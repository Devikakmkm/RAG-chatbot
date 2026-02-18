# 🤖 RAG Chatbot

A local, privacy-first RAG (Retrieval-Augmented Generation) chatbot that runs entirely on your machine. Chat with your documents using free, open-source tools.

## ✨ Features

- 📄 **Document Support**: PDFs, DOCX, and TXT files
- 🧠 **Local Embeddings**: Uses `sentence-transformers` (all-MiniLM-L6-v2)
- 💾 **Local Storage**: In-memory embeddings with disk persistence
- 🚀 **Free LLM**: Runs Ollama + Llama 3.2 locally
- 💬 **Streaming Responses**: Real-time chat with source citations
- 🔒 **No API Keys**: Everything runs on your machine
- ⚡ **Fast Retrieval**: Cosine similarity-based semantic search

## 📋 Requirements

- Windows, macOS, or Linux
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- ~2GB disk space for Ollama model

## 🚀 Quick Start

### 1. Install Ollama
Download from [ollama.com](https://ollama.com) and install it.

Then pull the model:
```bash
ollama pull llama3.2
```

### 2. Set Up Project
```bash
# Clone or download this repo
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add Your Documents
Create a `documents/` folder and add your files:
```
documents/
├── document1.pdf
├── document2.docx
└── notes.txt
```

### 4. Ingest Documents
```bash
python ingest.py
```

### 5. Launch Chatbot
In one terminal, start Ollama:
```bash
ollama serve
```

In another terminal, start the app:
```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501` 🎉

## 📁 Project Structure

```
rag-chatbot/
├── app.py              # Streamlit UI
├── rag.py              # Core RAG logic
├── ingest.py           # Document ingestion
├── requirements.txt    # Dependencies
├── documents/          # Your documents go here
├── embeddings.pkl      # Cached embeddings (auto-created)
└── README.md          # This file
```

## 🔧 How It Works

1. **Ingestion** (`ingest.py`)
   - Loads PDFs, DOCX, TXT files
   - Chunks text into 500-character overlapping segments
   - Embeds chunks using `all-MiniLM-L6-v2`
   - Saves embeddings to `embeddings.pkl`

2. **Retrieval** (`rag.py`)
   - Embeds user query
   - Uses cosine similarity to find top-5 relevant chunks
   - Reranks using keyword overlap
   - Returns top-3 most relevant documents

3. **Generation**
   - Builds prompt with context + conversation history
   - Streams response from Ollama (Llama 3.2)
   - Shows source citations

## 📦 Dependencies

| Component | Tool | Version |
|-----------|------|---------|
| LLM | Ollama + Llama 3.2 | Latest |
| Embeddings | sentence-transformers | 3.0+ |
| Storage | In-memory + pickle | Native |
| UI | Streamlit | 1.28+ |
| PDF | pdfplumber | 0.11+ |
| DOCX | python-docx | 0.8+ |

## 🎯 Optional Upgrades

### Better LLM Quality
Edit `rag.py` line 26:
```python
OLLAMA_MODEL = "mistral"  # or "llama3.1:8b"
```

### Better Embeddings
Edit `rag.py` line 21:
```python
EMBED_MODEL = "all-mpnet-base-v2"  # Slower but higher quality
```

## 🐛 Troubleshooting

**Error: "Ollama not running"**
- Start Ollama: `ollama serve`
- Make sure it's running on `localhost:11434`

**Error: "No documents found"**
- Add PDF/DOCX/TXT files to `documents/` folder
- Re-run: `python ingest.py`

**Slow ingestion?**
- Normal for first run (downloads embeddings model ~100MB)
- Subsequent runs are much faster

**Out of memory?**
- Reduce chunk size in `ingest.py` (line 13)
- Use fewer/smaller documents
- Close other applications

## 📝 Commands Summary

```bash
# Setup (one-time)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Ingest documents
python ingest.py

# Run chatbot (in separate terminals)
ollama serve
streamlit run app.py
```

## 🌐 Push to GitHub

### First Time Setup

1. **Create a GitHub repo**
   - Go to [github.com/new](https://github.com/new)
   - Create repository: `rag-chatbot`
   - Don't initialize with README (we have one)

2. **Initialize Git** (in your project folder)
   ```bash
   git init
   git add .
   git commit -m "Initial commit: RAG chatbot setup"
   git branch -M main
   ```

3. **Add Remote & Push**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/rag-chatbot.git
   git push -u origin main
   ```

### Quick Push (after changes)
```bash
git add .
git commit -m "Your message here"
git push
```

## 📄 License

MIT License - feel free to use and modify

## 🤝 Contributing

Contributions welcome! Feel free to:
- Submit issues
- Suggest improvements
- Fork and create pull requests

## 📚 Resources

- [Ollama Docs](https://ollama.com)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

**Questions?** Open an issue on GitHub! 🚀
