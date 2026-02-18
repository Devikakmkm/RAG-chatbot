"""
RAG (Retrieval-Augmented Generation) core logic - Simplified version.
Uses in-memory embeddings storage instead of ChromaDB (no C++ required).
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import Client
import pickle
import os

# Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"
RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3
EMBEDDINGS_FILE = "embeddings.pkl"

class RAGSystem:
    def __init__(self):
        """Initialize RAG system with in-memory embeddings."""
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.ollama_client = Client(base_url=OLLAMA_HOST)
        
        # In-memory storage
        self.chunks = []
        self.embeddings = None
        self.conversation_history = []
        
        # Load existing embeddings if available
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load embeddings from disk if they exist."""
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data["chunks"]
                    self.embeddings = data["embeddings"]
                print(f"✅ Loaded {len(self.chunks)} chunks from disk")
            except Exception as e:
                print(f"⚠️ Could not load embeddings: {e}")
    
    def save_embeddings(self):
        """Save embeddings to disk."""
        try:
            with open(EMBEDDINGS_FILE, "wb") as f:
                pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
        except Exception as e:
            print(f"⚠️ Could not save embeddings: {e}")
    
    def add_chunks(self, chunks):
        """Add new chunks and embed them."""
        print("🧠 Embedding chunks...")
        new_embeddings = self.embedder.encode([c["text"] for c in chunks])
        
        if self.embeddings is None:
            self.chunks = chunks
            self.embeddings = np.array(new_embeddings)
        else:
            self.chunks.extend(chunks)
            self.embeddings = np.vstack([self.embeddings, np.array(new_embeddings)])
        
        self.save_embeddings()
        print(f"✅ Added {len(chunks)} chunks")
    
    def retrieve(self, query, top_k=RETRIEVAL_TOP_K):
        """Retrieve relevant chunks using cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity([query_embedding[0]], self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Only include relevant chunks
                retrieved.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "file": self.chunks[idx]["file"],
                    "similarity": float(similarities[idx])
                })
        
        return retrieved
    
    def rerank(self, query, retrieved_docs, top_k=RERANK_TOP_K):
        """Simple keyword-overlap based reranking."""
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc in retrieved_docs:
            text_words = set(doc["text"].lower().split())
            overlap = len(query_words & text_words)
            score = overlap / (len(query_words) + 1)
            scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
    
    def build_prompt(self, query, context_docs):
        """Build the prompt with context for the LLM."""
        context_text = "\n\n".join([
            f"[{doc['source']}]\n{doc['text']}"
            for doc in context_docs
        ])
        
        history_text = ""
        if self.conversation_history:
            history_text = "Previous conversation:\n"
            for msg in self.conversation_history[-4:]:
                history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        prompt = f"""You are a helpful assistant. Use the provided context to answer questions accurately.

{history_text}

Context documents:
{context_text}

Question: {query}

Answer based on the context above. If the context doesn't contain relevant information, say so."""
        
        return prompt
    
    def generate(self, query, stream=False):
        """Generate response using Ollama with streaming support."""
        retrieved = self.retrieve(query)
        reranked = self.rerank(query, retrieved)
        
        if not reranked:
            return "No relevant documents found to answer this question.", []
        
        prompt = self.build_prompt(query, reranked)
        
        full_response = ""
        response = self.ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=True
        )
        
        for chunk in response:
            if "response" in chunk:
                text = chunk["response"]
                full_response += text
                if stream:
                    yield text
        
        self.conversation_history.append({
            "user": query,
            "assistant": full_response
        })
        
        if not stream:
            yield full_response
    
    def generate_non_streaming(self, query):
        """Generate response without streaming."""
        full_response = ""
        for chunk in self.generate(query, stream=False):
            full_response += chunk
        return full_response
    
    def get_sources(self, query):
        """Get source documents used for the answer."""
        retrieved = self.retrieve(query)
        reranked = self.rerank(query, retrieved)
        
        sources = {}
        for doc in reranked:
            file = doc["file"]
            source = doc["source"]
            if file not in sources:
                sources[file] = []
            sources[file].append(source)
        
        return sources
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_document_count(self):
        """Get total number of chunks in memory."""
        return len(self.chunks) if self.chunks else 0

