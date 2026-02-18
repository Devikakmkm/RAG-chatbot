"""
Streamlit UI for the RAG Chatbot.
Provides file upload, chat interface, streaming responses, and source citations.
"""

import streamlit as st
from pathlib import Path
import shutil
import subprocess
from rag import RAGSystem

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 4px solid #0066cc;
    }
    .chat-message {
        margin: 10px 0;
        padding: 10px;
        border-radius: 5px;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20px;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.rag_ready = False
    st.session_state.chat_history = []

# Sidebar
st.sidebar.title("🤖 RAG Chatbot Setup")

with st.sidebar.expander("📋 Setup Instructions", expanded=False):
    st.markdown("""
    ### Quick Start
    
    1. **Install Ollama** from [ollama.com](https://ollama.com)
    
    2. **Pull the model**:
    ```bash
    ollama pull llama3.2
    ```
    
    3. **Start Ollama** (runs on localhost:11434)
    
    4. **Add documents** to `documents/` folder
    
    5. **Ingest** with the button below
    
    6. **Chat!** 💬
    """)

# Check Ollama connection
def check_ollama():
    try:
        from ollama import Client
        client = Client(host="http://localhost:11434")
        client.list()
        return True
    except:
        return False

# Initialize RAG system
def init_rag():
    try:
        st.session_state.rag_system = RAGSystem()
        st.session_state.rag_ready = True
        return True
    except Exception as e:
        st.error(f"Error initializing RAG: {e}")
        return False

# File upload section
st.sidebar.subheader("📁 Document Management")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = Path("documents") / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"✅ Saved {uploaded_file.name}")

# Ingest documents button
if st.sidebar.button("🔄 Ingest Documents", use_container_width=True):
    with st.spinner("Processing documents..."):
        try:
            subprocess.run(["python", "ingest.py"], check=True)
            st.sidebar.success("✨ Documents ingested successfully!")
            st.session_state.rag_system = None
            st.session_state.rag_ready = False
        except subprocess.CalledProcessError as e:
            st.sidebar.error(f"Error during ingestion: {e}")
        except FileNotFoundError:
            st.sidebar.error("Could not run ingest.py. Make sure it's in the current directory.")

# Status section
st.sidebar.divider()
st.sidebar.subheader("📊 System Status")

ollama_status = "🟢 Connected" if check_ollama() else "🔴 Disconnected"
st.sidebar.write(f"**Ollama**: {ollama_status}")

if st.session_state.rag_ready:
    doc_count = st.session_state.rag_system.get_document_count()
    st.sidebar.write(f"**Documents loaded**: {doc_count} chunks")
    st.sidebar.write("**Status**: ✨ Ready to chat")
else:
    if st.sidebar.button("🚀 Initialize RAG System", use_container_width=True):
        with st.spinner("Initializing..."):
            if init_rag():
                st.sidebar.success("✅ RAG System Ready!")
                st.rerun()
            else:
                st.sidebar.error("Failed to initialize RAG system")

# Clear history button
if st.sidebar.button("🗑️  Clear Chat History", use_container_width=True):
    st.session_state.chat_history = []
    if st.session_state.rag_system:
        st.session_state.rag_system.clear_history()
    st.rerun()

# Main chat interface
st.title("🤖 RAG Chatbot")

if not st.session_state.rag_ready:
    st.warning("⚠️ RAG System not initialized. Initialize it in the sidebar first.")
    st.info("👉 Click '🚀 Initialize RAG System' in the left sidebar to get started.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            st.write(message["assistant"])
            if "sources" in message:
                st.markdown("---")
                st.markdown("**📚 Sources:**")
                for file, sources in message["sources"].items():
                    source_text = ", ".join(sources)
                    st.markdown(f"- **{file}**: {source_text}")
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                full_response = ""
                response_placeholder = st.empty()
                
                # Stream the response
                for chunk in st.session_state.rag_system.generate(user_input, stream=True):
                    full_response += chunk
                    response_placeholder.write(full_response)
                
                # Get sources
                sources = st.session_state.rag_system.get_sources(user_input)
                
                # Display sources
                if sources:
                    st.markdown("---")
                    st.markdown("**📚 Sources:**")
                    for file, source_list in sources.items():
                        source_text = ", ".join(source_list)
                        st.markdown(f"- **{file}**: {source_text}")
        
        # Add to history
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": full_response,
            "sources": sources
        })

# Footer
st.divider()
st.caption("🚀 RAG Chatbot | Powered by Ollama, ChromaDB & Streamlit | All runs locally on your machine")
