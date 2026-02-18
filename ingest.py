"""
Ingest documents into memory embeddings storage.
Run once to process your documents, then re-run when adding new documents.
"""

import os
import glob
from pathlib import Path
import pdfplumber
from docx import Document
from rag import RAGSystem

# Configuration
DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def load_pdf(file_path):
    """Extract text from PDF."""
    texts = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                texts.append({
                    "content": text,
                    "source": f"{Path(file_path).name}:page_{page_num + 1}",
                    "file": Path(file_path).name
                })
    return texts

def load_docx(file_path):
    """Extract text from DOCX."""
    doc = Document(file_path)
    texts = []
    full_text = "\n".join([para.text for para in doc.paragraphs])
    if full_text:
        texts.append({
            "content": full_text,
            "source": Path(file_path).name,
            "file": Path(file_path).name
        })
    return texts

def load_txt(file_path):
    """Extract text from TXT."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [{
        "content": content,
        "source": Path(file_path).name,
        "file": Path(file_path).name
    }]

def chunk_text(texts, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split texts into overlapping chunks."""
    chunks = []
    for item in texts:
        content = item["content"]
        source = item["source"]
        file = item["file"]
        
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            if len(chunk) > 50:  # Skip very small chunks
                chunks.append({
                    "text": chunk,
                    "source": source,
                    "file": file
                })
    return chunks

def ingest_documents():
    """Main ingestion pipeline."""
    print("🔄 Starting document ingestion...")
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Load all documents
    all_texts = []
    doc_files = []
    
    for pdf_file in glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf")):
        print(f"📄 Loading PDF: {Path(pdf_file).name}")
        all_texts.extend(load_pdf(pdf_file))
        doc_files.append(Path(pdf_file).name)
    
    for docx_file in glob.glob(os.path.join(DOCUMENTS_DIR, "*.docx")):
        print(f"📄 Loading DOCX: {Path(docx_file).name}")
        all_texts.extend(load_docx(docx_file))
        doc_files.append(Path(docx_file).name)
    
    for txt_file in glob.glob(os.path.join(DOCUMENTS_DIR, "*.txt")):
        print(f"📄 Loading TXT: {Path(txt_file).name}")
        all_texts.extend(load_txt(txt_file))
        doc_files.append(Path(txt_file).name)
    
    if not all_texts:
        print("⚠️  No documents found in 'documents/' folder!")
        return
    
    print(f"✅ Loaded {len(all_texts)} text sections")
    
    # Chunk texts
    print("✂️  Chunking documents...")
    chunks = chunk_text(all_texts)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Add chunks to RAG system
    print("📤 Storing embeddings...")
    rag.add_chunks(chunks)
    
    print(f"\n✨ Ingestion complete! {len(chunks)} chunks stored.")
    print(f"📊 Files processed: {', '.join(doc_files)}")

if __name__ == "__main__":
    ingest_documents()

