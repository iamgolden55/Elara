"""
🔍🧠 Elara AI - RAG Integration Module

This module integrates the Retrieval Augmented Generation (RAG) system
with the backend model to improve response quality by retrieving relevant
medical information before generating answers.
"""

import os
import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
import time

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the settings from our config
import config
from config import Settings

# Initialize settings
settings = Settings()

# Check for required packages
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Run 'pip install sentence-transformers' for RAG.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Run 'pip install faiss-cpu' for RAG.")

# Paths
RAG_DIR = settings.data_dir / "rag"
FAISS_INDEX_PATH = RAG_DIR / "medical_faiss_index.bin"
DOCUMENTS_PATH = RAG_DIR / "medical_documents.json"
METADATA_PATH = RAG_DIR / "medical_rag_metadata.json"

# Default embedding model (should match what was used in build_medical_rag.py)
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

class MedicalRAG:
    """
    Retrieval Augmented Generation system for medical knowledge
    
    This class provides methods to search the medical knowledge base
    and retrieve relevant information for improved AI responses.
    """
    
    def __init__(self):
        """Initialize the RAG system"""
        self.index = None
        self.chunks = None
        self.metadatas = None
        self.embedding_model_name = DEFAULT_EMBEDDING_MODEL
        self.embedding_model = None
        self.initialized = False
        self.embedding_dimension = 0
        
        print("🔍 Initializing Medical RAG system...")
    
    async def initialize(self):
        """Initialize the RAG system asynchronously"""
        
        if self.initialized:
            return True
        
        # Check if required packages are available
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE:
            print("⚠️ RAG system disabled: Required packages not installed")
            return False
        
        # Check if RAG files exist
        if not all([FAISS_INDEX_PATH.exists(), DOCUMENTS_PATH.exists(), METADATA_PATH.exists()]):
            print(f"⚠️ RAG system disabled: Files not found at {RAG_DIR}")
            print("   Run data/scripts/build_medical_rag.py to create the RAG index")
            return False
        
        try:
            # Load metadata first (it's small)
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.embedding_model_name = metadata.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            self.metadatas = metadata.get("metadatas", [])
            
            # Load embedding model
            print(f"🧠 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # Load document chunks
            print(f"📚 Loading document chunks from {DOCUMENTS_PATH}")
            with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            # Load FAISS index
            print(f"🔍 Loading FAISS index from {FAISS_INDEX_PATH}")
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            
            print(f"✅ RAG system initialized with {len(self.chunks)} chunks and {self.index.ntotal} vectors")
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            return False
    
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the medical knowledge base for information relevant to a query
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of search results with content and metadata
        """
        
        if not self.initialized:
            print("⚠️ RAG system not initialized")
            return []
        
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
            
            # Normalize the query vector (for cosine similarity)
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks) and idx >= 0:  # Ensure valid index
                    results.append({
                        "content": self.chunks[idx],
                        "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                        "score": float(distances[0][i]),
                        "index": int(idx)
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching RAG system: {e}")
            return []
    
    async def get_context_for_query(self, query: str, max_results: int = 3) -> str:
        """
        Get formatted context information for a query
        
        Args:
            query: The user's question
            max_results: Maximum number of results to include
            
        Returns:
            Formatted context string to include in the prompt
        """
        
        if not self.initialized:
            return ""
        
        # Search for relevant information
        results = await self.search(query, k=max_results)
        
        if not results:
            return ""
        
        # Format context
        context = "\n\nRelevant medical information:\n"
        
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            title = metadata.get("title", "")
            source_type = metadata.get("type", "")
            
            # Add source info
            if title:
                context += f"\n--- Source {i+1}: {title}"
                
                if "journal" in metadata and metadata.get("journal"):
                    context += f" (from {metadata['journal']}"
                    
                    if "publish_year" in metadata and metadata.get("publish_year"):
                        context += f", {metadata['publish_year']}"
                    
                    context += ")"
            
            # Add content
            content = result.get("content", "").replace("Title: ", "").replace("\nAbstract: ", ": ")
            context += f"\n{content}\n"
        
        return context
    
    def is_available(self) -> bool:
        """Check if the RAG system is available"""
        return self.initialized

# Create a global instance
medical_rag = MedicalRAG()
