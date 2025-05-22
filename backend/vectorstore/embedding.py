from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from config import settings

# Global variables
_model = None
_index = None
_documents = []

def get_embedding_model():
    """
    Load or get cached sentence transformer model for embeddings
    """
    global _model
    
    # Return cached model if already loaded
    if _model is not None:
        return _model
    
    # Load model
    print("Loading sentence transformer model for embeddings")
    _model = SentenceTransformer('all-mpnet-base-v2')
    
    return _model

def get_faiss_index():
    """
    Load or create FAISS index for vector search
    """
    global _index, _documents
    
    # Return cached index if already loaded
    if _index is not None:
        return _index, _documents
    
    # Get base path
    base_path = settings.VECTORSTORE_PATH
    index_path = f"{base_path}.faiss"
    documents_path = f"{base_path}.pkl"
    
    # Check if index exists
    if os.path.exists(index_path) and os.path.exists(documents_path):
        print(f"Loading existing FAISS index from {index_path}")
        
        # Load index
        _index = faiss.read_index(index_path)
        
        # Load documents
        with open(documents_path, 'rb') as f:
            _documents = pickle.load(f)
    else:
        print("Creating new FAISS index")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Create empty index with dimension 768 (matches the embedding model)
        _index = faiss.IndexFlatL2(768)
        _documents = []
        
        # Save empty index and documents
        faiss.write_index(_index, index_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(_documents, f)
    
    return _index, _documents

async def add_documents(texts: List[str], metadata: List[Dict[str, Any]] = None):
    """
    Add documents to the FAISS index
    
    Args:
        texts: List of text documents to add
        metadata: Optional list of metadata dictionaries for each document
    """
    global _index, _documents
    
    if metadata is None:
        metadata = [{} for _ in texts]
    
    if len(texts) != len(metadata):
        raise ValueError("Number of texts and metadata must match")
    
    # Get embedding model and index
    model = get_embedding_model()
    index, documents = get_faiss_index()
    
    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    
    # Add to index
    index.add(np.array(embeddings).astype('float32'))
    
    # Add documents and metadata
    for i, (text, meta) in enumerate(zip(texts, metadata)):
        documents.append({
            "text": text,
            "metadata": meta,
            "id": len(documents) + i
        })
    
    # Save updated index and documents
    base_path = settings.VECTORSTORE_PATH
    faiss.write_index(index, f"{base_path}.faiss")
    with open(f"{base_path}.pkl", 'wb') as f:
        pickle.dump(documents, f)
    
    _index = index
    _documents = documents

async def similarity_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar documents in the FAISS index
    
    Args:
        query: Query text
        k: Number of results to return
        
    Returns:
        List of document dictionaries with text, metadata, and similarity score
    """
    # Get embedding model and index
    model = get_embedding_model()
    index, documents = get_faiss_index()
    
    # Check if index is empty
    if index.ntotal == 0:
        return []
    
    # Generate query embedding
    query_embedding = model.encode([query], convert_to_tensor=False)[0].reshape(1, -1).astype('float32')
    
    # Search index
    scores, indices = index.search(query_embedding, min(k, index.ntotal))
    
    # Prepare results
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(documents):  # Ensure valid index
            doc = documents[idx]
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "id": doc["id"],
                "score": float(score)  # Convert to Python float for JSON serialization
            })
    
    return results
