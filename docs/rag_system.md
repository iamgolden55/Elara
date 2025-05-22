# Elara AI: RAG System Documentation

## Introduction to Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) is a technique that enhances large language models by incorporating external knowledge retrieval. In Elara AI, the RAG system allows the model to access and reference up-to-date medical information rather than relying solely on its pre-trained knowledge.

This approach offers several key advantages:
- Reduces hallucinations (fabricated information)
- Provides access to more current medical information
- Enables citations to authoritative sources
- Improves accuracy for specific medical questions

## System Architecture

The Elara AI RAG system consists of the following components:

1. **Document Corpus**: A collection of medical texts, guidelines, and articles
2. **Vector Database**: FAISS index storing embeddings of document chunks
3. **Retrieval Module**: System to find relevant documents based on queries
4. **Context Integration**: Process of combining retrieved information with the query

![RAG System Architecture](https://i.imgur.com/zDHGpJj.png)

## Implementation Details

### 1. Document Processing Pipeline

Before RAG can be used, medical documents must be processed:

```
Document Collection → Text Extraction → Chunking → Embedding → Vector Database
```

Each document is:
1. Loaded and parsed (removing formatting, etc.)
2. Split into chunks of appropriate size (e.g., 256 tokens)
3. Embedded using a sentence transformer model (e.g., all-mpnet-base-v2)
4. Stored in a FAISS index along with metadata

The `rag_utils.py` file contains the `MedicalRAG` class which handles this functionality.

### 2. Query Processing

When a user asks a question, the RAG system:

1. Embeds the query using the same model used for documents
2. Searches the FAISS index for similar document chunks
3. Retrieves the top-k most relevant chunks
4. Formats these chunks as context for the LLM

Code from `rag_utils.py`:

```python
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
```

### 3. Integration with the LLM

The retrieved context is integrated into the prompt before it's sent to the LLM:

```python
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
```

This context is then added to the prompt in the `generate_medical_response` method of the `ModelManager` class:

```python
# Construct prompt with RAG context
prompt = f"Human: {question}\n"

# Add context if available
context_text = ""

# Check if RAG is available for enhanced context
if medical_rag.is_available():
    rag_context = await medical_rag.get_context_for_query(question)
    if rag_context:
        context_text += rag_context

# Add the context to the prompt
if context_text:
    prompt += f"{context_text.strip()}\n"

# Add the Assistant prefix for completion
prompt += "Assistant:"
```

## Building the RAG Index

The RAG index is built using the `build_medical_rag.py` script, which should be run before using the system:

```bash
python data/scripts/build_medical_rag.py
```

This script:
1. Collects medical documents from various sources
2. Processes them into chunks
3. Creates embeddings for each chunk
4. Builds and saves the FAISS index
5. Saves the document content and metadata separately

The built index is stored in `data/rag/medical_faiss_index.bin`, and the document content is stored in `data/rag/medical_documents.json`. Metadata is stored in `data/rag/medical_rag_metadata.json`.

## Data Sources for RAG

The Elara AI RAG system can use various medical data sources:

1. **PubMed Articles**: Open-access medical research papers
2. **Clinical Guidelines**: Medical practice guidelines from reputable organizations
3. **Medical Textbooks**: Sections from open-source medical textbooks
4. **Public Health Information**: Content from government health websites

When building your own RAG system, ensure all sources are properly cited and used according to their licenses.

## Fine-tuning the RAG System

Several parameters can be adjusted to optimize RAG performance:

1. **Chunk Size**: Smaller chunks (128-256 tokens) for precise retrieval, larger chunks (512-1024 tokens) for more context
2. **Number of Results (k)**: How many document chunks to retrieve (typically 3-5)
3. **Embedding Model**: The model used to create embeddings affects retrieval quality
4. **Similarity Threshold**: Minimum similarity score for relevant results

These parameters can be adjusted in `config.py` or directly in the `rag_utils.py` file.

## Evaluating RAG Performance

To evaluate how well the RAG system is working:

1. **Relevance Assessment**: Are retrieved documents relevant to the query?
2. **Factual Accuracy**: Does the model's response accurately reflect the retrieved information?
3. **Citation Correctness**: Are citations accurate and properly formatted?
4. **Answer Completeness**: Does the response include all important information from retrieved sources?

You can create an evaluation script that measures these metrics on a set of test questions.

## Limitations and Best Practices

While RAG significantly improves the quality of medical responses, it has limitations:

1. **Dependency on Index Quality**: The system is only as good as the documents in the index
2. **Search Limitations**: Vector search may miss relevant information if the query is worded differently
3. **Context Window Constraints**: Too much retrieved content may exceed the model's context window

Best practices:
- Regularly update the document corpus with new medical information
- Implement hybrid search (combining vector and keyword search)
- Use chunking strategies that preserve document context
- Filter out low-quality or irrelevant sources
- Add metadata to chunks for better filtering (e.g., medical specialty, publication date)

## Future Enhancements

Potential improvements to the RAG system:

1. **Hybrid Search**: Combining vector search with BM25 or other keyword-based methods
2. **Re-ranking**: Adding a second-stage ranking model to improve result relevance
3. **Multi-query Expansion**: Generating multiple queries to broaden the search
4. **Hierarchical Retrieval**: Using tiered approaches for better performance
5. **Streaming RAG**: Implementing streamed retrieval for faster response times

## Troubleshooting

Common issues and solutions:

1. **"RAG system not initialized" error**: Check that the FAISS index and document files exist in the correct location
2. **Poor retrieval quality**: Try a different embedding model or adjust chunk size
3. **Slow retrieval**: Use a quantized FAISS index or reduce the index size
4. **Out of memory errors**: Reduce batch size or use a more memory-efficient embedding model

## Conclusion

The RAG system is a critical component of Elara AI, enabling it to provide accurate, up-to-date medical information. By combining the strengths of retrieval systems with generative AI, we can build a more reliable and trustworthy medical assistant.
