from typing import Dict, List, Any, Optional
import asyncio
from models.mistral7b.load_model import generate_text
from models.bloom1.7b.translate import translate_text, detect_language
from vectorstore.embedding import similarity_search

async def generate_response(question: str, 
                           language: str = "en", 
                           user_id: Optional[str] = None,
                           context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Generate a response using RAG (Retrieval Augmented Generation) approach
    
    Args:
        question: User's question
        language: Language code (e.g., 'en', 'es')
        user_id: Optional user ID for personalization
        context: Additional context for the question
        
    Returns:
        Dictionary containing answer, sources, and metadata
    """
    # If not in English, translate the question
    original_question = question
    detected_language = language
    
    if language != "en":
        # Translate question to English for better retrieval
        question = await translate_text(question, language, "en")
    
    # Retrieve relevant documents from vector store
    search_results = await similarity_search(question, k=5)
    
    # Extract relevant context for the LLM
    context_texts = [doc["text"] for doc in search_results]
    
    # Prepare system prompt for medical assistant
    system_prompt = (
        "You are Elara, a helpful medical AI assistant. Provide accurate, evidence-based " 
        "medical information. For serious concerns, always advise consulting with a healthcare professional. "
        "Based on the provided context, answer the user's question. If you don't know the answer or the "
        "information isn't in the context, be honest about it."
    )
    
    # Format RAG prompt with retrieved context
    rag_prompt = f"Context information:\n"
    for i, text in enumerate(context_texts):
        rag_prompt += f"[{i+1}] {text}\n\n"
    
    rag_prompt += f"Question: {question}\n\nPlease provide a helpful and accurate answer based on the context information."
    
    # Generate response
    response = await generate_text(
        prompt=rag_prompt,
        system_prompt=system_prompt,
        max_new_tokens=512,
        temperature=0.7
    )
    
    # If original question was not in English, translate the response back
    if language != "en":
        response = await translate_text(response, "en", language)
    
    # Prepare result
    result = {
        "answer": response,
        "sources": [
            {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": doc["score"]
            } for doc in search_results
        ],
        "metadata": {
            "original_language": detected_language,
            "translated": language != "en"
        }
    }
    
    return result
