#!/usr/bin/env python3
"""
💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥
                            ELARA AI - CHAT ROUTES
                        The Medical AI Restaurant Menu! 🍽️
💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥💬🏥
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import time
import uuid
from datetime import datetime

# Import our schemas (data structures)
from schemas import (
    ChatRequest, 
    ChatResponse, 
    ConversationHistory,
    HealthQuery,
    TranslationRequest
)

# Create router - like a section of our restaurant menu! 📋
chat_router = APIRouter()

# In-memory conversation storage (in production, use a database!)
conversations: Dict[str, ConversationHistory] = {}

@chat_router.post("/ask", response_model=ChatResponse)
async def ask_medical_question(
    request: ChatRequest,
    http_request: Request
):
    """
    🧠 ASK ELARA - Main medical Q&A endpoint
    
    This is where the magic happens! Send a medical question,
    get an AI-powered answer with sources and safety checks.
    """
    
    print(f"🤔 New medical question: {request.question[:50]}...")
    
    try:
        # Get model manager from request state
        model_manager = getattr(http_request.state, 'model_manager', None)
        
        if not model_manager:
            # Fallback response if no AI models loaded
            return ChatResponse(
                response="Hello! I'm Elara AI, but my AI models are currently loading. Please try again in a moment!",
                confidence=0.0,
                sources=[],
                language_detected=request.language or "en",
                processing_time=0.1,
                conversation_id=str(uuid.uuid4()),
                safety_warning="AI models not yet available"
            )
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Step 1: Detect language (if not provided)
        detected_language = request.language or "en"
        if not request.language:
            detected_language = await model_manager.detect_language(request.question)
        
        # Step 2: Translate to English if needed (for processing)
        english_question = request.question
        if detected_language != "en":
            english_question = await model_manager.translate_to_english(
                request.question, 
                detected_language
            )
        
        # Step 3: Retrieve relevant medical context (RAG)
        relevant_context = await model_manager.retrieve_medical_context(
            english_question,
            max_sources=5
        )
        
        # Step 4: Generate AI response
        ai_response = await model_manager.generate_medical_response(
            question=english_question,
            context=relevant_context,
            user_type=request.user_type,
            include_sources=request.include_sources
        )
        
        # Step 5: Translate back to user's language
        final_response = ai_response["response"]
        if detected_language != "en":
            final_response = await model_manager.translate_from_english(
                ai_response["response"],
                detected_language
            )
        
        # Step 6: Apply safety checks
        safety_warning = None
        if ai_response.get("needs_professional_consultation"):
            safety_warning = "This response is for informational purposes only. Please consult a healthcare professional for personalized advice."
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = ChatResponse(
            response=final_response,
            confidence=ai_response.get("confidence", 0.8),
            sources=ai_response.get("sources", []),
            language_detected=detected_language,
            processing_time=processing_time,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            safety_warning=safety_warning
        )
        
        # Store conversation (if user wants history)
        if request.conversation_id:
            await store_conversation(request, response)
        
        print(f"✅ Response generated in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing your question: {str(e)}"
        )

@chat_router.post("/translate", response_model=Dict[str, Any])
async def translate_text(
    request: TranslationRequest,
    http_request: Request
):
    """
    🌍 TRANSLATE - Convert text between languages
    
    Useful for multilingual medical communication!
    """
    
    try:
        model_manager = getattr(http_request.state, 'model_manager', None)
        
        if not model_manager:
            raise HTTPException(
                status_code=503,
                detail="Translation service not available"
            )
        
        # Perform translation
        translated_text = await model_manager.translate_text(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language
        )
        
        return {
            "original_text": request.text,
            "translated_text": translated_text,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "service": "Elara AI Translation"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation error: {str(e)}"
        )

@chat_router.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """
    📚 GET CONVERSATION - Retrieve chat history
    
    Get the full conversation history for a session.
    """
    
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )
    
    return conversations[conversation_id]

@chat_router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    🗑️ DELETE CONVERSATION - Clear chat history
    
    Remove conversation for privacy (GDPR compliance).
    """
    
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": "Conversation deleted successfully"}
    
    raise HTTPException(
        status_code=404,
        detail="Conversation not found"
    )

@chat_router.post("/health-check", response_model=Dict[str, Any])
async def health_assessment(
    request: HealthQuery,
    http_request: Request
):
    """
    🏥 HEALTH CHECK - Symptom analysis
    
    Analyze symptoms and provide health guidance.
    WARNING: This is for informational purposes only!
    """
    
    try:
        model_manager = getattr(http_request.state, 'model_manager', None)
        
        if not model_manager:
            raise HTTPException(
                status_code=503,
                detail="Health assessment service not available"
            )
        
        # Analyze symptoms
        assessment = await model_manager.analyze_symptoms(
            symptoms=request.symptoms,
            age=request.age,
            gender=request.gender,
            medical_history=request.medical_history
        )
        
        # Add mandatory disclaimers
        assessment["disclaimer"] = {
            "warning": "🚨 This is NOT a medical diagnosis!",
            "advice": "Please consult a healthcare professional for proper medical evaluation.",
            "emergency": "If experiencing emergency symptoms, call emergency services immediately."
        }
        
        return assessment
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health assessment error: {str(e)}"
        )

@chat_router.get("/models/status")
async def get_model_status(http_request: Request):
    """
    🤖 MODEL STATUS - Check which AI models are available
    
    Useful for debugging and monitoring.
    """
    
    model_manager = getattr(http_request.state, 'model_manager', None)
    
    if not model_manager:
        return {
            "models_loaded": False,
            "status": "Model manager not initialized",
            "available_models": []
        }
    
    return {
        "models_loaded": model_manager.are_models_loaded(),
        "available_models": model_manager.get_available_models(),
        "model_details": model_manager.get_model_details(),
        "memory_usage": model_manager.get_memory_usage(),
        "last_updated": datetime.now().isoformat()
    }

# Helper function to store conversations
async def store_conversation(request: ChatRequest, response: ChatResponse):
    """Store conversation in memory (would be database in production)"""
    
    conversation_id = response.conversation_id
    
    if conversation_id not in conversations:
        conversations[conversation_id] = ConversationHistory(
            conversation_id=conversation_id,
            started_at=datetime.now(),
            messages=[]
        )
    
    # Add question and response to history
    conversations[conversation_id].messages.extend([
        {
            "type": "user",
            "content": request.question,
            "timestamp": datetime.now(),
            "language": response.language_detected
        },
        {
            "type": "assistant", 
            "content": response.response,
            "timestamp": datetime.now(),
            "confidence": response.confidence,
            "sources": response.sources
        }
    ])
    
    conversations[conversation_id].last_updated = datetime.now()

# Demo route for testing
@chat_router.get("/demo")
async def demo_response():
    """
    🎭 DEMO - Test the API without AI models
    
    Returns a sample response for testing.
    """
    from schemas import MedicalSource
    
    demo_source = MedicalSource(
        title="Demo Mode",
        url=None,
        type="demo",
        confidence=1.0,
        relevance_score=1.0
    )
    
    return ChatResponse(
        response="Hello! I'm Elara AI, your medical assistant. This is a demo response. How can I help you with your health questions today?",
        confidence=1.0,
        sources=[demo_source],
        language_detected="en",
        processing_time=0.001,
        conversation_id="demo-conversation",
        safety_warning="This is a demo response for testing purposes."
    )
