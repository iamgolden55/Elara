from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import List, Optional
from .schemas import QuestionRequest, AnswerResponse, TranscriptionRequest, VoiceResponse
from auth.oauth2 import get_current_user, get_current_active_user
from agents.rag_agent import generate_response
from models.whisper.speech_to_text import transcribe_audio

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, current_user = Depends(get_current_active_user)):
    """
    Process a text question and return an AI-generated response
    """
    try:
        # Generate response using RAG agent
        response = await generate_response(
            question=request.question,
            language=request.language,
            user_id=current_user.id if current_user else None,
            context=request.context
        )
        
        return AnswerResponse(
            answer=response["answer"],
            sources=response.get("sources", []),
            metadata=response.get("metadata", {})
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

@router.post("/voice", response_model=VoiceResponse)
async def process_voice_input(request: TranscriptionRequest, current_user = Depends(get_current_user)):
    """
    Process voice audio and return the transcription
    """
    try:
        # Transcribe audio using Whisper
        transcription = await transcribe_audio(
            audio_data=request.audio_data,
            language=request.language
        )
        
        return VoiceResponse(
            transcription=transcription,
            detected_language=request.language,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error transcribing audio: {str(e)}"
        )

@router.get("/healthcheck")
async def healthcheck():
    """
    Health check endpoint for API status
    """
    return {"status": "healthy"}
