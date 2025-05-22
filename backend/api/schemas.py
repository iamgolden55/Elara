from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask Elara AI")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'es', 'fr')")
    context: Optional[List[Dict[str, Any]]] = Field(default=None, description="Additional context for the question")
    user_type: Optional[str] = Field(default=None, description="User type (e.g., 'doctor', 'patient')")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The AI-generated answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used to generate the answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")

class TranscriptionRequest(BaseModel):
    audio_data: str = Field(..., description="Base64-encoded audio data")
    language: Optional[str] = Field(default=None, description="Expected language code, if known")

class VoiceResponse(BaseModel):
    transcription: str = Field(..., description="Transcribed text from audio")
    detected_language: Optional[str] = Field(default=None, description="Detected language code")
    success: bool = Field(..., description="Whether transcription was successful")

class UserBase(BaseModel):
    email: str
    full_name: Optional[str] = None
    role: str = "patient"  # Default role

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
