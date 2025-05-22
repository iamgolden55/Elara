#!/usr/bin/env python3
"""
📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬
                            ELARA AI - DATA SCHEMAS  
                        The Recipe Cards for our API! 🧾
📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬📋🔬
"""

from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

# User types for personalized responses
class UserType(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    MEDICAL_STUDENT = "medical_student"
    RESEARCHER = "researcher"
    GENERAL_PUBLIC = "general_public"

# Request schemas (what users send to us)
class ChatRequest(BaseModel):
    """💬 Main chat request - when user asks a medical question"""
    
    question: str = Field(..., min_length=1, max_length=1000, description="Medical question to ask")
    language: Optional[str] = Field(None, description="User's language (auto-detected if not provided)")
    user_type: UserType = Field(UserType.GENERAL_PUBLIC, description="Type of user for personalized responses")
    conversation_id: Optional[str] = Field(None, description="ID to continue existing conversation")
    include_sources: bool = Field(True, description="Include sources in response")
    max_response_length: int = Field(500, ge=50, le=2000, description="Maximum response length in words")
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the symptoms of diabetes?",
                "language": "en",
                "user_type": "patient",
                "include_sources": True
            }
        }

class TranslationRequest(BaseModel):
    """🌍 Translation request - convert text between languages"""
    
    text: str = Field(..., min_length=1, max_length=5000)
    source_language: str = Field(..., min_length=2, max_length=5, description="Source language code (e.g. 'en', 'es')")
    target_language: str = Field(..., min_length=2, max_length=5, description="Target language code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "What is diabetes?",
                "source_language": "en", 
                "target_language": "es"
            }
        }

class HealthQuery(BaseModel):
    """🏥 Health assessment request - symptom analysis"""
    
    symptoms: List[str] = Field(..., min_items=1, max_items=10, description="List of symptoms")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$", description="Patient gender")
    medical_history: Optional[List[str]] = Field(None, max_items=5, description="Relevant medical history")
    urgency_level: Optional[str] = Field("normal", pattern="^(low|normal|high|emergency)$")
    
    @validator('symptoms', each_item=True)
    def symptom_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Symptom cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": ["headache", "fever", "nausea"],
                "age": 30,
                "gender": "male",
                "medical_history": ["hypertension"],
                "urgency_level": "normal"
            }
        }

# Response schemas (what we send back to users)
class MedicalSource(BaseModel):
    """📚 Medical source reference"""
    
    title: str
    url: Optional[str] = None
    type: str = Field(..., description="Source type: 'journal', 'guideline', 'database', etc.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    relevance_score: float = Field(..., ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    """💡 Main chat response - AI's answer to user question"""
    
    response: str = Field(..., description="AI-generated response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the response")
    sources: List[MedicalSource] = Field(default=[], description="Supporting sources")
    language_detected: str = Field(..., description="Detected user language")
    processing_time: float = Field(..., ge=0.0, description="Time taken to generate response (seconds)")
    conversation_id: str = Field(..., description="Conversation identifier")
    safety_warning: Optional[str] = Field(None, description="Safety disclaimer if needed")
    follow_up_questions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Diabetes is a condition where blood sugar levels are too high...",
                "confidence": 0.95,
                "sources": [
                    {
                        "title": "Diabetes Overview - Mayo Clinic",
                        "url": "https://mayoclinic.org/diabetes",
                        "type": "medical_website",
                        "confidence": 0.98,
                        "relevance_score": 0.92
                    }
                ],
                "language_detected": "en",
                "processing_time": 1.25,
                "conversation_id": "conv_abc123",
                "safety_warning": "This information is for educational purposes only."
            }
        }

class ConversationMessage(BaseModel):
    """💬 Single message in a conversation"""
    
    type: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime
    language: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    sources: Optional[List[MedicalSource]] = None

class ConversationHistory(BaseModel):
    """📚 Full conversation history"""
    
    conversation_id: str
    started_at: datetime
    last_updated: datetime
    messages: List[ConversationMessage] = []
    user_type: Optional[UserType] = None
    language: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_abc123",
                "started_at": "2024-01-15T10:30:00Z",
                "last_updated": "2024-01-15T10:35:00Z",
                "messages": [
                    {
                        "type": "user",
                        "content": "What is diabetes?",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "language": "en"
                    },
                    {
                        "type": "assistant",
                        "content": "Diabetes is a condition...",
                        "timestamp": "2024-01-15T10:30:05Z",
                        "confidence": 0.95
                    }
                ]
            }
        }

# Health assessment response
class HealthAssessment(BaseModel):
    """🏥 Health assessment result"""
    
    risk_level: str = Field(..., pattern="^(low|moderate|high|emergency)$")
    possible_conditions: List[Dict[str, Any]] = Field(default=[])
    recommendations: List[str] = Field(default=[])
    urgency: str = Field(..., pattern="^(routine|soon|urgent|emergency)$")
    disclaimer: Dict[str, str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_level": "moderate",
                "possible_conditions": [
                    {
                        "name": "Common Cold",
                        "probability": 0.7,
                        "description": "Viral upper respiratory infection"
                    }
                ],
                "recommendations": [
                    "Rest and stay hydrated",
                    "Monitor symptoms",
                    "Consider seeing a doctor if symptoms worsen"
                ],
                "urgency": "routine",
                "disclaimer": {
                    "warning": "This is not a medical diagnosis",
                    "advice": "Consult a healthcare professional"
                },
                "confidence": 0.78
            }
        }

# Model status responses
class ModelStatus(BaseModel):
    """🤖 AI model status information"""
    
    model_name: str
    status: str = Field(..., pattern="^(loaded|loading|error|not_available)$")
    memory_usage_mb: Optional[float] = None
    last_used: Optional[datetime] = None
    version: Optional[str] = None
    capabilities: List[str] = Field(default=[])

class SystemStatus(BaseModel):
    """⚡ Overall system status"""
    
    service_name: str = "Elara AI Medical Assistant"
    version: str = "1.0.0"
    status: str = Field(..., pattern="^(healthy|degraded|down)$")
    models: List[ModelStatus] = Field(default=[])
    uptime_seconds: float
    memory_usage_percent: Optional[float] = None
    last_health_check: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "service_name": "Elara AI Medical Assistant",
                "version": "1.0.0", 
                "status": "healthy",
                "models": [
                    {
                        "model_name": "mistral-7b-medical",
                        "status": "loaded",
                        "memory_usage_mb": 4096.5,
                        "capabilities": ["medical_qa", "reasoning"]
                    }
                ],
                "uptime_seconds": 3600.0,
                "memory_usage_percent": 65.3,
                "last_health_check": "2024-01-15T10:30:00Z"
            }
        }

# Error response schemas
class ErrorResponse(BaseModel):
    """❌ Error response format"""
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input provided",
                "details": {"field": "question", "issue": "cannot be empty"},
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_abc123"
            }
        }
