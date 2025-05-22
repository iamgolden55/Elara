#!/usr/bin/env python3
"""
🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖
                            ELARA AI BACKEND PACKAGE
                        Your Medical AI Assistant Backend! 🏥
🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖
"""

__version__ = "1.0.0"
__author__ = "Elara AI Team"
__description__ = "Medical AI Assistant Backend - Multilingual healthcare AI powered by FastAPI"

# Import main components for easy access
from .config import Settings, get_settings
from .models import ModelManager
from .schemas import (
    ChatRequest,
    ChatResponse, 
    TranslationRequest,
    HealthQuery,
    UserType
)

# Package metadata
__all__ = [
    "Settings",
    "get_settings", 
    "ModelManager",
    "ChatRequest",
    "ChatResponse",
    "TranslationRequest", 
    "HealthQuery",
    "UserType"
]

# Welcome message when package is imported
print("🏥 Elara AI Backend initialized! Ready to heal the world with AI! 🤖✨")
