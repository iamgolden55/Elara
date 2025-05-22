#!/usr/bin/env python3
"""
🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖
                            ELARA AI - MAIN APPLICATION
                        The Heart of Your Medical AI Assistant!
🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖🏥🤖
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our custom modules (we'll create these!)
from routes import chat_router
from models import ModelManager
from config import Settings

# Create FastAPI app instance - THE BRAIN! 🧠
app = FastAPI(
    title="Elara AI Medical Assistant",
    description="A multilingual AI assistant for medical questions and support",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Load configuration
settings = Settings()

# Add CORS middleware - allows frontend to talk to backend 🌐
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Global model manager - our AI chef! 👨‍🍳
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI models when server starts"""
    global model_manager
    
    print("🚀 ELARA AI STARTING UP...")
    print("=" * 50)
    
    try:
        # Initialize model manager
        model_manager = ModelManager(settings)
        
        # Load models (start with lightweight ones)
        await model_manager.initialize_models()
        
        print("✅ Elara AI is ready to help!")
        print("🏥 Medical AI Assistant: ONLINE")
        print("🌍 Multilingual Support: READY")
        print("🤖 Models: LOADED")
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        print("💡 Starting without AI models (API-only mode)")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when server shuts down"""
    global model_manager
    
    print("🛑 ELARA AI SHUTTING DOWN...")
    
    if model_manager:
        await model_manager.cleanup()
    
    print("✅ Cleanup complete. Goodbye! 👋")

# Health check endpoint - is the server alive? ❤️
@app.get("/health")
async def health_check():
    """Check if the API is running and models are loaded"""
    global model_manager
    
    status = {
        "status": "healthy",
        "service": "Elara AI Medical Assistant",
        "version": "1.0.0",
        "models_loaded": False
    }
    
    if model_manager:
        status["models_loaded"] = model_manager.are_models_loaded()
        status["available_models"] = model_manager.get_available_models()
    
    return status

# Root endpoint - welcome message! 👋
@app.get("/")
async def root():
    """Welcome message for Elara AI"""
    return {
        "message": "Welcome to Elara AI Medical Assistant! 🏥🤖",
        "description": "Your multilingual AI companion for medical questions",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "features": [
            "🧠 Medical Q&A with AI",
            "🌍 200+ languages supported",
            "🎙️ Voice input/output",
            "📊 Evidence-based responses",
            "🔒 Privacy-first design"
        ]
    }

# Include chat routes - the main functionality! 💬
app.include_router(chat_router, prefix="/chat", tags=["chat"])

# Global exception handler - catch all errors! 🛡️
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors gracefully"""
    print(f"❌ Unexpected error: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong. Please try again.",
            "support": "Contact support if this persists."
        }
    )

# Add middleware to inject model manager into requests
@app.middleware("http")
async def add_model_manager(request, call_next):
    """Add model manager to request state"""
    request.state.model_manager = model_manager
    response = await call_next(request)
    return response

# Development server runner 🏃‍♂️
if __name__ == "__main__":
    print("\n🎯 STARTING ELARA AI DEVELOPMENT SERVER")
    print("=" * 50)
    print("🏥 Medical AI Assistant - FastAPI Backend")
    print("🌐 Access at: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("💡 Stop with: Ctrl+C")
    print("=" * 50)
    
    # Run the server with auto-reload for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-restart on code changes
        log_level="info"
    )
