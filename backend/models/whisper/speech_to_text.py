import whisper
import torch
import base64
import io
import numpy as np
from typing import Optional, Dict, Any
from config import settings

# Global variable for loaded model
_model = None

def get_model():
    """
    Load or get cached Whisper model
    """
    global _model
    
    # Return cached model if already loaded
    if _model is not None:
        return _model
    
    # Model size from settings
    model_size = settings.WHISPER_MODEL_PATH
    
    print(f"Loading Whisper {model_size} model")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = whisper.load_model(model_size, device=device)
    
    return _model

async def transcribe_audio(audio_data: str, language: Optional[str] = None) -> str:
    """
    Transcribe audio using OpenAI's Whisper model
    
    Args:
        audio_data: Base64-encoded audio data
        language: Optional language code to guide transcription
        
    Returns:
        Transcribed text
    """
    # Load model
    model = get_model()
    
    # Decode base64 audio data
    try:
        audio_bytes = base64.b64decode(audio_data)
        audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        raise ValueError(f"Error decoding audio data: {str(e)}")
    
    # Set transcription options
    options = {}
    if language:
        options["language"] = language
    
    # Transcribe audio
    result = model.transcribe(audio_np, **options)
    
    return result["text"]

async def detect_language_from_audio(audio_data: str) -> str:
    """
    Detect language from audio using Whisper
    
    Args:
        audio_data: Base64-encoded audio data
        
    Returns:
        Detected language code
    """
    # Load model
    model = get_model()
    
    # Decode base64 audio data
    try:
        audio_bytes = base64.b64decode(audio_data)
        audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        raise ValueError(f"Error decoding audio data: {str(e)}")
    
    # Detect language
    audio_features = whisper.pad_or_trim(audio_np)
    
    # Log mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_features).to(model.device)
    
    # Detect language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    
    return detected_language
