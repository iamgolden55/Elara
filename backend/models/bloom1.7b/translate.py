import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional
from config import settings

# Global variables to store loaded models and tokenizers
_model = None
_tokenizer = None

def get_model_and_tokenizer():
    """
    Load or get cached BLOOM 1.7B model and tokenizer
    """
    global _model, _tokenizer
    
    # Return cached model if already loaded
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    # Model path from settings
    model_path = settings.BLOOM_MODEL_PATH
    
    print(f"Loading BLOOM model from {model_path}")
    
    # Load model with low precision for efficiency
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return _model, _tokenizer

async def translate_text(text: str, 
                         source_lang: str, 
                         target_lang: str,
                         max_length: int = 512) -> str:
    """
    Translate text from source language to target language using BLOOM model
    
    Args:
        text: Text to translate
        source_lang: Source language code (e.g., 'en', 'es')
        target_lang: Target language code (e.g., 'en', 'es')
        max_length: Maximum length of generated translation
        
    Returns:
        Translated text
    """
    model, tokenizer = get_model_and_tokenizer()
    
    # Create translation prompt
    prompt = f"Translate this text from {source_lang} to {target_lang}: {text}\n\nTranslation:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.3,  # Lower temperature for more reliable translations
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the translated part
    translation = translation.split("Translation:")[-1].strip()
    
    return translation

# Language code mapping for common languages
LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
    "arabic": "ar",
    "hindi": "hi"
}

async def detect_language(text: str) -> str:
    """
    Simple language detection using the BLOOM model
    
    Args:
        text: Text to detect language for
        
    Returns:
        Detected language code
    """
    model, tokenizer = get_model_and_tokenizer()
    
    # Create language detection prompt
    prompt = f"What language is this text written in? Only respond with the language name: {text}"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # Short response needed
            temperature=0.1,    # Low temperature for consistent results
            do_sample=False,    # Deterministic generation
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    language = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the language name and convert to lowercase
    language = language.split(":")[-1].strip().lower()
    
    # Map to language code
    language_code = LANGUAGE_CODES.get(language, "en")  # Default to English if not found
    
    return language_code
