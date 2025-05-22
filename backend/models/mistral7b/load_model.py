import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import Dict, Any, Optional
from config import settings
import os

# Global variables to store loaded models and tokenizers
_model = None
_tokenizer = None

def get_model_and_tokenizer():
    """
    Load or get cached Mistral 7B model and tokenizer
    """
    global _model, _tokenizer
    
    # Return cached model if already loaded
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    # Model path from settings
    model_path = settings.MISTRAL_MODEL_PATH
    
    # Check for any LoRA adapters
    lora_path = os.path.join(os.path.dirname(model_path), "lora_adapter")
    has_lora = os.path.exists(lora_path)
    
    print(f"Loading Mistral model from {model_path}")
    
    # Configure quantization for lower memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"
    
    # Apply LoRA adapter if available
    if has_lora:
        print(f"Applying LoRA adapter from {lora_path}")
        _model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        _model = base_model
    
    return _model, _tokenizer

async def generate_text(prompt: str, 
                       max_new_tokens: int = 512, 
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       system_prompt: Optional[str] = None) -> str:
    """
    Generate text response using Mistral 7B model
    """
    model, tokenizer = get_model_and_tokenizer()
    
    # Prepare system prompt if provided
    if system_prompt:
        full_prompt = f"<s>[INST] {system_prompt} [/INST]</s>\n\n<s>[INST] {prompt} [/INST]"
    else:
        # Default medical assistant system prompt
        default_system = "You are Elara, a helpful medical AI assistant. Provide accurate, evidence-based medical information. For serious concerns, always advise consulting with a healthcare professional."
        full_prompt = f"<s>[INST] {default_system} [/INST]</s>\n\n<s>[INST] {prompt} [/INST]"
    
    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part after the prompt
    response = response.split(prompt)[-1].strip()
    
    return response
