#!/usr/bin/env python3
"""
🧪 Test script for model loading - verify DialoGPT + LoRA integration
"""

import torch
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import time

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the settings from our config
from backend.config import Settings

def test_model_loading():
    """Test loading the medical LoRA model"""
    
    print("🧪 Testing model loading...")
    settings = Settings()
    
    # Get the medical_qa model config
    model_config = settings.get_model_config("medical_qa")
    lora_path = model_config["path"]
    
    print(f"📁 Looking for LoRA adapter at: {lora_path}")
    
    if not lora_path.exists():
        print(f"❌ Error: LoRA path doesn't exist: {lora_path}")
        return False
    
    try:
        # Load the PEFT configuration
        print("📄 Loading PEFT configuration...")
        peft_config = PeftConfig.from_pretrained(str(lora_path))
        print(f"✅ Found config. Base model: {peft_config.base_model_name_or_path}")
        
        # Load tokenizer from LoRA path (not base model) since it was modified during training
        print("🔤 Loading tokenizer from LoRA adapter...")
        tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"🔤 LoRA tokenizer loaded with vocab size: {len(tokenizer)}")
        
        # Load base model and resize embeddings to match tokenizer
        print("🤖 Loading base model. This may take a moment...")
        start_time = time.time()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float32,    # Use float32 for CPU compatibility
            low_cpu_mem_usage=True        # Optimize memory usage
        )
        
        # CRITICAL: Resize model embeddings to match tokenizer vocab size
        print(f"📏 Original model vocab size: {base_model.config.vocab_size}")
        print(f"📏 Tokenizer vocab size: {len(tokenizer)}")
        
        if base_model.config.vocab_size != len(tokenizer):
            print("🔧 Resizing model embeddings to match tokenizer...")
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"✅ Model embeddings resized to: {base_model.config.vocab_size}")
        
        print(f"⏱️ Base model loaded in {time.time() - start_time:.2f} seconds")
        
        # Load the LoRA adapter
        print("🧠 Loading LoRA adapter...")
        start_time = time.time()
        
        model = PeftModel.from_pretrained(
            base_model,
            str(lora_path),
            torch_dtype=torch.float32,    # Use float32 for CPU
            is_trainable=False            # Inference mode only
        )
        
        print(f"⏱️ LoRA adapter loaded in {time.time() - start_time:.2f} seconds")
        
        # Test a simple prompt
        print("🔍 Testing inference...")
        
        test_prompt = "Human: What are the symptoms of type 2 diabetes and how is it diagnosed?\nAssistant:"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(test_prompt):].strip()
        
        print("\n📋 GENERATED RESPONSE:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        print("✅ Model loading and inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("🎉 All tests passed! Your model is working correctly.")
    else:
        print("❌ Tests failed. Please check the error messages above.")
