#!/usr/bin/env python3
"""
Test script for Mistral 7B model
Verifies the model loads correctly and can generate medical responses
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_mistral():
    print("🔥 TESTING MISTRAL 7B FOR ELARA AI 🔥")
    print("=" * 50)
    
    model_path = "/Users/new/elara_main/models_files/mistral"
    
    try:
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded! Vocab size: {tokenizer.vocab_size:,}")
        
        print("\n🧠 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ Model loaded successfully!")
        
        # Test with a medical query
        print("\n🩺 Testing medical reasoning...")
        test_prompt = """### Human: A 65-year-old patient presents with chest pain and shortness of breath. What should be the immediate assessment priorities?

### Assistant:"""
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_response = response.split("### Assistant:")[-1].strip()
        
        print("\n🎯 MODEL RESPONSE:")
        print("-" * 40)
        print(assistant_response)
        print("-" * 40)
        
        print("\n🎉 SUCCESS! Mistral 7B is ready for LoRA fine-tuning!")
        print(f"✨ Model: Hermes-2-Pro-Mistral-7B")
        print(f"📊 Parameter count: ~7 billion")
        print(f"🔧 Ready for medical_professional_lora training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_mistral()
    sys.exit(0 if success else 1)
