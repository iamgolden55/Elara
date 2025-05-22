#!/usr/bin/env python3
"""
🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠
                            ELARA AI - MODEL MANAGER
                        The Brain of Your AI Assistant! 🧠
🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠
"""

import torch
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path
import time
import traceback
from datetime import datetime

# Import necessary libraries with error handling
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers or PEFT not available - some features will be disabled")
    TRANSFORMERS_AVAILABLE = False

class ModelManager:
    """
    🧠 ELARA AI MODEL MANAGER
    
    This class manages all AI models for Elara AI:
    - Loading your trained medical LoRA model
    - Text generation for medical Q&A
    - Model status monitoring
    - Memory management
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.models = {}
        self.tokenizers = {}
        self.model_status = {}
        self.last_loaded = {}
        
        print("🧠 ModelManager initialized!")
        
    async def initialize_models(self):
        """🚀 Initialize and load all enabled models"""
        print("🚀 Starting model initialization...")
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Cannot load models - transformers library not available")
            return False
            
        try:
            # Load the medical LoRA model (your trained model!)
            success = await self._load_medical_model()
            
            if success:
                print("✅ Medical models initialized successfully!")
                return True
            else:
                print("⚠️ Some models failed to load, but continuing...")
                return False
                
        except Exception as e:
            print(f"❌ Error during model initialization: {e}")
            traceback.print_exc()
            return False
    
    async def _load_medical_model(self):
        """🏥 Load the Mistral-7B model with LoRA adapter (your powerful model!)"""
        print("🏥 Loading Mistral-7B model...")
        
        try:
            # SWITCHED TO MISTRAL! Use the Mistral model path instead of DialoGPT
            mistral_path = self.settings.MISTRAL_MODEL_PATH
            mistral_lora_path = Path(mistral_path) / "lora_adapter"
            
            print(f"📁 Mistral model path: {mistral_path}")
            
            # Check if the Mistral model exists
            if not Path(mistral_path).exists():
                print(f"❌ Mistral model not found at: {mistral_path}")
                print("Falling back to DialoGPT model...")
                return await self._load_fallback_model()
            
            # Use CPU-friendly loading instead of 4-bit quantization
            print("🔧 Setting up CPU-friendly model loading...")
            # No quantization for CPU compatibility
            use_quantization = False
            
            try:
                # Try to import bitsandbytes to see if it works on this CPU
                import bitsandbytes as bnb
                if bnb.__version__ and torch.cuda.is_available():
                    print("✅ GPU detected! Setting up 4-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    use_quantization = True
                else:
                    print("⚠️ No GPU detected. Using CPU-friendly mode.")
            except ImportError:
                print("⚠️ BitsAndBytes not properly installed. Using CPU-friendly mode.")
            except Exception as e:
                print(f"⚠️ Quantization error: {e}. Using CPU-friendly mode.")
            
            # Load Mistral tokenizer using fast tokenizers (no sentencepiece needed!)
            print("🔤 Loading Mistral tokenizer using fast tokenizers...")
            
            # Use the local tokenizer files but with use_fast=True to avoid sentencepiece
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    mistral_path,
                    use_fast=True,  # Force using fast Rust-based tokenizers
                    local_files_only=True,  # Don't try to download
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"⚠️ Error with fast tokenizer: {e}")
                print("🔄 Trying to create a compatible tokenizer manually...")
                
                # Fallback: Create a compatible tokenizer
                from tokenizers import Tokenizer
                from transformers import PreTrainedTokenizerFast
                
                try:
                    # Try to load tokenizer.json which should be in the model folder
                    tokenizer_json_path = Path(mistral_path) / "tokenizer.json"
                    if tokenizer_json_path.exists():
                        # Create fast tokenizer from the tokenizer.json file
                        base_tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
                        tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
                        print("✅ Successfully created fast tokenizer from tokenizer.json")
                    else:
                        print("❌ tokenizer.json not found - falling back to DialoGPT")
                        return await self._load_fallback_model()
                except Exception as fallback_error:
                    print(f"❌ Fallback tokenizer error: {fallback_error}")
                    print("🚨 Falling back to DialoGPT model")
                    return await self._load_fallback_model()
            
            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            
            print(f"🔤 Mistral tokenizer loaded with vocab size: {len(tokenizer)}")
            
            # Load Mistral model with or without quantization based on capability
            print("🤖 Loading Mistral base model...")
            
            # Different loading approaches depending on hardware
            if use_quantization:
                print("🔥 Using 4-bit quantization for faster inference!")
                base_model = AutoModelForCausalLM.from_pretrained(
                    mistral_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                print("🐪 Using CPU-friendly mode (slower but more compatible)")
                base_model = AutoModelForCausalLM.from_pretrained(
                    mistral_path,
                    device_map=None,  # Use default CPU placement
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # CRITICAL: Resize model embeddings to match tokenizer vocab size
            print(f"📏 Original model vocab size: {base_model.config.vocab_size}")
            print(f"📏 Tokenizer vocab size: {len(tokenizer)}")
            
            if base_model.config.vocab_size != len(tokenizer):
                print("🔧 Resizing model embeddings to match tokenizer...")
                base_model.resize_token_embeddings(len(tokenizer))
                print(f"✅ Model embeddings resized to: {base_model.config.vocab_size}")
            
            # Check if we have a LoRA adapter for Mistral
            has_lora = mistral_lora_path.exists()
            
            # Apply LoRA adapter if available, otherwise use base model
            if has_lora:
                print(f"🧠 Found LoRA adapter at {mistral_lora_path}, applying it...")
                model = PeftModel.from_pretrained(
                    base_model,
                    str(mistral_lora_path),
                    torch_dtype=torch.float16,
                    is_trainable=False  # Inference mode
                )
            else:
                print("🚨 No LoRA adapter found for Mistral, using base model")
                model = base_model
            
            # Store the model and tokenizer
            self.models["medical_qa"] = model
            self.tokenizers["medical_qa"] = tokenizer
            self.model_status["medical_qa"] = "loaded"
            self.last_loaded["medical_qa"] = datetime.now()
            
            print("✅ Medical LoRA model loaded successfully!")
            
            # Test the model quickly
            await self._test_medical_model()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load medical model: {e}")
            traceback.print_exc()
            self.model_status["medical_qa"] = "failed"
            return False
    
    async def _load_fallback_model(self):
        """🚨 Load fallback DialoGPT model if Mistral is not available"""
        print("🚨 Loading fallback DialoGPT model...")
        
        try:
            # Get model configuration for the original DialoGPT model
            model_config = self.settings.get_model_config("medical_qa")
            lora_path = model_config["path"]
            
            print(f"📁 Fallback LoRA path: {lora_path}")
            
            if not lora_path.exists():
                print(f"❌ Fallback LoRA adapter not found at: {lora_path}")
                return False
            
            # Load PEFT configuration
            print("📄 Loading fallback PEFT configuration...")
            peft_config = PeftConfig.from_pretrained(str(lora_path))
            base_model_name = peft_config.base_model_name_or_path
            
            print(f"🤖 Fallback base model: {base_model_name}")
            
            # Load tokenizer from LoRA path
            print("🔤 Loading fallback tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
            
            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"🔤 Fallback tokenizer loaded with vocab size: {len(tokenizer)}")
            
            # Load base model
            print("🤖 Loading fallback base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                low_cpu_mem_usage=True
            )
            
            # Resize model embeddings if needed
            if base_model.config.vocab_size != len(tokenizer):
                print("🔧 Resizing fallback model embeddings...")
                base_model.resize_token_embeddings(len(tokenizer))
            
            # Load LoRA adapter
            print("🧠 Loading fallback LoRA adapter...")
            model = PeftModel.from_pretrained(
                base_model,
                str(lora_path),
                torch_dtype=torch.float32,
                is_trainable=False  # Inference mode
            )
            
            # Store the model and tokenizer
            self.models["medical_qa"] = model
            self.tokenizers["medical_qa"] = tokenizer
            self.model_status["medical_qa"] = "loaded_fallback"
            self.last_loaded["medical_qa"] = datetime.now()
            
            print("⚠️ Fallback model loaded successfully - using DialoGPT instead of Mistral")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load fallback model: {e}")
            traceback.print_exc()
            self.model_status["medical_qa"] = "failed"
            return False
    
    async def _test_medical_model(self):
        """🧪 Quick test of the medical model"""
        print("🧪 Testing medical model...")
        
        try:
            model = self.models["medical_qa"]
            tokenizer = self.tokenizers["medical_qa"]
            
            test_prompt = "<|system|>\nYou are Elara, a professional medical AI assistant.\n<|user|>\nWhat is diabetes?\n<|assistant|>\n"
            
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = response[len(test_prompt):].strip()
            
            print(f"🧪 Test response preview: {generated_part[:100]}...")
            print("✅ Medical model test successful!")
            
        except Exception as e:
            print(f"⚠️ Model test failed: {e}")
    
    async def generate_medical_response(
        self, 
        question: str, 
        context: Optional[str] = None,
        user_type: str = "patient",
        include_sources: bool = False
    ) -> Dict[str, Any]:
        """🏥 Generate medical response using your trained LoRA model"""
        
        if "medical_qa" not in self.models:
            return {
                "response": "I'm sorry, my medical knowledge model is not available right now. Please try again later.",
                "confidence": 0.0,
                "sources": [],
                "needs_professional_consultation": True
            }
        
        try:
            model = self.models["medical_qa"]
            tokenizer = self.tokenizers["medical_qa"]
            
            # Create prompt in the format your model was trained on
            system_msg = "You are Elara, a professional medical AI assistant trained to provide accurate, evidence-based medical information. Always provide helpful, safe, and compassionate responses."
            
            if user_type == "doctor":
                system_msg += " You are speaking with a healthcare professional, so you can use medical terminology."
            else:
                system_msg += " You are speaking with a patient, so use clear, simple language."
            
            # Format the prompt
            if context:
                prompt = f"<|system|>\n{system_msg}\n\nRelevant context: {context}\n<|user|>\n{question}\n<|assistant|>\n"
            else:
                prompt = f"<|system|>\n{system_msg}\n<|user|>\n{question}\n<|assistant|>\n"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.settings.generation_config.get("max_tokens", 300),
                    temperature=self.settings.generation_config.get("temperature", 0.7),
                    top_p=self.settings.generation_config.get("top_p", 0.9),
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode the response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_response = full_response[len(prompt):].strip()
            
            # Clean up the response
            generated_response = self._clean_medical_response(generated_response)
            
            # Add medical disclaimer
            if self.settings.medical_config.get("auto_add_disclaimers", True):
                disclaimer = self.settings.get_medical_disclaimer()
                generated_response += f"\n\n⚠️ {disclaimer}"
            
            # Create proper MedicalSource objects for sources
            sources = []
            if include_sources:
                from schemas import MedicalSource
                sources = [
                    MedicalSource(
                        title="Medical Knowledge Base",
                        url=None,
                        type="knowledge_base",
                        confidence=0.8,
                        relevance_score=0.9
                    )
                ]
            
            return {
                "response": generated_response,
                "confidence": 0.8,  # You could implement confidence scoring
                "sources": sources,
                "needs_professional_consultation": self._needs_professional_consultation(question)
            }
            
        except Exception as e:
            print(f"❌ Error generating medical response: {e}")
            traceback.print_exc()
            
            return {
                "response": "I apologize, but I'm having trouble processing your question right now. Please try rephrasing it or contact a healthcare professional directly.",
                "confidence": 0.0,
                "sources": [],
                "needs_professional_consultation": True
            }
    
    def _clean_medical_response(self, response: str) -> str:
        """🧹 Clean and format the medical response"""
        # Remove any training artifacts or unwanted tokens
        response = response.replace("<|system|>", "")
        response = response.replace("<|user|>", "")
        response = response.replace("<|assistant|>", "")
        response = response.replace("</response>", "")
        response = response.replace("[END]", "")
        
        # Remove excessive newlines
        while "\n\n\n" in response:
            response = response.replace("\n\n\n", "\n\n")
        
        return response.strip()
    
    def _needs_professional_consultation(self, question: str) -> bool:
        """🚨 Determine if question needs professional consultation"""
        emergency_keywords = self.settings.medical_config.get("emergency_keywords", [])
        
        question_lower = question.lower()
        
        # Check for emergency keywords
        if any(keyword in question_lower for keyword in emergency_keywords):
            return True
        
        # Check for specific medical advice requests
        advice_keywords = ["should i take", "what medication", "how much should i", "am i having"]
        if any(keyword in question_lower for keyword in advice_keywords):
            return True
        
        return False
    
    async def detect_language(self, text: str) -> str:
        """🌍 Detect language (simplified implementation)"""
        # For now, just return 'en' - you could implement proper language detection
        return "en"
    
    async def translate_to_english(self, text: str, source_language: str) -> str:
        """🔄 Translate text to English"""
        # For now, just return the original text
        # You could implement BLOOM translation here
        return text
    
    async def translate_from_english(self, text: str, target_language: str) -> str:
        """🔄 Translate from English to target language"""
        # For now, just return the original text
        # You could implement BLOOM translation here
        return text
    
    async def retrieve_medical_context(self, question: str, max_sources: int = 5) -> str:
        """📚 Retrieve relevant medical context (RAG)"""
        # For now, return empty context
        # You could implement FAISS vector search here
        return ""
    
    async def analyze_symptoms(
        self, 
        symptoms: List[str], 
        age: Optional[int] = None,
        gender: Optional[str] = None,
        medical_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """🔍 Analyze symptoms using medical model"""
        
        # Create a symptom analysis prompt
        symptom_text = ", ".join(symptoms)
        context_info = []
        
        if age:
            context_info.append(f"Age: {age}")
        if gender:
            context_info.append(f"Gender: {gender}")
        if medical_history:
            context_info.append(f"Medical history: {', '.join(medical_history)}")
        
        context = " | ".join(context_info) if context_info else ""
        
        question = f"A person is experiencing the following symptoms: {symptom_text}. {context}. What could be the possible causes and what should they do?"
        
        # Use the medical model to analyze
        result = await self.generate_medical_response(
            question=question,
            user_type="patient",
            include_sources=True
        )
        
        return {
            "analysis": result["response"],
            "confidence": result["confidence"],
            "recommendations": [
                "Consult a healthcare professional for proper evaluation",
                "Monitor symptoms and seek immediate care if they worsen"
            ],
            "urgency_level": "medium"  # You could implement urgency classification
        }
    
    def are_models_loaded(self) -> bool:
        """✅ Check if models are loaded"""
        return len([status for status in self.model_status.values() if status == "loaded"]) > 0
    
    def get_available_models(self) -> List[str]:
        """📋 Get list of available models"""
        return [name for name, status in self.model_status.items() if status == "loaded"]
    
    def get_model_details(self) -> Dict[str, Any]:
        """📊 Get detailed model information"""
        return {
            "models": self.model_status,
            "last_loaded": {k: v.isoformat() for k, v in self.last_loaded.items()},
            "total_models": len(self.model_status),
            "loaded_models": len([s for s in self.model_status.values() if s == "loaded"])
        }
    
    def get_model_status(self) -> Dict[str, str]:
        """📈 Get current model status"""
        return self.model_status.copy()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """💾 Get memory usage information"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "memory_used_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "models_loaded": len(self.models)
            }
        except ImportError:
            return {
                "memory_used_mb": "unknown",
                "memory_percent": "unknown", 
                "models_loaded": len(self.models)
            }
    
    async def cleanup(self):
        """🧹 Clean up models and free memory"""
        print("🧹 Cleaning up models...")
        
        for model_name in list(self.models.keys()):
            try:
                if model_name in self.models:
                    del self.models[model_name]
                if model_name in self.tokenizers:
                    del self.tokenizers[model_name]
                    
                self.model_status[model_name] = "unloaded"
                print(f"✅ {model_name} cleaned up")
            except Exception as e:
                print(f"⚠️  Error cleaning up {model_name}: {e}")
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Cleanup complete!")
