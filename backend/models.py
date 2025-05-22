#!/usr/bin/env python3
"""
🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠
                            ELARA AI - MODEL MANAGER (CONTINUED)
                        The AI Chef Squad Manager! 👨‍🍳
🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠🤖🧠
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import psutil
from datetime import datetime

try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig  # Add imports for LoRA adapters
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch/Transformers not available. Running in simulation mode.")

# Import the RAG system
import rag_utils
from rag_utils import medical_rag

import config
from config import Settings

class ModelManager:
    """
    🧠 THE AI BRAIN MANAGER
    
    Manages all AI models: medical reasoning, translation, speech, etc.
    Think of this as the head chef coordinating all the kitchen staff!
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = {}
        self.model_status = {}
        self.memory_usage = {}
        self.startup_time = time.time()
        
        # Medical knowledge base (simplified for demo)
        self.medical_knowledge = self._load_medical_knowledge()
        
        print("🤖 Model Manager initialized!")
    
    async def initialize_models(self):
        """🚀 Load all AI models on startup"""
        
        print("🔄 Initializing AI models...")
        
        if not TORCH_AVAILABLE:
            print("💡 Running in simulation mode (no actual AI models)")
            await self._initialize_simulation_mode()
            return
        
        # Initialize the RAG system
        print("🔍 Initializing RAG system...")
        await medical_rag.initialize()
        
        # Try to load lightweight models first
        try:
            await self._load_lightweight_models()
            print("✅ Lightweight models loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load AI models: {e}")
            print("💡 Falling back to simulation mode")
            await self._initialize_simulation_mode()
    
    async def _initialize_simulation_mode(self):
        """🎭 Initialize simulation mode for demo/development"""
        
        self.models = {
            "medical_qa": "simulation",
            "translator": "simulation", 
            "language_detector": "simulation",
            "symptom_analyzer": "simulation"
        }
        
        self.model_status = {
            "medical_qa": "loaded",
            "translator": "loaded",
            "language_detector": "loaded", 
            "symptom_analyzer": "loaded"
        }
        
        print("🎭 Simulation mode: Ready to demo!")
    
    async def _load_lightweight_models(self):
        """🪶 Load medical LoRA adapter and other models"""
        
        # Path to your LoRA adapter
        lora_path = self.settings.models_dir / "medical_lora"
        print(f"📁 Checking for LoRA adapter at: {lora_path}")
        
        try:
            # Check if medical LoRA adapter exists
            if lora_path.exists():
                print("💫 Found medical LoRA adapter! Loading...")
                
                # Load the PEFT configuration to get the base model name
                peft_config = PeftConfig.from_pretrained(str(lora_path))
                print(f"🔄 Base model: {peft_config.base_model_name_or_path}")
                
                try:
                    # First, try to load the tokenizer from the LoRA adapter directory
                    print("🔧 Loading tokenizer from LoRA adapter...")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
                        print("✅ Using saved tokenizer from LoRA adapter")
                    except Exception as e:
                        print(f"⚠️ Could not load saved tokenizer, using base model tokenizer: {e}")
                        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
                    
                    # Load base model
                    print("📥 Loading DialoGPT base model...")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        peft_config.base_model_name_or_path,
                        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                        low_cpu_mem_usage=True      # Memory optimization
                    )
                    
                    # Resize model embeddings to match tokenizer if needed
                    if len(tokenizer) != base_model.config.vocab_size:
                        print(f"🔧 Resizing model embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
                        base_model.resize_token_embeddings(len(tokenizer))
                    
                    # Load LoRA adapter
                    print("🧠 Applying medical LoRA adapter...")
                    model = PeftModel.from_pretrained(
                        base_model,
                        str(lora_path),
                        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                        is_trainable=False          # Inference mode only
                    )
                    
                    # Add special tokens if not in tokenizer
                    special_tokens = ["<USER>", "<ASSISTANT>", "<SYSTEM>"]
                    special_tokens_dict = {}
                    
                    new_tokens = []
                    for token in special_tokens:
                        if token not in tokenizer.get_vocab():
                            new_tokens.append(token)
                    
                    if new_tokens:
                        print(f"➕ Adding special tokens: {new_tokens}")
                        special_tokens_dict["additional_special_tokens"] = new_tokens
                        tokenizer.add_special_tokens(special_tokens_dict)
                    
                    # Store model and tokenizer directly (don't use pipeline due to PeftModel issues)
                    print("🚀 Setting up medical QA model...")
                    self.models["medical_qa"] = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "type": "lora"
                    }
                    self.model_status["medical_qa"] = "loaded"
                    print("✅ Medical LoRA model loaded successfully!")
                    
                    # Track memory usage
                    self.memory_usage["medical_qa"] = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                    
                except Exception as e:
                    print(f"❌ Error loading medical LoRA adapter: {e}")
                    print("⚠️ Falling back to gpt2 model...")
                    # Fall back to gpt2 if there's an issue with the LoRA model
                    self._load_fallback_model()
            else:
                print("⚠️ No medical LoRA adapter found at: " + str(lora_path))
                print("⚠️ Falling back to gpt2 model...")
                # Fall back to gpt2 if LoRA adapter isn't found
                self._load_fallback_model()
                
        except Exception as e:
            print(f"⚠️ Could not load models: {e}")
            raise
    
    def _load_fallback_model(self):
        """🔄 Load a fallback model if LoRA isn't available"""
        try:
            print("📥 Loading fallback text generator (gpt2)...")
            generator = pipeline(
                "text-generation",
                model="gpt2",  # Small, fast model for testing
                tokenizer="gpt2",
                device=0 if torch.cuda.is_available() else -1  # GPU if available
            )
            
            self.models["medical_qa"] = generator
            self.model_status["medical_qa"] = "loaded (fallback)"
            print("✅ Fallback model loaded!")
        except Exception as e:
            print(f"❌ Error loading fallback model: {e}")
            raise
    
    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """📚 Load medical knowledge base for RAG"""
        
        # In a real system, this would load from your vector database
        # For now, let's use some sample medical facts
        
        return {
            "diabetes": {
                "definition": "A group of metabolic disorders characterized by high blood sugar levels",
                "symptoms": ["increased urination", "increased thirst", "increased hunger", "weight loss"],
                "types": ["Type 1", "Type 2", "Gestational"],
                "treatments": ["insulin therapy", "medications", "diet modification", "exercise"]
            },
            "hypertension": {
                "definition": "A condition where blood pressure in arteries is persistently elevated",
                "symptoms": ["headache", "dizziness", "blurred vision", "often asymptomatic"],
                "risk_factors": ["obesity", "stress", "high sodium intake", "age"],
                "treatments": ["medications", "lifestyle changes", "diet modification", "exercise"]
            },
            "covid-19": {
                "definition": "Infectious disease caused by SARS-CoV-2 virus",
                "symptoms": ["fever", "dry cough", "fatigue", "loss of taste/smell"],
                "prevention": ["vaccination", "masks", "social distancing", "hand hygiene"],
                "treatments": ["supportive care", "antivirals", "monoclonal antibodies"]
            }
        }
    
    # Main AI functions
    async def detect_language(self, text: str) -> str:
        """🌍 Detect the language of input text"""
        
        # Simple language detection (in production, use proper NLP)
        if self.models.get("language_detector") == "simulation":
            # Simple heuristic: detect some common patterns
            if any(word in text.lower() for word in ["el", "la", "es", "que", "como"]):
                return "es"  # Spanish
            elif any(word in text.lower() for word in ["le", "la", "est", "que", "comment"]):
                return "fr"  # French
            else:
                return "en"  # Default to English
        
        # Real implementation would use a language detection model
        return "en"
    
    async def translate_to_english(self, text: str, source_language: str) -> str:
        """🌍➡️🇺🇸 Translate text to English"""
        
        if source_language == "en":
            return text
        
        if self.models.get("translator") == "simulation":
            # Simulation: just add a prefix to show translation happened
            return f"[Translated from {source_language}] {text}"
        
        # Real implementation would use BLOOM or another translation model
        return text
    
    async def translate_from_english(self, text: str, target_language: str) -> str:
        """🇺🇸➡️🌍 Translate text from English to target language"""
        
        if target_language == "en":
            return text
        
        if self.models.get("translator") == "simulation":
            # Simulation: add prefix to show translation
            return f"[Translated to {target_language}] {text}"
        
        # Real implementation would use BLOOM or another translation model
        return text
    
    async def translate_text(self, text: str, source_language: str, target_language: str) -> str:
        """🌍🔄🌍 Direct translation between any two languages"""
        
        # For now, route through English
        if source_language != "en":
            english_text = await self.translate_to_english(text, source_language)
        else:
            english_text = text
        
        if target_language != "en":
            return await self.translate_from_english(english_text, target_language)
        else:
            return english_text
    
    async def retrieve_medical_context(self, question: str, max_sources: int = 5) -> List[Dict[str, Any]]:
        """📚 Retrieve relevant medical context (RAG - Retrieval Augmented Generation)"""
        
        # If RAG system is available, use it
        if medical_rag.is_available():
            print("🔍 Using RAG system to retrieve relevant medical information...")
            results = await medical_rag.search(question, k=max_sources)
            
            # Convert RAG results to our format
            context = []
            for result in results:
                metadata = result.get("metadata", {})
                
                context_item = {
                    "title": f"Medical Information: {metadata.get('title', 'Unknown')}",
                    "content": result.get("content", ""),
                    "relevance_score": result.get("score", 0.0),
                    "source_type": metadata.get("source", "medical_database"),
                    "condition": metadata.get("title", "").lower()  # Use title as condition
                }
                context.append(context_item)
            
            return context
        
        # Simple keyword matching fallback (in production, use vector similarity search)
        question_lower = question.lower()
        relevant_context = []
        
        for condition, info in self.medical_knowledge.items():
            # Check if condition is mentioned in question
            if condition in question_lower or any(symptom in question_lower for symptom in info.get("symptoms", [])):
                context_item = {
                    "title": f"Medical Information: {condition.title()}",
                    "content": json.dumps(info, indent=2),
                    "relevance_score": 0.9,  # Simplified scoring
                    "source_type": "medical_database",
                    "condition": condition
                }
                relevant_context.append(context_item)
        
        # Limit to max_sources
        return relevant_context[:max_sources]
    
    async def generate_medical_response(
        self, 
        question: str, 
        context: List[Dict[str, Any]], 
        user_type: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """🧠 Generate AI response to medical question"""
        
        # Extract relevant information from context
        context_text = ""
        sources = []
        
        for ctx in context:
            context_text += f"\n{ctx['content']}"
            if include_sources:
                sources.append({
                    "title": ctx['title'],
                    "type": ctx['source_type'],
                    "confidence": 0.95,
                    "relevance_score": ctx['relevance_score']
                })
        
        # Generate response based on model type
        if self.models.get("medical_qa") == "simulation":
            response = await self._generate_simulation_response(question, context, user_type)
        else:
            response = await self._generate_ai_response(question, context, user_type)
        
        # Determine if professional consultation is needed
        needs_consultation = self._needs_professional_consultation(question, user_type)
        
        return {
            "response": response,
            "confidence": 0.85,  # Simplified confidence score
            "sources": sources,
            "needs_professional_consultation": needs_consultation,
            "context_used": len(context) > 0
        }
    
    async def _generate_simulation_response(
        self, 
        question: str, 
        context: List[Dict[str, Any]], 
        user_type: str
    ) -> str:
        """🎭 Generate simulated response for demo purposes"""
        
        # Extract condition from context if available
        condition = None
        if context:
            for ctx in context:
                if "condition" in ctx:
                    condition = ctx["condition"]
                    break
        
        # Generate response based on detected condition
        if condition and condition in self.medical_knowledge:
            info = self.medical_knowledge[condition]
            
            if user_type == "patient":
                # Patient-friendly response
                response = f"I'd be happy to help you understand {condition.replace('_', ' ').title()}. "
                response += f"This is {info['definition']}. "
                
                if "symptoms" in info:
                    response += f"Common symptoms include: {', '.join(info['symptoms'][:3])}. "
                
                response += "However, this information is for educational purposes only. "
                response += "For personalized medical advice and proper diagnosis, please consult with a healthcare professional."
                
            elif user_type == "doctor":
                # More technical response for doctors
                response = f"Regarding {condition.replace('_', ' ')}: {info['definition']}. "
                
                if "treatments" in info:
                    response += f"Treatment options typically include: {', '.join(info['treatments'])}. "
                
                response += "Please consider the individual patient's medical history and current condition when making treatment decisions."
                
            else:
                # General public response
                response = f"{condition.replace('_', ' ').title()} is {info['definition']}. "
                response += "This information is provided for educational purposes. "
                response += "For specific medical concerns, please consult a healthcare provider."
        
        else:
            # Generic response when no specific condition is detected
            if user_type == "patient":
                response = "Thank you for your question. While I can provide general medical information, "
                response += "I cannot replace professional medical advice. For specific health concerns, "
                response += "please consult with your healthcare provider who can properly assess your situation."
            else:
                response = "I can provide general medical information to help answer your question. "
                response += "However, for specific clinical guidance, please refer to current medical guidelines "
                response += "and consider individual patient factors."
        
        return response
    
    async def _generate_ai_response(
        self, 
        question: str, 
        context: List[Dict[str, Any]], 
        user_type: str
    ) -> str:
        """🤖 Generate AI response using medical LoRA model"""
        
        # Construct prompt with RAG context
        prompt = f"Human: {question}\n"
        
        # Add context if available
        context_text = ""
        
        # Check if RAG is available for enhanced context
        if medical_rag.is_available():
            rag_context = await medical_rag.get_context_for_query(question)
            if rag_context:
                context_text += rag_context
        
        # Add any other context (traditional method)
        if context:
            for ctx in context:
                if "condition" in ctx:
                    condition = ctx["condition"]
                    if condition in self.medical_knowledge:
                        info = self.medical_knowledge[condition]
                        # Add relevant info from medical knowledge
                        context_text += f"\nRelevant information about {condition}:\n"
                        context_text += f"- Definition: {info.get('definition', '')}\n"
                        if 'symptoms' in info:
                            context_text += f"- Symptoms: {', '.join(info.get('symptoms', []))}\n"
                        if 'treatments' in info:
                            context_text += f"- Treatments: {', '.join(info.get('treatments', []))}\n"
        
        # Only add context text if we have any
        if context_text:
            prompt += f"{context_text.strip()}\n"
        
        # Add the Assistant prefix for completion
        prompt += "Assistant:"
        
        try:
            # Check if we're using a pipeline or a custom model object
            if isinstance(self.models["medical_qa"], dict) and "model" in self.models["medical_qa"] and "tokenizer" in self.models["medical_qa"]:
                # We're using our custom model object with the LoRA adapter
                print("🧠 Generating response with medical LoRA model...")
                
                model = self.models["medical_qa"]["model"]
                tokenizer = self.models["medical_qa"]["tokenizer"]
                
                # Tokenize the input
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Generate text
                with torch.no_grad():
                    output_sequences = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=300,  # Longer responses for detailed medical info
                        do_sample=True,
                        temperature=0.7,       # More balanced temperature
                        top_p=0.95,            # Wider sampling
                        repetition_penalty=1.1, # Milder repetition penalty
                        no_repeat_ngram_size=2  # Smaller n-gram size for medical domain
                    )
                
                # Decode the output
                generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
                
            else:
                # We're using the transformer pipeline
                print("🧠 Generating response with fallback model...")
                
                generator = self.models["medical_qa"]
                outputs = generator(
                    prompt,
                    max_new_tokens=250,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
                
                # Extract the generated text
                generated_text = outputs[0]['generated_text']
            
            # Extract only the assistant's response (after the prompt)
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            elif "Human:" in generated_text and len(generated_text.split("Human:")) > 1:
                # For DialoGPT format, get the last response
                parts = generated_text.split("Human:")
                if len(parts) > 1 and "Assistant:" in parts[-1]:
                    response = parts[-1].split("Assistant:")[-1].strip()
                else:
                    response = generated_text[len(prompt):].strip()
            else:
                response = generated_text[len(prompt):].strip()
            
            # Clean up any remaining special tokens or artifacts
            response = response.replace("<USER>", "").replace("<ASSISTANT>", "").replace("<s>", "").replace("</s>", "").replace("<|endoftext|>", "").strip()
            
            # Format the response into a structured format with sections (for longer responses)
            if len(response.split()) > 30 and "⸻" not in response:
                # Try to structure the response if it's not already structured
                response = self._format_medical_response(response, user_type)
            
            # If the response is empty, provide a fallback
            if not response:
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            # Add safety disclaimer based on user type
            if user_type == "patient":
                response += "\n\nPlease note: This information is for educational purposes only. Always consult a healthcare professional for personalized medical advice."
            elif user_type == "doctor":
                response += "\n\nPlease use your clinical judgment and follow standard medical protocols when treating patients."
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating AI response: {e}")
            # Fallback to simulation mode
            return await self._generate_simulation_response(question, context, user_type)
    
    def _format_medical_response(self, text: str, user_type: str) -> str:
        """Format the response into a structured medical format with sections"""
        
        # Already structured, return as is
        if "⸻" in text or "Main Symptoms" in text:
            return text
        
        # Split into sentences
        sentences = text.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return text  # Too short to structure
        
        # Simplified formatting for short responses
        formatted = sentences[0] + "\n\n⸻\n\n"
        
        # Different sections based on user type
        if user_type == "patient":
            formatted += "🔑 Key Information\n"
            for sentence in sentences[1:5]:  # Take a few sentences for key info
                formatted += f"\t• {sentence}\n"
            
            if len(sentences) > 5:
                formatted += "\n⸻\n\n"
                formatted += "📌 Additional Details\n"
                for sentence in sentences[5:10]:  # Take a few more for additional info
                    formatted += f"\t• {sentence}\n"
        else:
            # More technical format for professionals
            formatted += "Clinical Information\n"
            for sentence in sentences[1:]:
                formatted += f"• {sentence}\n"
        
        return formatted
    
    def _needs_professional_consultation(self, question: str, user_type: str) -> bool:
        """🩺 Determine if the question requires professional medical consultation"""
        
        # Keywords that suggest urgent medical attention
        urgent_keywords = [
            "emergency", "urgent", "severe", "chest pain", "difficulty breathing",
            "unconscious", "bleeding", "overdose", "allergic reaction", "stroke",
            "heart attack", "suicide", "self-harm"
        ]
        
        question_lower = question.lower()
        
        # Always recommend consultation for urgent symptoms
        if any(keyword in question_lower for keyword in urgent_keywords):
            return True
        
        # For patients asking about specific symptoms or treatments
        if user_type == "patient" and any(word in question_lower for word in ["should i", "treatment", "medication", "dose"]):
            return True
        
        return False
    
    async def analyze_symptoms(
        self, 
        symptoms: List[str], 
        age: Optional[int] = None,
        gender: Optional[str] = None,
        medical_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """🏥 Analyze symptoms and provide health assessment"""
        
        # Simple symptom analysis (in production, use specialized medical reasoning)
        possible_conditions = []
        recommendations = []
        risk_level = "low"
        urgency = "routine"
        
        # Check symptoms against known conditions
        for condition, info in self.medical_knowledge.items():
            if "symptoms" in info:
                matching_symptoms = set(symptoms) & set(info["symptoms"])
                if matching_symptoms:
                    probability = len(matching_symptoms) / len(info["symptoms"])
                    possible_conditions.append({
                        "name": condition.replace("_", " ").title(),
                        "probability": probability,
                        "matching_symptoms": list(matching_symptoms),
                        "description": info["definition"]
                    })
        
        # Sort by probability
        possible_conditions.sort(key=lambda x: x["probability"], reverse=True)
        
        # Generate recommendations
        if possible_conditions:
            risk_level = "moderate" if possible_conditions[0]["probability"] > 0.5 else "low"
            recommendations = [
                "Monitor your symptoms",
                "Stay hydrated and get adequate rest",
                "Consider consulting a healthcare provider if symptoms persist or worsen"
            ]
        else:
            recommendations = [
                "Your symptoms don't match common conditions in our database",
                "Consider keeping a symptom diary",
                "Consult a healthcare provider for proper evaluation"
            ]
        
        # Check for urgent symptoms
        urgent_symptoms = ["chest pain", "difficulty breathing", "severe headache", "loss of consciousness"]
        if any(urgent in " ".join(symptoms).lower() for urgent in urgent_symptoms):
            risk_level = "high"
            urgency = "urgent"
            recommendations = ["Seek immediate medical attention"] + recommendations
        
        return {
            "risk_level": risk_level,
            "possible_conditions": possible_conditions[:3],  # Top 3 matches
            "recommendations": recommendations,
            "urgency": urgency,
            "confidence": 0.75  # Simplified confidence score
        }
    
    # System monitoring functions
    def are_models_loaded(self) -> bool:
        """✅ Check if models are loaded and ready"""
        return len(self.models) > 0 and all(
            status == "loaded" for status in self.model_status.values()
        )
    
    def get_available_models(self) -> List[str]:
        """📋 Get list of available models"""
        return list(self.models.keys())
    
    def get_model_details(self) -> Dict[str, Dict[str, Any]]:
        """📊 Get detailed information about loaded models"""
        details = {}
        
        for model_name, model in self.models.items():
            details[model_name] = {
                "status": self.model_status.get(model_name, "unknown"),
                "type": "simulation" if model == "simulation" else "neural_network",
                "memory_usage_mb": self.memory_usage.get(model_name, 0),
                "capabilities": self._get_model_capabilities(model_name)
            }
        
        return details
    
    def _get_model_capabilities(self, model_name: str) -> List[str]:
        """🎯 Get capabilities for a specific model"""
        capabilities = {
            "medical_qa": ["medical_reasoning", "question_answering", "health_information"],
            "translator": ["translation", "multilingual_support", "language_detection"],
            "language_detector": ["language_identification", "text_analysis"],
            "symptom_analyzer": ["symptom_analysis", "risk_assessment", "health_screening"]
        }
        
        return capabilities.get(model_name, [])
    
    def get_memory_usage(self) -> Dict[str, float]:
        """💾 Get current memory usage statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "total_memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception:
            return {"error": "Could not retrieve memory usage"}
    
    async def cleanup(self):
        """🧹 Clean up models and free memory"""
        print("🧹 Cleaning up models...")
        
        for model_name in list(self.models.keys()):
            try:
                if self.models[model_name] != "simulation":
                    # Clear model from memory
                    del self.models[model_name]
                    
                self.model_status[model_name] = "unloaded"
                print(f"✅ {model_name} cleaned up")
            except Exception as e:
                print(f"⚠️  Error cleaning up {model_name}: {e}")
        
        # Clear GPU cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Cleanup complete!")
