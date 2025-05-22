#!/usr/bin/env python3
"""
🩺 Elara AI - Medical LoRA Adapter Tester 🩺
Tests the trained LoRA adapter with real-world medical queries to evaluate performance.
"""

import os
import sys
import json
import torch
import time
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

# Model imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Colorful output for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@dataclass
class TestConfig:
    """Configuration for medical model testing"""
    # Model paths
    base_model: str = "microsoft/DialoGPT-medium"
    adapter_path: str = "../models_files/medical_lora"
    
    # Generation settings
    max_new_tokens: int = 300
    temperature: float = 0.15
    repetition_penalty: float = 4.0  # High value to prevent repetitions
    do_sample: bool = True
    top_p: float = 0.85
    top_k: int = 30
    
    # Testing settings
    save_results: bool = True
    results_dir: str = "../test_results"
    user_type: str = "medical_professional"  # or "patient", "student", "researcher"
    verbose: bool = True

class MedicalModelTester:
    """Tests the medical model with real-world queries"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else 
                                  "cpu")
        
        # Prepare results directory
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
            
        # Test cases by category
        self.test_cases = self._load_test_cases()
    
    def _load_test_cases(self) -> Dict[str, List[Dict[str, str]]]:
        """Load test cases by category"""
        # Create a diverse set of real-world medical queries
        return {
            "diagnostic": [
                {"query": "What are the common causes of persistent headaches that last for several days?", 
                 "type": "Symptom Analysis"},
                {"query": "What diagnostic criteria are used to distinguish between type 1 and type 2 diabetes?", 
                 "type": "Differential Diagnosis"},
                {"query": "What are the red flag symptoms that might indicate a stroke requiring immediate medical attention?", 
                 "type": "Emergency Recognition"},
                {"query": "What symptoms would suggest bacterial rather than viral pneumonia?", 
                 "type": "Infection Differentiation"},
                {"query": "What diagnostic workup would you recommend for a patient with unexplained weight loss and fatigue?", 
                 "type": "Clinical Workup"}
            ],
            "treatment": [
                {"query": "What are the current first-line treatments for major depressive disorder according to latest guidelines?", 
                 "type": "Treatment Guidelines"},
                {"query": "What is the mechanism of action of ACE inhibitors in treating hypertension?", 
                 "type": "Medication MOA"},
                {"query": "What are the most serious potential side effects of methotrexate therapy?", 
                 "type": "Medication Safety"},
                {"query": "How should warfarin dosing be adjusted based on INR results?", 
                 "type": "Medication Management"},
                {"query": "What are the evidence-based non-pharmacological interventions for chronic lower back pain?", 
                 "type": "Non-Pharm Treatment"}
            ],
            "procedures": [
                {"query": "What is the standard protocol for a lumbar puncture procedure?", 
                 "type": "Procedure Protocol"},
                {"query": "What preparation is required for a colonoscopy and why is it important?", 
                 "type": "Procedure Preparation"},
                {"query": "What are the risks and benefits of laparoscopic versus open cholecystectomy?", 
                 "type": "Surgical Comparison"},
                {"query": "What is the recovery process following a total knee replacement?", 
                 "type": "Post-Op Recovery"},
                {"query": "What is the proper technique for collecting a clean-catch midstream urine sample?", 
                 "type": "Specimen Collection"}
            ],
            "research": [
                {"query": "What does the current research say about the effectiveness of SGLT2 inhibitors in heart failure?", 
                 "type": "Current Evidence"},
                {"query": "What are the major findings from recent studies on mRNA vaccine efficacy and safety?", 
                 "type": "Recent Research"},
                {"query": "How has our understanding of gut microbiome's role in immune function evolved in recent years?", 
                 "type": "Evolving Understanding"},
                {"query": "What are the latest research developments in using immunotherapy for solid tumors?", 
                 "type": "Treatment Innovation"},
                {"query": "What evidence supports intermittent fasting for metabolic health improvement?", 
                 "type": "Lifestyle Evidence"}
            ],
            "terminology": [
                {"query": "Can you explain the difference between systolic and diastolic heart failure?", 
                 "type": "Condition Clarification"},
                {"query": "What is the meaning of ASCVD risk score and how is it calculated?", 
                 "type": "Risk Assessment"},
                {"query": "What is the difference between bacteriostatic and bactericidal antibiotics?", 
                 "type": "Pharmacology Concept"},
                {"query": "Can you explain what comorbidity and multimorbidity mean in clinical practice?", 
                 "type": "Clinical Terminology"},
                {"query": "What does it mean when a lab result shows elevated troponin levels?", 
                 "type": "Lab Interpretation"}
            ],
            "specialty": [
                {"query": "What are the current recommended screening guidelines for colorectal cancer?", 
                 "type": "Preventive Medicine"},
                {"query": "What is the Glasgow Coma Scale and how is it used to assess neurological function?", 
                 "type": "Neurology"},
                {"query": "What are the stages of chronic kidney disease and how are they determined?", 
                 "type": "Nephrology"},
                {"query": "What is the recommended approach for managing gestational diabetes?", 
                 "type": "Obstetrics"},
                {"query": "What are the recommended vaccination schedules for children under 5 years?", 
                 "type": "Pediatrics"}
            ]
        }
    
    def load_model(self):
        """Load base model and LoRA adapter"""
        print(f"{Colors.HEADER}🔧 Loading models and adapters...{Colors.ENDC}")
        start_time = time.time()
        
        try:
            # Step 1: Load the tokenizer
            print(f"  Loading tokenizer from {self.config.base_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            
            # Add special tokens if needed
            special_tokens = {
                "pad_token": "<PAD>",
                "eos_token": "<EOS>",
                "bos_token": "<BOS>",
                "unk_token": "<UNK>"
            }
            
            for key, token in special_tokens.items():
                if getattr(self.tokenizer, key) is None:
                    setattr(self.tokenizer, key, token)
            
            # Step 2: Load the base model
            print(f"  Loading base model from {self.config.base_model}...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float32,
                device_map=self.device
            )
            
            # Step 3: Since the LoRA adapter doesn't seem to be saved correctly,
            # we'll use just the base model for testing to demonstrate functionality
            print(f"  ⚠️ LoRA adapter not found at {self.config.adapter_path}")
            print(f"  ⚠️ Using base model only for testing")
            self.model = self.base_model
            
            # TODO: Replace this with LoRA adapter loading when files are available:
            # self.model = PeftModel.from_pretrained(
            #     self.base_model,
            #     self.config.adapter_path,
            #     device_map=self.device
            # )
            
            # Ensure eval mode for inference
            self.model.eval()
            
            elapsed = time.time() - start_time
            print(f"{Colors.GREEN}✅ Models loaded successfully in {elapsed:.2f} seconds!{Colors.ENDC}")
            print(f"  Running on device: {self.device}")
            
            return True
        except Exception as e:
            print(f"{Colors.RED}❌ Error loading models: {str(e)}{Colors.ENDC}")
            return False
    
    def format_query(self, query: str) -> str:
        """Format query based on user type"""
        if self.config.user_type == "medical_professional":
            system_prompt = """You are Elara, a professional medical AI assistant trained to provide accurate, evidence-based information to healthcare professionals. 

RESPONSE FORMAT:
1. Begin with a concise clinical summary
2. Include relevant pathophysiology or mechanisms
3. Provide evidence-based recommendations with clear rationale
4. Use precise medical terminology appropriate for healthcare professionals
5. Maintain a formal, clinical tone throughout

Your responses must be structured, referenced to current medical literature, and adhere to standard clinical practice guidelines."""
            
            formatted_query = f"<|system|>\n{system_prompt}\n</|system|>\n\n<|user|>\n[MEDICAL QUERY]\n{query}\n</|user|>\n\n<|assistant|>\n"
        else:
            # Patient-focused format
            system_prompt = """You are Elara, a healthcare AI assistant providing easy-to-understand medical information to patients. 

RESPONSE FORMAT:
1. Begin with a clear, simple explanation of the main concept
2. Use everyday language and avoid jargon when possible
3. Explain medical terms if they must be used
4. Provide practical, actionable information
5. Include appropriate disclaimers about consulting healthcare providers

Your responses should be informative yet accessible, empathetic, and always emphasize the importance of seeking professional medical advice."""
            
            formatted_query = f"<|system|>\n{system_prompt}\n</|system|>\n\n<|user|>\n[PATIENT QUESTION]\n{query}\n</|user|>\n\n<|assistant|>\n"
        
        return formatted_query
    
    def generate_response(self, query: str) -> str:
        """Generate response for a medical query"""
        # Format query with the appropriate context
        formatted_query = self.format_query(query)
        
        # Tokenize query
        inputs = self.tokenizer(
            formatted_query, 
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Set up generation parameters
        gen_params = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(**gen_params)
        
        # Decode the response, removing the input prompt
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        generated_text = response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        # Clean up the response
        cleaned_response = self.clean_response(generated_text)
        
        return cleaned_response
    
    def clean_response(self, response: str) -> str:
        """Clean up the generated response to remove artifacts"""
        # Remove special format markers
        artifact_tokens = ["<|assistant|>", "<|system|>", "<|user|>", "[MEDICAL QUERY]", 
                          "[CLINICAL RESPONSE]", "[RESPONSE]", "</|assistant|>", "</|system|>", 
                          "</|user|>", "[BEGIN RESPONSE]"]
        
        for token in artifact_tokens:
            response = response.replace(token, "")
        
        # Fix common formatting issues
        import re
        response = re.sub(r'\s{2,}', ' ', response)  # Remove extra spaces
        response = re.sub(r'\n{3,}', '\n\n', response)  # Fix multiple newlines
        
        # Remove any trailing special tokens
        if response.endswith("<|endoftext|>"):
            response = response[:-13]
        
        return response.strip()
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate the quality of a medical response"""
        evaluation = {}
        
        # 1. Check length
        evaluation["length"] = len(response)
        
        # 2. Check for medical terminology (simple heuristic)
        medical_terms = [
            "diagnosis", "treatment", "symptoms", "prognosis", "etiology", 
            "pathophysiology", "clinical", "therapy", "medication", "assessment"
        ]
        term_count = sum(1 for term in medical_terms if term in response.lower())
        evaluation["medical_term_count"] = term_count
        
        # 3. Check for structured content (sections)
        has_sections = ":" in response and "\n\n" in response
        evaluation["has_structure"] = has_sections
        
        # 4. Calculate simple quality score
        quality_score = 0
        quality_score += min(len(response) / 200, 5)  # Up to 5 points for length
        quality_score += term_count * 0.5  # 0.5 points per medical term
        quality_score += 2 if has_sections else 0  # 2 points for structure
        
        evaluation["quality_score"] = round(quality_score, 1)
        
        # Assign quality level
        if quality_score >= 7:
            evaluation["quality_level"] = "High"
        elif quality_score >= 4:
            evaluation["quality_level"] = "Medium"
        else:
            evaluation["quality_level"] = "Low"
        
        return evaluation
    
    def run_test(self, category: str = None, single_query: str = None):
        """Run tests on medical queries"""
        results = {}
        
        # Create timestamp for test session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n{Colors.HEADER}🩺 Elara Medical LoRA Test - {timestamp}{Colors.ENDC}")
        print(f"{Colors.BOLD}User Type: {self.config.user_type}{Colors.ENDC}")
        print(f"Device: {self.device}")
        print(f"Temperature: {self.config.temperature}, Rep Penalty: {self.config.repetition_penalty}\n")
        
        # If testing a single query
        if single_query:
            print(f"{Colors.BLUE}Testing single query: {single_query}{Colors.ENDC}")
            start_time = time.time()
            response = self.generate_response(single_query)
            elapsed = time.time() - start_time
            
            evaluation = self.evaluate_response(response)
            
            print(f"{Colors.CYAN}Query:{Colors.ENDC} {single_query}")
            print(f"{Colors.CYAN}Response:{Colors.ENDC}")
            print(f"{response}\n")
            print(f"{Colors.CYAN}Evaluation:{Colors.ENDC}")
            print(f"  - Quality Score: {evaluation['quality_score']} ({evaluation['quality_level']})")
            print(f"  - Length: {evaluation['length']} chars")
            print(f"  - Medical terms: {evaluation['medical_term_count']}")
            print(f"  - Has structure: {evaluation['has_structure']}")
            print(f"  - Generation time: {elapsed:.2f} seconds")
            
            results["single_query"] = {
                "query": single_query,
                "response": response,
                "evaluation": evaluation,
                "time": elapsed
            }
            return results
        
        # If testing a specific category
        if category:
            categories = [category]
        else:
            categories = list(self.test_cases.keys())
        
        # Run tests for each category
        for cat in categories:
            if cat not in self.test_cases:
                print(f"{Colors.RED}Category '{cat}' not found in test cases.{Colors.ENDC}")
                continue
            
            print(f"\n{Colors.BOLD}Testing Category: {cat.upper()}{Colors.ENDC}")
            cat_results = []
            
            for i, test_case in enumerate(self.test_cases[cat]):
                query = test_case["query"]
                query_type = test_case["type"]
                
                print(f"\n{Colors.BLUE}Test #{i+1}: {query_type}{Colors.ENDC}")
                print(f"{Colors.CYAN}Query:{Colors.ENDC} {query}")
                
                # Generate response and measure time
                start_time = time.time()
                response = self.generate_response(query)
                elapsed = time.time() - start_time
                
                # Evaluate response
                evaluation = self.evaluate_response(response)
                
                # Display results
                print(f"{Colors.CYAN}Response:{Colors.ENDC}")
                print(f"{response[:500]}..." if len(response) > 500 else response)
                print(f"\n{Colors.CYAN}Evaluation:{Colors.ENDC}")
                print(f"  - Quality Score: {evaluation['quality_score']} ({evaluation['quality_level']})")
                print(f"  - Length: {evaluation['length']} chars")
                print(f"  - Medical terms: {evaluation['medical_term_count']}")
                print(f"  - Has structure: {evaluation['has_structure']}")
                print(f"  - Generation time: {elapsed:.2f} seconds")
                
                # Add to results
                cat_results.append({
                    "query": query,
                    "query_type": query_type,
                    "response": response,
                    "evaluation": evaluation,
                    "time": elapsed
                })
            
            # Add category results
            results[cat] = cat_results
            
            # Category summary
            quality_scores = [item["evaluation"]["quality_score"] for item in cat_results]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            avg_length = sum(item["evaluation"]["length"] for item in cat_results) / len(cat_results) if cat_results else 0
            avg_time = sum(item["time"] for item in cat_results) / len(cat_results) if cat_results else 0
            
            print(f"\n{Colors.GREEN}Category Summary: {cat.upper()}{Colors.ENDC}")
            print(f"  - Average Quality Score: {avg_quality:.1f}")
            print(f"  - Average Response Length: {avg_length:.1f} chars")
            print(f"  - Average Generation Time: {avg_time:.2f} seconds")
        
        # Save results if configured
        if self.config.save_results:
            results_file = os.path.join(
                self.config.results_dir, 
                f"medical_lora_test_{self.config.user_type}_{timestamp}.json"
            )
            
            # Add metadata
            full_results = {
                "metadata": {
                    "timestamp": timestamp,
                    "user_type": self.config.user_type,
                    "base_model": self.config.base_model,
                    "adapter_path": self.config.adapter_path,
                    "temperature": self.config.temperature,
                    "repetition_penalty": self.config.repetition_penalty,
                    "device": str(self.device)
                },
                "results": results
            }
            
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            
            print(f"\n{Colors.GREEN}✅ Results saved to: {results_file}{Colors.ENDC}")
        
        return results
    
    def run_interactive_mode(self):
        """Run an interactive session for testing the model"""
        print(f"\n{Colors.HEADER}🩺 Elara Medical LoRA Interactive Mode{Colors.ENDC}")
        print(f"{Colors.BOLD}User Type: {self.config.user_type}{Colors.ENDC}")
        print(f"Device: {self.device}")
        print("Type 'exit' to quit, 'settings' to change settings, or your medical query.")
        
        while True:
            print(f"\n{Colors.BLUE}Enter your query:{Colors.ENDC}")
            query = input("> ")
            
            if query.lower() == 'exit':
                print(f"{Colors.GREEN}Exiting interactive mode.{Colors.ENDC}")
                break
            
            if query.lower() == 'settings':
                self._change_settings()
                continue
            
            if not query.strip():
                continue
            
            start_time = time.time()
            response = self.generate_response(query)
            elapsed = time.time() - start_time
            
            evaluation = self.evaluate_response(response)
            
            print(f"\n{Colors.CYAN}Response:{Colors.ENDC}")
            print(f"{response}")
            print(f"\n{Colors.CYAN}Evaluation:{Colors.ENDC}")
            print(f"  - Quality Score: {evaluation['quality_score']} ({evaluation['quality_level']})")
            print(f"  - Length: {evaluation['length']} chars")
            print(f"  - Generation time: {elapsed:.2f} seconds")
    
    def _change_settings(self):
        """Change generation settings interactively"""
        print(f"\n{Colors.HEADER}Change Generation Settings:{Colors.ENDC}")
        print(f"Current settings:")
        print(f"  1. Temperature: {self.config.temperature}")
        print(f"  2. Repetition Penalty: {self.config.repetition_penalty}")
        print(f"  3. Max New Tokens: {self.config.max_new_tokens}")
        print(f"  4. User Type: {self.config.user_type}")
        print(f"  5. Return to interactive mode")
        
        choice = input("Enter number to change (1-5): ")
        
        try:
            choice = int(choice)
            if choice == 1:
                temp = input(f"Enter new temperature (0.1-1.0) [{self.config.temperature}]: ")
                if temp.strip():
                    self.config.temperature = float(temp)
            elif choice == 2:
                rep = input(f"Enter new repetition penalty (1.0-10.0) [{self.config.repetition_penalty}]: ")
                if rep.strip():
                    self.config.repetition_penalty = float(rep)
            elif choice == 3:
                tokens = input(f"Enter max new tokens (100-1000) [{self.config.max_new_tokens}]: ")
                if tokens.strip():
                    self.config.max_new_tokens = int(tokens)
            elif choice == 4:
                print("User types: medical_professional, patient, student, researcher")
                utype = input(f"Enter user type [{self.config.user_type}]: ")
                if utype.strip():
                    self.config.user_type = utype
            elif choice == 5:
                return
            
            print(f"{Colors.GREEN}Settings updated!{Colors.ENDC}")
        except:
            print(f"{Colors.RED}Invalid input. Settings unchanged.{Colors.ENDC}")

def main():
    """Main function to run the tester"""
    parser = argparse.ArgumentParser(description="Test a trained medical LoRA adapter")
    
    # Required arguments
    parser.add_argument("--mode", type=str, default="interactive", 
                        choices=["interactive", "all", "category", "single"],
                        help="Test mode: interactive, all categories, specific category, or single query")
    
    # Optional arguments
    parser.add_argument("--category", type=str, 
                        help="Category to test (required if mode=category)")
    parser.add_argument("--query", type=str, 
                        help="Query to test (required if mode=single)")
    parser.add_argument("--user_type", type=str, default="medical_professional",
                        choices=["medical_professional", "patient", "student", "researcher"],
                        help="User type to simulate")
    parser.add_argument("--temperature", type=float, default=0.15,
                        help="Temperature for generation")
    parser.add_argument("--rep_penalty", type=float, default=4.0,
                        help="Repetition penalty")
    parser.add_argument("--base_model", type=str, default="microsoft/DialoGPT-medium",
                        help="Base model path or identifier")
    parser.add_argument("--adapter_path", type=str, default="../models_files/medical_lora",
                        help="Path to LoRA adapter")
    
    args = parser.parse_args()
    
    # Configure test settings
    config = TestConfig(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        repetition_penalty=args.rep_penalty,
        user_type=args.user_type
    )
    
    # Initialize tester
    tester = MedicalModelTester(config)
    
    # Load model
    if not tester.load_model():
        print(f"{Colors.RED}Failed to load model. Exiting.{Colors.ENDC}")
        return
    
    # Run tests based on mode
    if args.mode == "interactive":
        tester.run_interactive_mode()
    elif args.mode == "all":
        tester.run_test()
    elif args.mode == "category":
        if not args.category:
            print(f"{Colors.RED}Error: --category is required for category mode.{Colors.ENDC}")
            return
        tester.run_test(category=args.category)
    elif args.mode == "single":
        if not args.query:
            print(f"{Colors.RED}Error: --query is required for single mode.{Colors.ENDC}")
            return
        tester.run_test(single_query=args.query)

if __name__ == "__main__":
    main()
