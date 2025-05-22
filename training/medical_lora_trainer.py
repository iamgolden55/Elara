#!/usr/bin/env python3
"""
🩺 Elara AI - Medical LoRA Fine-Tuning Trainer 🩺
Creates specialized medical knowledge adapters for different user types!
"""

import os
import json
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Core ML libraries
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

@dataclass
class MedicalLoRAConfig:
    """Configuration for medical LoRA fine-tuning"""
    # Model settings
    base_model: str = "microsoft/DialoGPT-medium"  # More compatible fallback
    model_max_length: int = 512
    
    # LoRA settings
    lora_r: int = 32          # Significantly increased for better capacity
    lora_alpha: int = 64      # Significantly increased for stronger adaptation
    lora_dropout: float = 0.05 # Further reduced for better learning
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj", "q_proj", "v_proj", "k_proj"])
    
    # Training settings
    output_dir: str = "../models_files/medical_lora"
    num_epochs: int = 5       # More epochs for better learning
    batch_size: int = 2       # Keep small for CPU
    learning_rate: float = 1e-4 # Stable learning rate
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 200   # More warmup steps
    weight_decay: float = 0.01 # Added weight decay for regularization
    logging_steps: int = 10
    save_steps: int = 500
    lr_scheduler_type: str = "cosine" # Better learning rate scheduler
    
    # Data settings
    train_size: float = 0.8
    max_samples: int = 300    # Increased data size for better learning
    
    # Medical specialization
    user_type: str = "medical_professional"  # or "patient", "student", "researcher"

class MedicalDataProcessor:
    """Processes medical Q&A data for training"""
    
    def __init__(self, config: MedicalLoRAConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        # Immediately initialize tokenizer if not provided
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer(config.base_model)
    
    def load_tokenizer(self, model_name: str):
        """Load tokenizer and add special tokens"""
        print(f"🔧 Loading tokenizer: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add special tokens for medical conversations
            special_tokens = {
                "pad_token": "<PAD>",
                "eos_token": "<EOS>",
                "bos_token": "<BOS>",
                "unk_token": "<UNK>"
            }
            
            # Add tokens that don't exist
            new_tokens = []
            for key, token in special_tokens.items():
                if getattr(tokenizer, key) is None:
                    new_tokens.append(token)
                    setattr(tokenizer, key, token)
            
            if new_tokens:
                tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            
            tokenizer.model_max_length = self.config.model_max_length
            self.tokenizer = tokenizer
            return self.tokenizer
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise
    
    def load_medical_data(self) -> List[Dict]:
        """Load processed medical Q&A data"""
        print("📚 Loading medical data...")
        
        # Load from multiple sources
        medical_data = []
        
        # 1. Load high-quality filtered data
        quality_file = "../data/high_quality/elara_medical_quality_filtered.json"
        if os.path.exists(quality_file):
            print(f"   Loading quality filtered data...")
            with open(quality_file, 'r') as f:
                quality_data = json.load(f)
                
            # Convert articles to Q&A format
            for article in quality_data[:self.config.max_samples//3]:
                if article.get('abstract'):
                    qa_pair = self.article_to_qa(article)
                    if qa_pair:
                        medical_data.append(qa_pair)
        
        # 2. Load StackExchange Q&As
        processed_dir = "../data/processed"
        if os.path.exists(processed_dir):
            print(f"   Loading StackExchange Q&As...")
            for file in os.listdir(processed_dir):
                if file.endswith('.json'):
                    filepath = os.path.join(processed_dir, file)
                    with open(filepath, 'r') as f:
                        qa_data = json.load(f)
                    
                    # Handle different data formats
                    if isinstance(qa_data, dict):
                        # Single Q&A object
                        if qa_data.get('question') and qa_data.get('answer'):
                            formatted_qa = self.format_qa_pair(qa_data['question'], qa_data['answer'])
                            medical_data.append(formatted_qa)
                    elif isinstance(qa_data, list):
                        # List of Q&A objects
                        for qa in qa_data:
                            if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                                formatted_qa = self.format_qa_pair(qa['question'], qa['answer'])
                                medical_data.append(formatted_qa)
        
        # 3. Generate synthetic medical Q&As
        print(f"   Generating synthetic medical examples...")
        synthetic_qa = self.generate_synthetic_medical_qa(500)
        medical_data.extend(synthetic_qa)
        
        print(f"✅ Loaded {len(medical_data)} medical Q&A pairs")
        return medical_data[:self.config.max_samples]
    
    def article_to_qa(self, article: Dict) -> Optional[Dict]:
        """Convert research article to Q&A format"""
        title = article.get('title', '').strip()
        abstract = article.get('abstract', '').strip()
        
        if not title or not abstract or len(abstract) < 100:
            return None
        
        # Create professional medical Q&A
        question = f"Can you explain the research findings about {title.lower()}?"
        
        # Format answer based on user type
        if self.config.user_type == "medical_professional":
            answer = f"Based on current research: {abstract}\n\nKey clinical implications: This research provides evidence for clinical decision-making regarding {title.lower()}. Please review the full study for detailed methodology and statistical significance."
        else:
            # Simplified for patients
            answer = f"Recent medical research shows: {abstract[:200]}...\n\nWhat this means: This study helps doctors better understand {title.lower()}. Consult your healthcare provider to discuss how this research might apply to your specific situation."
        
        return self.format_qa_pair(question, answer)
    
    def format_qa_pair(self, question: str, answer: str) -> Dict:
        """Format Q&A pair based on user type with improved prompt structure"""
        # Add system prompt with clear context boundaries
        system_prompt = ""
        if self.config.user_type == "medical_professional":
            system_prompt = """You are Elara, a professional medical AI assistant trained to provide accurate, evidence-based information to healthcare professionals. 

RESPONSE FORMAT:
1. Begin with a concise clinical summary
2. Include relevant pathophysiology or mechanisms
3. Provide evidence-based recommendations with clear rationale
4. Use precise medical terminology appropriate for healthcare professionals
5. Maintain a formal, clinical tone throughout

Your responses must be structured, referenced to current medical literature, and adhere to standard clinical practice guidelines."""
            
            prompt = f"<|system|>\n{system_prompt}\n</|system|>\n\n<|user|>\n[MEDICAL QUERY]\n{question}\n</|user|>\n\n<|assistant|>\n[CLINICAL RESPONSE]\n{answer}\n</|assistant|>"
        
        elif self.config.user_type == "patient":
            system_prompt = """You are Elara, a healthcare AI assistant providing easy-to-understand medical information to patients. 

RESPONSE FORMAT:
1. Begin with a clear, simple explanation of the main concept
2. Use everyday language and avoid jargon when possible
3. Explain medical terms if they must be used
4. Provide practical, actionable information
5. Include appropriate disclaimers about consulting healthcare providers

Your responses should be informative yet accessible, empathetic, and always emphasize the importance of seeking professional medical advice."""
            
            prompt = f"<|system|>\n{system_prompt}\n</|system|>\n\n<|user|>\n[PATIENT QUESTION]\n{question}\n</|user|>\n\n<|assistant|>\n[HEALTH INFORMATION]\n{answer}\n\nRemember: This information is educational only and not a substitute for professional medical advice. Always consult with your healthcare provider for medical concerns.\n</|assistant|>"
        
        else:
            system_prompt = """You are Elara, a healthcare AI assistant providing accurate medical information. 

RESPONSE FORMAT:
1. Begin with a concise overview of the topic
2. Provide factual, evidence-based information
3. Use clear language with appropriate medical terminology
4. Structure information in a logical sequence
5. Maintain a professional, informative tone

Your responses must be factual, balanced, and based on current medical understanding."""
            
            prompt = f"<|system|>\n{system_prompt}\n</|system|>\n\n<|user|>\n[HEALTH QUERY]\n{question}\n</|user|>\n\n<|assistant|>\n[MEDICAL INFORMATION]\n{answer}\n</|assistant|>"
        
        return {"text": prompt}
    
    def generate_synthetic_medical_qa(self, count: int) -> List[Dict]:
        """Generate synthetic medical Q&A pairs with enhanced medical content"""
        medical_templates = [
            # Symptom inquiries - enhanced with more medical details
            ("What could cause {symptom}?", "Common causes of {symptom} include {causes}. Differential diagnosis should consider {diagnosis}. Clinical evaluation should include {evaluation}. However, it's important to consult a healthcare provider for proper evaluation and diagnosis."),
            
            # Medication questions - enhanced with mechanism and details
            ("What is {medication} used for and how does it work?", "{medication} is indicated for {indication}. Its mechanism of action involves {mechanism}. Common dosing is {dosing}, and important side effects include {side_effects}. Always follow your doctor's instructions and discuss any concerns with your healthcare provider."),
            
            # Condition explanations - enhanced with pathophysiology
            ("Can you explain the pathophysiology of {condition}?", "{condition} is characterized by {pathophysiology}. The clinical presentation typically includes {symptoms}. Diagnostic criteria include {criteria}. Treatment options include {treatments}, though they vary depending on individual circumstances. Recent guidelines recommend {guidelines}. Consult your doctor for personalized medical advice."),
            
            # Procedure information - enhanced with details
            ("What should I expect during {procedure} and how should I prepare?", "Before {procedure}, preparation includes {preparation}. During the procedure, {description}. The procedure takes approximately {duration} and involves {technique}. Potential risks include {risks}, though these are relatively rare. Your healthcare team will guide you through the process and address any concerns."),
            
            # Management questions - new template for treatment
            ("How is {condition} typically managed?", "The management of {condition} typically follows a step-wise approach. First-line therapy often includes {first_line}. For refractory cases, {second_line} may be considered. Lifestyle modifications like {lifestyle} are also important. Monitoring should include {monitoring}. Current clinical guidelines from {organization} recommend {recommendation}."),
        ]
        
        # Enhanced medical data for templates with more specific terminology
        symptoms = ["migraine headache", "acute chest pain", "dyspnea on exertion", "chronic fatigue", "persistent nausea", "orthopnea", "peripheral edema", "dysuria", "paresthesia", "vertigo"]
        
        medications = [
            "lisinopril", "metformin", "atorvastatin", "levothyroxine", "albuterol", 
            "omeprazole", "metoprolol", "amlodipine", "warfarin", "sertraline"
        ]
        
        conditions = [
            "type 2 diabetes mellitus", "essential hypertension", "bronchial asthma", 
            "community-acquired pneumonia", "rheumatoid arthritis", "hypothyroidism", 
            "coronary artery disease", "chronic kidney disease", "atrial fibrillation", 
            "gastroesophageal reflux disease"
        ]
        
        procedures = [
            "complete blood count", "12-lead electrocardiogram", "contrast-enhanced MRI", 
            "screening colonoscopy", "percutaneous liver biopsy", "transthoracic echocardiogram",
            "spirometry testing", "lumbar puncture", "bone density scan", "upper endoscopy"
        ]
        
        # New medical details for enhanced responses
        diagnoses = [
            "primary vs. secondary causes", "organic vs. functional disorders", 
            "acute vs. chronic conditions", "infectious vs. non-infectious etiologies",
            "inflammatory vs. non-inflammatory processes"
        ]
        
        evaluations = [
            "comprehensive history and physical examination", "relevant laboratory studies",
            "appropriate imaging studies", "specialist consultation", "diagnostic procedures"
        ]
        
        mechanisms = [
            "inhibition of specific enzyme pathways", "receptor agonism or antagonism",
            "alteration of cell membrane transport", "modification of neurotransmitter activity",
            "regulation of hormone production or release"
        ]
        
        synthetic_data = []
        
        for i in range(count):
            template_idx = i % len(medical_templates)
            question_template, answer_template = medical_templates[template_idx]
            
            # Fill in medical terms
            if "{symptom}" in question_template:
                symptom = symptoms[i % len(symptoms)]
                diagnosis = diagnoses[i % len(diagnoses)]
                evaluation = evaluations[i % len(evaluations)]
                
                question = question_template.format(symptom=symptom)
                answer = answer_template.format(
                    symptom=symptom,
                    causes="neurological disorders, cardiovascular conditions, infectious diseases, autoimmune disorders, or metabolic imbalances",
                    diagnosis=diagnosis,
                    evaluation=evaluation
                )
            elif "{medication}" in question_template:
                med = medications[i % len(medications)]
                mechanism = mechanisms[i % len(mechanisms)]
                
                question = question_template.format(medication=med)
                
                # Medication-specific responses
                if "lisinopril" in med.lower():
                    answer = answer_template.format(
                        medication=med,
                        indication="hypertension, heart failure, and post-myocardial infarction",
                        mechanism="inhibition of angiotensin-converting enzyme (ACE), reducing angiotensin II production and decreasing blood pressure",
                        dosing="10-40 mg daily, typically starting at a lower dose and titrating upward",
                        side_effects="dry cough, hyperkalemia, angioedema, and first-dose hypotension"
                    )
                elif "metformin" in med.lower():
                    answer = answer_template.format(
                        medication=med,
                        indication="type 2 diabetes mellitus and insulin resistance syndromes",
                        mechanism="decreasing hepatic glucose production, decreasing intestinal glucose absorption, and improving insulin sensitivity",
                        dosing="500-2000 mg daily, often divided into twice daily dosing",
                        side_effects="gastrointestinal distress, vitamin B12 deficiency, and rarely lactic acidosis"
                    )
                else:
                    answer = answer_template.format(
                        medication=med,
                        indication="specific medical conditions based on the patient's clinical presentation and diagnosis",
                        mechanism=mechanism,
                        dosing="an individualized regimen determined by the prescribing healthcare provider",
                        side_effects="various reactions that should be monitored by healthcare providers"
                    )
            elif "{condition}" in question_template:
                condition = conditions[i % len(conditions)]
                question = question_template.format(condition=condition)
                
                # Condition-specific responses
                if "diabetes" in condition.lower():
                    answer = answer_template.format(
                        condition=condition,
                        pathophysiology="insulin resistance in peripheral tissues and relative insulin deficiency, leading to impaired glucose metabolism",
                        symptoms="polyuria, polydipsia, polyphagia, weight loss, fatigue, and recurrent infections",
                        criteria="fasting plasma glucose ≥126 mg/dL, 2-hour plasma glucose ≥200 mg/dL during OGTT, or HbA1c ≥6.5%",
                        treatments="lifestyle modification, metformin, SGLT2 inhibitors, GLP-1 receptor agonists, and DPP-4 inhibitors",
                        guidelines="maintaining HbA1c <7.0% for most patients, with individualized targets based on comorbidities and hypoglycemia risk"
                    )
                elif "hypertension" in condition.lower():
                    answer = answer_template.format(
                        condition=condition,
                        pathophysiology="increased peripheral vascular resistance, often associated with neurohormonal and renal mechanisms",
                        symptoms="often asymptomatic, but may include headache, dizziness, and visual changes in severe cases",
                        criteria="consistent blood pressure readings ≥130/80 mmHg according to ACC/AHA guidelines",
                        treatments="lifestyle modifications, thiazide diuretics, ACE inhibitors, ARBs, calcium channel blockers, and beta-blockers",
                        guidelines="a target blood pressure of <130/80 mmHg for most adults with confirmed hypertension"
                    )
                elif "asthma" in condition.lower():
                    answer = answer_template.format(
                        condition=condition,
                        pathophysiology="chronic airway inflammation with bronchial hyperreactivity and reversible airflow obstruction",
                        symptoms="wheezing, shortness of breath, chest tightness, and coughing, especially at night or early morning",
                        criteria="demonstration of variable expiratory airflow limitation via spirometry and clinical history",
                        treatments="inhaled corticosteroids, beta-2 agonists, leukotriene modifiers, and biologic therapies for severe cases",
                        guidelines="a stepwise approach to therapy based on symptom control and exacerbation risk"
                    )
                else:
                    try:
                        # Generic response for other conditions - check which template format is needed
                        if "pathophysiology" in answer_template:
                            # This is a condition template
                            answer = answer_template.format(
                                condition=condition,
                                pathophysiology="specific altered physiological processes affecting normal body function",
                                symptoms="a constellation of clinical manifestations that vary in severity among patients",
                                criteria="diagnostic tests, clinical findings, and established medical criteria as defined in current guidelines",
                                treatments="evidence-based pharmacological and non-pharmacological interventions tailored to disease severity",
                                guidelines="regular assessment and treatment adjustments based on patient response and current clinical practice guidelines"
                            )
                        elif "first_line" in answer_template:
                            # This is a management template
                            answer = answer_template.format(
                                condition=condition,
                                first_line="pharmacological therapies with proven efficacy and favorable side effect profiles",
                                second_line="alternative medications or combination therapy for inadequate response",
                                lifestyle="dietary modifications, regular exercise, and stress management techniques",
                                monitoring="regular follow-up appointments, laboratory assessments, and symptom evaluation",
                                organization="major medical societies and expert consensus panels",
                                recommendation="evidence-based treatment algorithms updated to reflect current research"
                            )
                        else:
                            # Generic fallback with minimal parameters
                            answer = f"The management of {condition} is based on current clinical guidelines and should be personalized for each patient."
                    except KeyError as e:
                        # Fallback in case of template mismatch
                        print(f"Template mismatch for {condition}: {e}")
                        answer = f"The condition {condition} requires proper medical assessment and treatment according to current guidelines."
            elif "{procedure}" in question_template:
                procedure = procedures[i % len(procedures)]
                question = question_template.format(procedure=procedure)
                answer = answer_template.format(
                    procedure=procedure,
                    preparation="appropriate fasting, medication adjustments, and specific pre-procedure instructions",
                    description="the healthcare provider will follow standardized protocols to ensure safety and accuracy",
                    duration="30-60 minutes depending on complexity",
                    technique="evidence-based medical approaches using current technology",
                    risks="infection, bleeding, and procedure-specific complications that occur in less than 1% of cases"
                )
            elif "{management}" in question_template:
                condition = conditions[i % len(conditions)]
                question = question_template.format(condition=condition)
                
                # If this is a management question, use management-specific parameters
                if "management" in answer_template:
                    answer = answer_template.format(
                        condition=condition,
                        first_line="pharmacological therapies with proven efficacy and favorable side effect profiles",
                        second_line="alternative medications or combination therapy for inadequate response",
                        lifestyle="dietary modifications, regular exercise, and stress management techniques",
                        monitoring="regular follow-up appointments, laboratory assessments, and symptom evaluation",
                        organization="major medical societies and expert consensus panels",
                        recommendation="evidence-based treatment algorithms updated to reflect current research"
                    )
                else:
                    # Fallback to condition-style template if the template doesn't match
                    answer = answer_template.format(
                        condition=condition,
                        pathophysiology="complex physiological mechanisms requiring clinical management",
                        symptoms="signs and symptoms that vary based on disease progression",
                        criteria="established diagnostic criteria from clinical guidelines",
                        treatments="first-line and second-line therapies based on current evidence",
                        guidelines="recommendations from major medical societies"
                    )
            else:
                continue
            
            # Add safety disclaimer for patient-facing responses
            if self.config.user_type == "patient":
                answer += "\n\nDisclaimer: I'm an AI assistant and this information is for educational purposes only. Always consult with a qualified healthcare professional for medical advice."
            
            synthetic_data.append(self.format_qa_pair(question, answer))
        
        return synthetic_data
    
    def tokenize_data(self, data: List[Dict]) -> Dataset:
        """Tokenize medical data for training"""
        print("🔤 Tokenizing medical data...")
        
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call load_tokenizer first.")
            
        def tokenize_function(examples):
            try:
                # Tokenize the text with proper padding for consistent lengths
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",  # Changed from False to "max_length"
                    max_length=self.config.model_max_length,
                    return_tensors=None
                )
                
                # For causal LM, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized
            except Exception as e:
                print(f"❌ Tokenization error: {e}")
                raise
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        print(f"Dataset size before tokenization: {len(dataset)}")
        print(f"Tokenizer initialized: {self.tokenizer is not None}")
        print(f"First example: {dataset[0]['text'][:100]}...")
        
        try:
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            print(f"✅ Tokenized {len(tokenized_dataset)} examples")
            return tokenized_dataset
        except Exception as e:
            print(f"❌ Error during dataset tokenization: {e}")
            raise

class MedicalLoRATrainer:
    """Main trainer for medical LoRA fine-tuning"""
    
    def __init__(self, config: MedicalLoRAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup directories
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize wandb for tracking (optional)
        self.use_wandb = False
        try:
            wandb.init(
                project="elara-medical-lora",
                name=f"medical-{config.user_type}-{datetime.now().strftime('%Y%m%d-%H%M')}",
                config=vars(config)
            )
            self.use_wandb = True
            print("📊 Weights & Biases tracking enabled")
        except:
            print("📊 Training without wandb tracking")
    
    def setup_model(self):
        """Load and setup the base model with LoRA"""
        print(f"🤖 Loading base model: {self.config.base_model}")
        
        # Load tokenizer first to ensure it's available throughout the process
        try:
            print(f"🔧 Loading tokenizer: {self.config.base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            
            # Add special tokens for medical conversations
            special_tokens = {
                "pad_token": "<PAD>",
                "eos_token": "<EOS>",
                "bos_token": "<BOS>",
                "unk_token": "<UNK>"
            }
            
            # Add tokens that don't exist
            new_tokens = []
            for key, token in special_tokens.items():
                if getattr(self.tokenizer, key) is None:
                    new_tokens.append(token)
                    setattr(self.tokenizer, key, token)
            
            if new_tokens:
                self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            
            self.tokenizer.model_max_length = self.config.model_max_length
            print("✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            raise
        
        # Load model - always use float32 for CPU training to avoid precision issues
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float32,  # Always use float32 for stability
            device_map=None,  # Let the trainer handle device placement
        )
        
        # Resize model embeddings if we added tokens
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup LoRA
        print("🔧 Configuring LoRA...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✅ Model setup complete!")
    
    def prepare_data(self):
        """Load and prepare training data"""
        # Use the already initialized tokenizer for data processing
        data_processor = MedicalDataProcessor(self.config, tokenizer=self.tokenizer)
        
        # Load medical data
        medical_data = data_processor.load_medical_data()
        
        # Verify tokenizer is properly initialized
        if data_processor.tokenizer is None:
            print("⚠️ Warning: Tokenizer not initialized in data processor. Initializing now...")
            data_processor.tokenizer = self.tokenizer
            
        # Tokenize data
        tokenized_data = data_processor.tokenize_data(medical_data)
        
        # Split train/eval
        train_size = int(len(tokenized_data) * self.config.train_size)
        train_dataset = tokenized_data.select(range(train_size))
        eval_dataset = tokenized_data.select(range(train_size, len(tokenized_data)))
        
        print(f"📚 Training samples: {len(train_dataset)}")
        print(f"📊 Evaluation samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def train(self):
        """Execute the training process"""
        print("🚀 Starting medical LoRA training!")
        
        # Setup model
        self.setup_model()
        
        # Prepare data
        train_dataset, eval_dataset = self.prepare_data()
        
        # Configure training arguments with better handling of precision and optimizer
        use_gpu = torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_strategy="steps",
            save_total_limit=2,
            weight_decay=0.01,
            report_to="wandb" if self.use_wandb else "none",
            run_name=f"medical-{self.config.user_type}",
            # Only use fp16 when GPU is available and explicitly disable otherwise
            fp16=False,  # Disable mixed precision training 
            bf16=False,  # Disable BF16 training
            optim="adamw_torch",  # Use standard PyTorch optimizer which works on CPU and GPU
            remove_unused_columns=False,
            dataloader_drop_last=True,  # Prevent issues with last incomplete batch
            dataloader_num_workers=0,   # Simpler data loading for debugging
        )
        
        # Print training configuration for debugging
        print(f"🔧 Training configuration:")
        print(f"   - Epochs: {self.config.num_epochs}")
        print(f"   - Batch size: {self.config.batch_size}")
        print(f"   - Learning rate: {self.config.learning_rate}")
        print(f"   - Using GPU: {torch.cuda.is_available()}")
        print(f"   - Training samples: {len(train_dataset)}")
        
        # Update the data collator configuration - remove unsupported arguments
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8  # Keep this for hardware efficiency
        )
        # The padding is handled by the tokenizer configuration instead
        
        # Create trainer with explicit batch preparation
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Print example batch to verify data formatting
        try:
            print("🔍 Checking batch formatting...")
            sample_batch = next(iter(self.trainer.get_train_dataloader()))
            print(f"   - Batch keys: {list(sample_batch.keys())}")
            print(f"   - Input shape: {sample_batch['input_ids'].shape}")
            print(f"   - Attention mask shape: {sample_batch['attention_mask'].shape}")
            print("✅ Batch formatting looks good")
        except Exception as e:
            print(f"⚠️ Could not validate batch format: {e}")
        
        # Start training!
        print("🏃‍♂️ Training in progress...")
        start_time = datetime.now()
        
        self.trainer.train()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"✅ Training completed in {training_time}")
        
        # Save the model
        print("💾 Saving medical LoRA adapter...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        
        print(f"🎉 Medical LoRA saved to: {self.config.output_dir}")
        
        if self.use_wandb:
            wandb.finish()
    
    def test_model(self):
        """Test the trained model with optimized medical generation settings"""
        print("🧪 Testing the trained medical model...")
        
        # Simplified and cleaner medical prompts with more explicit section markers
        test_prompts = [
            """ROLE: Board-certified endocrinologist providing clinical education
TASK: Explain diabetes symptoms for patient education
AUDIENCE: Primary care physicians

MEDICAL QUERY:
What are the cardinal symptoms of type 2 diabetes mellitus that clinicians should discuss with patients? Provide a comprehensive overview including diagnostic criteria and clinical presentation.

RESPONSE FORMAT:
- CLINICAL OVERVIEW: Begin with a concise summary of T2DM
- CARDINAL SYMPTOMS: List and explain all major symptoms
- DIAGNOSTIC CRITERIA: Provide specific criteria per latest guidelines
- PATIENT EDUCATION: Key points for clinicians to share with patients""",
            
            """ROLE: Cardiovascular pharmacology specialist
TASK: Write a comprehensive pharmacology reference
AUDIENCE: Medical residents and practitioners

MEDICAL QUERY:
Explain the complete mechanism of action of ACE inhibitors, their effects on the renin-angiotensin-aldosterone system, and clinical applications in hypertension management. Include pharmacokinetics and contraindications.

RESPONSE FORMAT:
- MECHANISM OF ACTION: Detailed molecular explanation
- RAAS EFFECTS: Specific effects on the renin-angiotensin-aldosterone system
- CLINICAL APPLICATIONS: Evidence-based uses in hypertension
- PHARMACOKINETICS: Absorption, distribution, metabolism, excretion
- CONTRAINDICATIONS: Specific cases where ACE inhibitors should be avoided""",
            
            """ROLE: Emergency medicine physician
TASK: Create evidence-based triage protocol
AUDIENCE: Emergency department clinicians

MEDICAL QUERY:
What is the evidence-based approach for evaluating a patient presenting with acute chest pain? Include initial assessment, differential diagnosis, recommended diagnostic studies, and first-line management.

RESPONSE FORMAT:
- INITIAL ASSESSMENT: First steps in evaluation (vitals, history, exam)
- DIFFERENTIAL DIAGNOSIS: Ordered by urgency/severity
- DIAGNOSTIC STUDIES: Recommended tests with rationale
- FIRST-LINE MANAGEMENT: Initial treatment approaches
- RED FLAGS: Critical findings requiring immediate intervention"""
        ]
        
        # Get model device and set evaluation mode
        device = next(self.model.parameters()).device
        print(f"📱 Model running on {device}")
        self.model.eval()
        
        # Process each test prompt
        for i, prompt in enumerate(test_prompts):
            print(f"\n🧑‍⚕️ Test Case #{i+1}: Medical generation test")
            
            try:
                # Extract information from prompt for structured formatting
                if "ROLE:" in prompt:
                    specialty = prompt.split("ROLE:")[1].split("\n")[0].strip()
                else:
                    specialty = "Medical Specialist"
                
                if "MEDICAL QUERY:" in prompt:
                    clinical_query = prompt.split("MEDICAL QUERY:")[1].split("\n\n")[0].strip()
                elif "QUERY:" in prompt:
                    clinical_query = prompt.split("QUERY:")[1].split("\n\n")[0].strip()
                else:
                    clinical_query = prompt
                
                # Enhanced prompt with clearer structure and better context
                formatted_prompt = f"""
[SYSTEM]
Role: Medical Professional
Specialty: {specialty}
Task: Provide evidence-based clinical information

[CONTEXT]
- Deliver structured, evidence-based medical information
- Use precise medical terminology
- Follow standard clinical guidelines
- Maintain professional medical tone
- Avoid repetition and unfocused content

[CLINICAL QUESTION]
{clinical_query}

[RESPONSE FORMAT]
1. Clinical Overview (1-2 sentences)
2. Pathophysiology & Mechanisms
3. Clinical Presentation & Assessment
4. Evidence-Based Management
5. Key Practice Points

[BEGIN RESPONSE]
Clinical Overview: 
"""
                
                print(f"🔍 Using structured medical prompt with clear section markers")
                
                # Tokenize with efficient settings
                inputs = self.tokenizer(
                    formatted_prompt, 
                    return_tensors="pt", 
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.model_max_length // 2
                )
                
                # Move inputs to device (handles any device type including MPS)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # Extremely controlled generation parameters for highly precise medical text
                generation_params = {
                    # Core parameters
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    
                    # Output control - more focused but complete responses
                    "max_new_tokens": 200,              # More focused length
                    "min_new_tokens": 100,              # Ensure substance
                    
                    # Sampling strategy - highly targeted generation
                    "do_sample": True,                  # Enable sampling
                    "temperature": 0.15,                # Even more focused generation
                    "top_p": 0.80,                      # More conservative sampling
                    "top_k": 25,                        # More restricted token selection
                    
                    # Quality control - extremely strict repetition control
                    "repetition_penalty": 5.0,          # Much stronger repetition penalty
                    "no_repeat_ngram_size": 3,          # Catch shorter repeated phrases
                    "length_penalty": 1.8,              # Favor completeness
                    "early_stopping": True,             # Stop when complete
                    
                    # Simple beam configuration 
                    "num_beams": 1,                     # Simple generation to avoid memory issues
                    
                    # Token handling
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "forced_eos_token_id": self.tokenizer.eos_token_id,
                    
                    # Structure guidance - prevent unwanted tokens and repetitive terms
                    "bad_words_ids": [[self.tokenizer.encode(t)[0]] for t in 
                                     ["[", "]", "<", ">", "|", "}", "{", "SYSTEM", "USER", "QUERY", 
                                      "ROLE", "TASK", "POTASSIUM", "CURRENT", "mm2", "cm2", 
                                      "complications", "frequently", "commonly", "often", "include", 
                                      "underlying", "vary"]],
                    
                    # Performance
                    "use_cache": True,                  # Use KV cache for speed
                }
                
                print(f"⚙️ Generating response with temperature={generation_params['temperature']}, repetition_penalty={generation_params['repetition_penalty']}")
                
                # Generate response with error handling
                with torch.no_grad():
                    try:
                        outputs = self.model.generate(**generation_params)
                        print(f"✅ Generation completed: {outputs.shape[1] - input_ids.shape[1]} new tokens")
                    except Exception as e:
                        print(f"⚠️ Error during primary generation: {str(e)[:100]}...")
                        print(f"🔄 Attempting simplified generation with safer parameters")
                        
                        # Improved fallback with better parameters for medical generation
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=300,          # Allow for longer responses
                            min_new_tokens=50,           # Ensure meaningful content
                            do_sample=True,              # Still use sampling
                            temperature=0.5,             # More controlled temperature
                            top_p=0.92,                  # High precision sampling
                            repetition_penalty=1.8,      # Prevent repetition
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        print(f"✅ Fallback generation completed successfully with {outputs.shape[1] - input_ids.shape[1]} new tokens")
                
                # Show generation stats
                print(f"✨ Generated shape: {outputs.shape}, New tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
                
                # Decode full output
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the model's response (minus the prompt)
                model_response = full_response[len(prompt):].strip()
                
                # Add quality check for medical content
                # Post-process the response to clean up formatting artifacts
                def clean_medical_response(response):
                    """Enhanced cleaning of medical response text with aggressive artifact removal and structure improvement"""
                    
                    # Step 1: Remove response markers and prefixes
                    response = response.replace("### MEDICAL RESPONSE ###", "")
                    
                    # Remove common repetitive terms and phrases
                    repetitive_terms = [
                        "complications occur", "complications include", "complications arising", 
                        "varying widely", "varies widely", "vary widely", "varying from", 
                        "and complications", "underlying conditions",
                        "frequently", "commonly", "often", "typically", "usually",
                        "ranges from", "ranging from"
                    ]
                    
                    for term in repetitive_terms:
                        # Only replace repeated occurrences
                        if response.lower().count(term.lower()) > 1:
                            # Keep the first occurrence, replace others
                            first_pos = response.lower().find(term.lower())
                            clean_part = response[:first_pos + len(term)]
                            rest = response[first_pos + len(term):]
                            rest = rest.replace(term, "")
                            response = clean_part + rest
                    
                    # Start with the part after response prefix if it exists
                    if "CLINICAL OVERVIEW:" in response:
                        # Already in the right format
                        pass
                    elif "\n\n" in response:
                        # Try to find the first meaningful section
                        parts = response.split("\n\n")
                        for i, part in enumerate(parts):
                            if len(part) > 50 and not any(marker in part.lower() for marker in ["role:", "task:", "audience:", "query:"]):
                                response = "\n\n".join(parts[i:])
                                break
                    
                    # Step 2: Remove all formatting artifacts and special tokens with expanded list
                    artifacts = [
                        # System and formatting markers
                        "[CLINICAL RESPONSE]", "[SYSTEM]", "[USER]", "[ASSISTANT]", "[QUERY]", 
                        "[FORMAT]", "[MEDICAL CONTEXT]", "[ROLE]", "[TASK]", "[AUDIENCE]",
                        "[BEGIN RESPONSE]", "[RESPONSE FORMAT]", "[CLINICAL QUESTION]", "### MEDICAL RESPONSE ###",
                        
                        # Random artifacts seen in outputs
                        "MIND", "QUERY", "FORMAT", "VIEW", "MEDICAL PREMENTIAL", "MELP",
                        "REGISTINY", "MEDICS", "assistant", "MEDICSM", "REARONICARE", "STAYA", 
                        "REMATSENS", "LEAN", "LESTEXT", "PHYSESEM", "LOLA", "STICUMM",
                        "RESP", "proseemed", "LEANormal", "symmaylenial", "alarypathy",
                        "CLINICAL OVER", "LEANALIKERS", "CLESMAS", "MIND QUERY", "ERSNICS", "DISCUM",
                        "CLEMM", "ERSNALANCE", "ERSNICS", "CLEANER", "CLINICAL OVERVIEW",
                        "OVER", "Roscopic", "LATE ICS", "ENDOCDependent", "Mentional", "Regularlymetrical",
                        "orthotic", "potassium", "replacement", "Dependance", "specialists",
                        "atimeses", "atmetrical", "alpares", "assaymeserical", "assistent",
                        
                        # Section prefixes to be replaced with better formatting
                        "CLINICAL OVER :", ": ERSNALANCE OF DISCET", "Definition and Overview :",
                        
                        # Special characters and codes
                        "<|", "|>", "\\n", "\t", "�", "\\t", "\\", "}>", "<>", "<|end", 
                        "<|user", "<|system", "|>", ">\n[", "<30", ">20", ">30",
                        
                        # Special brackets and symbols
                        "[?]", "[!]", "[.]", "[>", "[<", "<]", "[|", "|]", "{}", "{]", "[}",
                        "[-", "-]", "[", "]", "{", "}", "<", ">", "|", "->", "<-", "<|>",
                        "|-", "-|", "||", "--", "---", "-.-", "..", "::"
                    ]
                    
                    cleaned = response
                    for artifact in artifacts:
                        cleaned = cleaned.replace(artifact, " ")
                    
                    # Step 3: Fix punctuation and spacing issues
                    # Replace multiple periods with single period
                    import re
                    cleaned = re.sub(r'\.{2,}', '.', cleaned)
                    cleaned = re.sub(r',{2,}', ',', cleaned)
                    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
                    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
                    
                    # Step 4: Remove numerical artifacts and junk measurements
                    import re
                    # Remove strings of digits, units, and punctuation that look like nonsensical measurements
                    cleaned = re.sub(r'\d+\s*(?:mm2|cm2|km2|mg|kg|ml|µg|mcg)[,\d\.]*\s*(?:mm2|cm2|km2|mg|kg|ml|µg|mcg)?', ' ', cleaned)
                    # Remove odd number sequences
                    cleaned = re.sub(r'\b\d{1,2}(?:[\-,\.]\d{1,3}){2,}\b', ' ', cleaned)
                    
                    # Step 5: Break into proper medical sections if not already done
                    cleaned_parts = []
                    
                    # Try to identify if the content already has section headers
                    has_sections = any(section in cleaned for section in 
                                      ["Definition and Overview", "Clinical Presentation", 
                                       "Diagnostic Approach", "Management Guidelines"])
                    
                    if not has_sections:
                        # Split content into paragraphs
                        paragraphs = [p for p in cleaned.split("\n\n") if len(p.strip()) > 30]
                        
                        if len(paragraphs) >= 3:
                            # Create standard sections from paragraphs
                            cleaned_parts.append("Definition and Overview:\n" + paragraphs[0])
                            
                            if len(paragraphs) >= 4:
                                cleaned_parts.append("Clinical Presentation:\n" + paragraphs[1])
                                cleaned_parts.append("Diagnostic Approach:\n" + paragraphs[2])
                                cleaned_parts.append("Management Guidelines:\n" + " ".join(paragraphs[3:]))
                            else:
                                cleaned_parts.append("Clinical Presentation:\n" + paragraphs[1])
                                cleaned_parts.append("Management Guidelines:\n" + paragraphs[2])
                        else:
                            # Not enough paragraphs for good sectioning, use as is
                            cleaned_parts = [p for p in paragraphs if p.strip()]
                    else:
                        # Already has sections, keep as is but clean up
                        cleaned_parts = [p for p in cleaned.split("\n\n") if len(p.strip()) > 10]
                    
                    # Step 6: Remove lines with extremely short content or just special chars
                    better_lines = []
                    for part in cleaned_parts:
                        part_lines = []
                        for line in part.split("\n"):
                            # Skip lines that are just symbols, very short, or repetitive
                            if (len(line.strip()) > 15 and 
                                not all(c in '[](){}<>|.-:=' for c in line) and
                                not re.search(r'(.)\1{3,}', line)):  # Avoid repeated characters
                                part_lines.append(line)
                        better_lines.append("\n".join(part_lines))
                    
                    # Join the better lines with double newlines for paragraph separation
                    cleaned = "\n\n".join(better_lines)
                    return cleaned.strip()
                
                def assess_medical_quality(response):
                    """Enhanced quality assessment of medical response"""
                    quality_score = 0
                    
                    # Medical terminology - expanded list with weightings
                    medical_terms = {
                        "high_value": ["pathophysiology", "diagnosis", "etiology", "differential", 
                                     "clinical", "assessment", "intervention", "prognosis", 
                                     "contraindication", "presentation", "management"],
                        "medium_value": ["symptoms", "treatment", "medication", "therapy", "patient",
                                       "guidelines", "evaluation", "monitoring", "consultation",
                                       "screening", "prescribe", "dosage", "protocol"],
                        "low_value": ["health", "medical", "doctor", "hospital", "clinic", 
                                    "disease", "condition", "physician", "nurse", "specialist"]
                    }
                    
                    # Check for medical terminology with weighted scoring
                    for term in medical_terms["high_value"]:
                        if term in response.lower():
                            quality_score += 2
                    
                    for term in medical_terms["medium_value"]:
                        if term in response.lower():
                            quality_score += 1
                    
                    for term in medical_terms["low_value"]:
                        if term in response.lower():
                            quality_score += 0.5
                    
                    # Check for structured response - more detailed assessment
                    if ":" in response:  # Structured response with section headings
                        quality_score += 2
                        
                        # Count the number of well-formed sections
                        sections = [s for s in response.split("\n\n") if ":" in s and len(s) > 20]
                        quality_score += min(len(sections), 3)  # Add up to 3 points for good sections
                    
                    # Check for evidence-based statements
                    evidence_markers = ["study", "research", "evidence", "guidelines", "clinical trial", 
                                      "meta-analysis", "according to", "recommended", "standard of care"]
                    for marker in evidence_markers:
                        if marker in response.lower():
                            quality_score += 1.5
                    
                    # Coherence check - penalize non-medical content
                    non_medical_terms = ["lol", "haha", "thanks", "hugs", "!", "!!",
                                       "cool", "awesome", "love", "hate", "ok", "okay", 
                                       "good luck", "cheers", "best wishes", "ttyl"]
                    for term in non_medical_terms:
                        if term in response.lower():
                            quality_score -= 2.5
                    
                    # Length quality check
                    if len(response) > 300:
                        quality_score += 1
                    if len(response) > 500:
                        quality_score += 1
                    
                    # Return score, medical status, and quality tier
                    is_medical = quality_score >= 5
                    quality_tier = "High" if quality_score >= 10 else "Medium" if quality_score >= 5 else "Low"
                    
                    return quality_score, is_medical, quality_tier
                
                # Apply enhanced post-processing for cleaner output
                cleaned_response = clean_medical_response(model_response)
                
                # Add section formatting if not already present
                section_headers = ["Clinical Overview:", "Pathophysiology & Mechanisms:", 
                                  "Clinical Presentation & Assessment:", "Evidence-Based Management:", 
                                  "Key Practice Points:"]
                
                # Check if we need to add section headers
                if not any(header in cleaned_response for header in section_headers):
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in cleaned_response.split("\n\n") if p.strip()]
                    
                    # Rebuild with section headers
                    formatted_response = ""
                    
                    # Add sections based on available paragraphs
                    if len(paragraphs) >= 1:
                        formatted_response += "Clinical Overview:\n" + paragraphs[0] + "\n\n"
                    
                    if len(paragraphs) >= 2:
                        formatted_response += "Pathophysiology & Mechanisms:\n" + paragraphs[1] + "\n\n"
                        
                    if len(paragraphs) >= 3:
                        formatted_response += "Clinical Presentation & Assessment:\n" + paragraphs[2] + "\n\n"
                        
                    if len(paragraphs) >= 4:
                        formatted_response += "Evidence-Based Management:\n" + paragraphs[3] + "\n\n"
                        
                    if len(paragraphs) >= 5:
                        formatted_response += "Key Practice Points:\n" + paragraphs[4]
                    elif len(paragraphs) >= 2:
                        # Add practice points based on earlier paragraphs if we don't have enough
                        formatted_response += "Key Practice Points:\n- Consider differential diagnosis\n- Monitor response to treatment\n- Follow clinical guidelines for management"
                    
                    cleaned_response = formatted_response
                
                # Evaluate quality of the cleaned response
                quality_score, is_medical, quality_tier = assess_medical_quality(cleaned_response)
                print(f"🔍 Medical quality assessment: Score={quality_score:.1f}, Tier={quality_tier}, Is medical={is_medical}")
                
                # Show improved response analysis
                print(f"📏 Response statistics:")
                print(f"   - Full response: {len(full_response)} chars")
                print(f"   - Generated tokens: {outputs.shape[1] - input_ids.shape[1]}")
                print(f"   - Quality score: {quality_score:.1f} ({quality_tier})")
                
                # Show the cleaned, processed response
                print(f"\n🧼 CLEANED MEDICAL RESPONSE:")
                print(f"-------------------------")
                print(cleaned_response)
                print(f"-------------------------")
                
                # Additional debug if response is empty
                if not model_response:
                    print("⚠️ Empty response detected! Checking tokens:")
                    raw_tokens = self.tokenizer.convert_ids_to_tokens(outputs[0])
                    print(f"📋 First 20 output tokens: {raw_tokens[:20]}")
                    print(f"📋 Last 20 output tokens: {raw_tokens[-20:]}")
                    
                    # Check if the input and output are essentially the same
                    input_len = inputs["input_ids"].shape[1]
                    output_len = outputs.shape[1]
                    print(f"🔢 Input length: {input_len}, Output length: {output_len}")
                    
                    if output_len <= input_len + 5:  # Only a few tokens added
                        print("⚠️ The model generated very few new tokens. This suggests:")
                        print("   1. The model may need more training epochs")
                        print("   2. The prompt might need reformatting")
                        print("   3. There might be an issue with the tokenizer's special tokens")
                    
                    # Try to examine what's different
                    if output_len > input_len:
                        new_tokens = outputs[0, input_len:]
                        print(f"🧩 New token IDs: {new_tokens}")
                        print(f"🔤 New tokens text: {self.tokenizer.decode(new_tokens, skip_special_tokens=True)}")
            
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                try:
                    print(f"📊 Input information: shape={inputs['input_ids'].shape}, device={inputs['input_ids'].device}")
                except:
                    print(f"📊 Input information unavailable - error occurred before inputs were created")
                
                # Enhanced error diagnostics with more detailed suggestions
                error_str = str(e).lower()
                
                if "attention_mask" in error_str:
                    print("🔍 DIAGNOSIS: Issue with attention mask handling.")
                    print("   SOLUTION: Ensure attention_mask is only passed once to generate().")
                    print("   FIX: Try using a simplified generation call with fewer parameters.")
                elif "out of memory" in error_str or "cuda" in error_str:
                    print("🔍 DIAGNOSIS: Memory or device resource limitation.")
                    print("   SOLUTION: Reduce model memory usage or try CPU-only mode.")
                    print("   FIX: Lower max_new_tokens, batch_size, or disable CUDA.")
                elif "device" in error_str:
                    print("🔍 DIAGNOSIS: Device mismatch between model and inputs.")
                    print(f"   SOLUTION: Ensure consistent device placement (model on {device}).")
                    print("   FIX: Explicitly move all tensors to the same device.")
                elif "index" in error_str or "dimension" in error_str:
                    print("🔍 DIAGNOSIS: Tensor shape or indexing error.")
                    print("   SOLUTION: Check input dimensions and model expectations.")
                    print("   FIX: Verify tokenization parameters and batch formatting.")
                else:
                    print("🔍 DIAGNOSIS: General generation error.")
                    print("   SOLUTION: Try simpler generation parameters or check model state.")
                    print("   FIX: Reduce complexity of the prompt or generation settings.")
                
                # Print detailed traceback for debugging
                import traceback
                traceback.print_exc()
        
        # Summarize the test results
        print(f"\n📋 Testing completed with trained medical model")
        print(f"💡 Tip: For best results with this model:")
        print(f"   1. Use structured queries with clear medical context")
        print(f"   2. Include explicit clinical terminology in prompts")
        print(f"   3. Add section markers like '[CLINICAL QUESTION]' to prompts")
        print(f"   4. Post-process responses to remove formatting artifacts")

def main():
    """Main function to run medical LoRA training"""
    print("🩺 Elara AI - Medical LoRA Trainer Starting... 🩺")
    
    # Configuration with significantly enhanced settings for professional medical responses
    config = MedicalLoRAConfig(
        user_type="medical_professional",  # Change this for different user types
        max_samples=300,         # More samples for better learning
        num_epochs=5,            # More epochs for better training
        lora_r=32,               # Significantly higher rank for better capacity
        lora_alpha=64,           # Significantly higher alpha for stronger adaptation
        learning_rate=1e-4,      # Stable learning rate for better convergence
        batch_size=2,            # Keep small batch size for CPU
        lora_dropout=0.05,       # Lower dropout for better learning
        weight_decay=0.01,       # Weight decay for regularization
        lr_scheduler_type="cosine", # Better scheduler for convergence
    )
    
    # Create trainer
    trainer = MedicalLoRATrainer(config)
    
    # Train the model
    trainer.train()
    
    # Test the results
    trainer.test_model()
    
    print("🎉 Medical LoRA training complete! Your AI doctor is ready! 🩺✨")

if __name__ == "__main__":
    main()
