#!/usr/bin/env python3
"""
🔥 ELARA AI MEDICAL PROFESSIONAL LORA TRAINER 🔥
Fine-tune Mistral 7B for medical professionals using LoRA
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path

# Import required libraries
try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    import bitsandbytes as bnb
    from accelerate import Accelerator
    print("✅ All required libraries imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install transformers peft datasets bitsandbytes accelerate")
    exit(1)

# Configuration
class Config:
    # Model settings
    model_name = "mistralai/Mistral-7B-v0.1"
    training_data_path = "/Users/new/elara_main/training_data/medical_professional_training.jsonl"
    
    # LoRA configuration
    lora_r = 16                    # Low-rank dimension
    lora_alpha = 32               # LoRA scaling parameter
    lora_dropout = 0.1            # LoRA dropout
    lora_target_modules = [        # Target all attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Training settings
    output_dir = "/Users/new/elara_main/lora_adapters/medical_professional_lora"
    num_epochs = 3
    batch_size = 1                # Small batch for limited hardware
    learning_rate = 2e-4          # Standard LoRA learning rate
    max_length = 2048             # Maximum sequence length
    
    # Quantization for memory efficiency
    use_4bit_quantization = True
    bnb_4bit_compute_dtype = torch.float16
    bnb_4bit_quant_type = "nf4"

def load_training_data(file_path):
    """Load and preprocess training data"""
    print(f"📚 Loading training data from {file_path}")
    
    training_examples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                training_examples.append(example)
    
    print(f"✅ Loaded {len(training_examples)} training examples")
    return training_examples

def format_training_example(example):
    """Format training example for instruction tuning"""
    instruction = example['instruction']
    response = example['response']
    
    # Use Mistral chat format
    formatted_text = f"<s>[INST] You are a medical professional assistant. {instruction} [/INST] {response}</s>"
    return formatted_text

def prepare_dataset(training_examples, tokenizer, max_length):
    """Prepare dataset for training"""
    print("🔄 Preparing dataset...")
    
    formatted_examples = []
    for example in training_examples:
        formatted_text = format_training_example(example)
        formatted_examples.append(formatted_text)
    
    # Tokenize
    tokenized = tokenizer(
        formatted_examples,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    })
    
    print(f"✅ Dataset prepared with {len(dataset)} examples")
    return dataset

def setup_model_and_tokenizer(config):
    """Setup model and tokenizer with quantization"""
    print(f"🔧 Loading model: {config.model_name}")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Set pad token (Mistral doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Quantization config for memory efficiency
    if config.use_4bit_quantization:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )
        print("🔧 Using 4-bit quantization for memory efficiency")
    else:
        quantization_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✅ Model and tokenizer setup complete!")
    return model, tokenizer

def train_model(model, tokenizer, dataset, config):
    """Train the model with LoRA"""
    print("🚀 Starting LoRA training...")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,  # Effective batch size of 4
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        fp16=True,  # Mixed precision training
        dataloader_drop_last=True,
        report_to=None,  # Disable wandb for now
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🔥 Training started!")
    trainer.train()
    
    # Save the trained model
    trainer.save_model()
    print(f"💾 Model saved to {config.output_dir}")
    
    return trainer

def test_model(model, tokenizer, test_prompt):
    """Test the trained model with a sample prompt"""
    print("🧪 Testing trained model...")
    
    # Format test prompt
    formatted_prompt = f"<s>[INST] You are a medical professional assistant. {test_prompt} [/INST]"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        model.eval()
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    generated_text = full_response[len(formatted_prompt):].strip()
    
    print(f"📝 Test prompt: {test_prompt}")
    print(f"🤖 Model response: {generated_text}")
    
    return generated_text

def save_training_info(config, training_examples):
    """Save training information for future reference"""
    info = {
        "model_name": config.model_name,
        "training_timestamp": datetime.now().isoformat(),
        "num_training_examples": len(training_examples),
        "lora_config": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules
        },
        "training_config": {
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_length": config.max_length
        },
        "specialties_covered": [
            "Cardiology (STEMI)",
            "Internal Medicine (UTI/AKI)", 
            "Infectious Disease (Meningitis)",
            "Hematology (Anticoagulation)",
            "Neurology (Stroke)",
            "Pediatrics (Febrile infant)",
            "Surgery/Trauma (MVA)",
            "Emergency Medicine (PE)",
            "Radiology (CXR interpretation)",
            "Psychiatry (Acute psychosis)",
            "Dermatology (Melanoma)",
            "Orthopedics (Hip fracture)",
            "OB/GYN (Preeclampsia)"
        ]
    }
    
    info_path = os.path.join(config.output_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📋 Training info saved to {info_path}")

def main():
    """Main training function"""
    print("🏁 ELARA AI MEDICAL PROFESSIONAL LORA TRAINING STARTED! 🏁\n")
    
    # Initialize config
    config = Config()
    
    # Load training data
    training_examples = load_training_data(config.training_data_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare dataset
    dataset = prepare_dataset(training_examples, tokenizer, config.max_length)
    
    # Train model
    trainer = train_model(model, tokenizer, dataset, config)
    
    # Test the trained model
    test_prompts = [
        "A 35-year-old patient presents with chest pain. What's your initial assessment?",
        "Explain the key considerations in pediatric fever management.",
        "What are the indications for emergency surgery in trauma patients?"
    ]
    
    print("\n🧪 TESTING TRAINED MODEL:")
    for prompt in test_prompts:
        test_model(model, tokenizer, prompt)
        print("-" * 80)
    
    # Save training information
    save_training_info(config, training_examples)
    
    print("\n🎉 LORA TRAINING COMPLETE! 🎉")
    print(f"📁 Trained adapter saved to: {config.output_dir}")
    print("🩺 Your medical professional AI is ready to assist healthcare providers!")

if __name__ == "__main__":
    main()
