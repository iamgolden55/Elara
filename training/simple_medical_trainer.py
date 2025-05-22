#!/usr/bin/env python3
"""
🩺 Elara AI - Simple Medical LoRA Trainer 🩺
Streamlined version for reliable training!
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def main():
    print("🩺 Elara AI - Simple Medical LoRA Trainer 🩺")
    
    # Configuration
    model_name = "microsoft/DialoGPT-medium"
    output_dir = "../models_files/medical_lora"
    max_length = 512
    batch_size = 2
    learning_rate = 2e-4
    num_epochs = 2
    max_samples = 1000  # Start small
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔧 Loading tokenizer and model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None,
    )
    
    # Configure LoRA
    print("⚙️ Setting up LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"],  # Just attention layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("📚 Preparing training data...")
    
    # Simple medical Q&A data
    medical_qa_pairs = [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, and blurred vision. If you experience these symptoms, consult your healthcare provider."
        },
        {
            "question": "What is hypertension?", 
            "answer": "Hypertension (high blood pressure) is a condition where blood pressure readings are consistently above 140/90 mmHg. It's often called the 'silent killer' because it typically has no symptoms."
        },
        {
            "question": "How does aspirin work?",
            "answer": "Aspirin works by blocking the production of prostaglandins, which are involved in pain, inflammation, and blood clotting. This is why it's used for pain relief and heart protection."
        },
        {
            "question": "What should I do if I have chest pain?",
            "answer": "Chest pain can be serious. If you experience severe chest pain, shortness of breath, or pain radiating to your arm or jaw, seek emergency medical attention immediately."
        },
        {
            "question": "What is the difference between a virus and bacteria?",
            "answer": "Viruses are much smaller than bacteria and need host cells to reproduce. Bacteria are single-celled organisms that can reproduce on their own. This is why antibiotics work against bacteria but not viruses."
        }
    ]
    
    # Load additional data from high-quality dataset
    quality_file = "../data/high_quality/elara_medical_quality_filtered.json"
    if os.path.exists(quality_file):
        print("📖 Loading additional medical articles...")
        with open(quality_file, 'r') as f:
            quality_data = json.load(f)
        
        # Convert articles to Q&A format
        for article in quality_data[:50]:  # Just use first 50
            if article.get('abstract') and len(article.get('abstract', '')) > 100:
                title = article.get('title', '').strip()
                abstract = article.get('abstract', '').strip()
                
                qa_pair = {
                    "question": f"Can you explain the medical research about {title.lower()}?",
                    "answer": f"Research shows: {abstract[:300]}... This study provides important insights for medical practice. Consult healthcare professionals for specific medical advice."
                }
                medical_qa_pairs.append(qa_pair)
    
    # Format for training
    training_texts = []
    for qa in medical_qa_pairs:
        text = f"Medical Question: {qa['question']}\nMedical Answer: {qa['answer']}<|endoftext|>"
        training_texts.append(text)
    
    print(f"📊 Created {len(training_texts)} training examples")
    
    # Tokenize the data
    print("🔤 Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(
            examples,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Convert to dataset
    tokenized_data = tokenize_function(training_texts)
    
    # Create labels (same as input_ids for causal LM)
    tokenized_data["labels"] = tokenized_data["input_ids"].clone()
    
    # Convert to HuggingFace dataset format
    dataset_dict = {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": tokenized_data["labels"]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train/eval
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"🏋️ Training samples: {len(train_dataset)}")
    print(f"📊 Evaluation samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],  # Disable wandb
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training!
    print("🚀 Starting medical LoRA training...")
    trainer.train()
    
    # Save the model
    print("💾 Saving trained model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Test the model
    print("🧪 Testing the trained model...")
    test_questions = [
        "What are the symptoms of diabetes?",
        "How does aspirin work?",
        "What should I do if I have chest pain?"
    ]
    
    device = "cpu"  # Force CPU for compatibility
    model.eval()
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        
        # Format input
        input_text = f"Medical Question: {question}\nMedical Answer:"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(input_text):].strip()
        print(f"🤖 Answer: {answer}")
    
    print("\n🎉 Medical LoRA training completed successfully!")
    print(f"💾 Model saved to: {output_dir}")
    print("🩺 Your AI doctor is ready to help patients!")

if __name__ == "__main__":
    main()
