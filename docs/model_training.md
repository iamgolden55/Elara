# Elara AI: Model Training Guide

This guide provides detailed instructions on training and fine-tuning language models for the Elara AI medical assistant. It covers the entire training process, from data preparation to LoRA fine-tuning and evaluation.

## Overview

The Elara AI system uses Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically Low-Rank Adaptation (LoRA), to adapt foundation models to the medical domain. This approach allows us to fine-tune large language models using minimal computational resources while achieving excellent performance.

## Prerequisites

Before training, ensure you have:

- Python 3.10+ installed
- PyTorch with CUDA support (for GPU training)
- Transformers, PEFT, and related libraries
- Sufficient disk space for models and datasets
- GPU with at least 16GB VRAM (for 7B models) or CPU with 32GB+ RAM

Install required packages:

```bash
pip install -r training_requirements.txt
```

## Data Preparation

High-quality training data is crucial for effective fine-tuning. Follow these steps to prepare your dataset:

### 1. Data Collection

Collect medical data from various sources:
- Medical Q&A pairs
- Clinical guidelines
- De-identified doctor-patient dialogues
- Medical textbooks and literature

### 2. Data Cleaning

Clean the collected data:

```python
import pandas as pd
import re

# Load raw data
data = pd.read_csv('raw_medical_data.csv')

# Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[\r\n\t]', ' ', text)  # Remove newlines, tabs
    text = text.strip()
    return text

data['question'] = data['question'].apply(clean_text)
data['answer'] = data['answer'].apply(clean_text)

# Remove duplicates
data = data.drop_duplicates()

# Filter out very short or very long texts
data = data[(data['question'].str.len() > 10) & 
            (data['question'].str.len() < 512) &
            (data['answer'].str.len() > 20) &
            (data['answer'].str.len() < 1024)]

# Save cleaned data
data.to_csv('cleaned_medical_data.csv', index=False)
```

### 3. Format for Training

Format data into instruction-tuning format:

```python
def format_for_instruction_tuning(row, user_type="general"):
    # Create a prompt with system instruction
    system_instruction = "You are Elara, a professional medical AI assistant trained to provide accurate, evidence-based information. Answer with compassion and clarity."
    
    # Format based on user type
    if user_type == "doctor":
        system_instruction += " You are communicating with a medical professional, so use appropriate technical terminology."
    elif user_type == "patient":
        system_instruction += " You are communicating with a patient, so use patient-friendly language and avoid excessive medical jargon."
    
    # Combine into full example
    formatted_example = f"<|system|>\n{system_instruction}\n<|user|>\n{row['question']}\n<|assistant|>\n{row['answer']}"
    
    return formatted_example

# Apply formatting
data['formatted_text'] = data.apply(format_for_instruction_tuning, axis=1)

# Save formatted data
with open('medical_training_data.jsonl', 'w') as f:
    for text in data['formatted_text']:
        f.write(json.dumps({'text': text}) + '\n')
```

## Training Configuration

Create a training configuration file for each model adapter you want to train:

```json
{
  "base_model": "microsoft/DialoGPT-medium",
  "model_type": "causal_lm",
  "output_dir": "../models_files/medical_lora",
  "train_data_path": "../training_data/medical_professional_training.jsonl",
  "eval_data_path": "../training_data/medical_professional_eval.jsonl",
  "lora_config": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.1,
    "bias": "none",
    "target_modules": ["c_attn", "c_proj"]
  },
  "training_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 10,
    "evaluation_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "fp16": true,
    "output_dir": "../models_files/medical_lora_checkpoints"
  }
}
```

Save this as `training_configs/medical_professional_lora.json`.

## Running the Training Process

### 1. Train the Model

Run the training script:

```bash
python training/medical_lora_trainer.py --config training_configs/medical_professional_lora.json
```

The training script performs the following steps:

1. Load the base model and tokenizer
2. Configure LoRA for efficient fine-tuning
3. Prepare the dataset and dataloaders
4. Train the model using the specified configuration
5. Save the trained LoRA adapter and tokenizer

### 2. Monitor Training

Monitor training progress with Weights & Biases integration:

```python
# In medical_lora_trainer.py
import wandb

wandb.init(project="elara-medical-lora")
wandb.config.update(config)

# Log metrics during training
wandb.log({
    "train_loss": loss,
    "epoch": epoch,
    "learning_rate": scheduler.get_last_lr()[0]
})
```

Access training logs at: https://wandb.ai/your-username/elara-medical-lora

## Training Multiple LoRA Adapters

For Elara AI, we train multiple specialized LoRA adapters:

1. **General Medical Adapter**: For broad medical knowledge
2. **Patient-Focused Adapter**: Uses simpler language for patients
3. **Doctor-Focused Adapter**: Uses technical language for professionals
4. **Emergency Adapter**: Specialized for urgent medical questions

To train each adapter, create a separate configuration file and run the training script with that config.

## Specialized Adapter Example: Patient-Friendly Responses

```json
{
  "base_model": "microsoft/DialoGPT-medium",
  "model_type": "causal_lm",
  "output_dir": "../models_files/patient_lora",
  "train_data_path": "../training_data/patient_friendly_training.jsonl",
  "eval_data_path": "../training_data/patient_friendly_eval.jsonl",
  "lora_config": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.1,
    "bias": "none",
    "target_modules": ["c_attn", "c_proj"]
  },
  "training_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-4
  }
}
```

Save as `training_configs/patient_lora.json` and train with:

```bash
python training/medical_lora_trainer.py --config training_configs/patient_lora.json
```

## Understanding LoRA Parameters

The key LoRA parameters are:

- **r**: Rank of the low-rank matrices. Lower values = smaller adapter but less expressive. Higher values = more expressive but larger adapter. Typical values: 8, 16, 32.
- **alpha**: Scaling factor for the LoRA adaptation. Typically set to 2x the rank. Controls how much influence the adapter has.
- **dropout**: Regularization to prevent overfitting. Typical values: 0.05-0.1.
- **target_modules**: Which layers of the base model to adapt. For transformer models, attention layers are common targets.

## Evaluating Trained Models

After training, evaluate your model on test data:

```bash
python scripts/test_medical_lora.py --model_path models_files/medical_lora --test_data path/to/test_data.jsonl
```

Key evaluation metrics:

1. **Medical accuracy**: Correctness of medical information
2. **Response relevance**: How well the response addresses the question
3. **Tone appropriateness**: Matching the tone to the user type
4. **Safety**: Avoiding harmful or misleading information

## Training Tips and Best Practices

1. **Start small**: Begin with a smaller model or dataset to debug the training pipeline
2. **Gradient accumulation**: If your GPU memory is limited, use gradient accumulation to simulate larger batch sizes
3. **Learning rate**: Start with a learning rate around 2e-4 and adjust based on training loss
4. **Early stopping**: Use early stopping to prevent overfitting if validation loss starts increasing
5. **Data quality over quantity**: A smaller, high-quality dataset often yields better results than a large, noisy one
6. **Test regularly**: Evaluate the model frequently during development to catch issues early
7. **Data augmentation**: Consider techniques like back-translation to expand your dataset
8. **Monitor GPU usage**: Keep an eye on GPU memory usage and adjust batch size accordingly

## Debugging Common Issues

### Out of Memory Errors

If you encounter CUDA out-of-memory errors:

1. Reduce batch size
2. Enable gradient accumulation
3. Use a smaller model
4. Enable mixed precision training (fp16)
5. Reduce sequence length

### Poor Training Performance

If the model isn't learning well:

1. Check your learning rate (too high or too low)
2. Examine the training data quality
3. Increase training epochs
4. Try different LoRA parameters (r, alpha)
5. Consider a different base model

### Training Code

Here's a simplified version of the training code from `medical_lora_trainer.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

def train_medical_lora(config):
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    
    # Add special tokens if needed
    special_tokens = {"pad_token": "<PAD>"}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["alpha"],
        lora_dropout=config["lora_config"]["dropout"],
        bias=config["lora_config"]["bias"],
        target_modules=config["lora_config"]["target_modules"],
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": config["train_data_path"],
        "validation": config["eval_data_path"]
    })
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config["training_args"]["output_dir"],
        num_train_epochs=config["training_args"]["num_train_epochs"],
        per_device_train_batch_size=config["training_args"]["per_device_train_batch_size"],
        learning_rate=config["training_args"]["learning_rate"],
        weight_decay=config["training_args"]["weight_decay"],
        logging_steps=config["training_args"]["logging_steps"],
        save_strategy=config["training_args"]["save_strategy"],
        save_steps=config["training_args"]["save_steps"],
        evaluation_strategy=config["training_args"]["evaluation_strategy"],
        eval_steps=config["training_args"]["eval_steps"],
        fp16=config["training_args"].get("fp16", False)
    )
    
    # Create Trainer
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )
    
    # Train model
    trainer.train()
    
    # Save the model
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    return model, tokenizer

if __name__ == "__main__":
    import json
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    train_medical_lora(config)
```

## Advanced Training Techniques

### Continued Pre-training

Before LoRA fine-tuning, you can perform continued pre-training on a larger medical corpus:

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load your medical corpus dataset
dataset = load_dataset("text", data_files={"train": "medical_corpus.txt"})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Create a data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./continued_pretrain_output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

# Train
trainer.train()
```

### Reinforcement Learning from Human Feedback (RLHF)

After fine-tuning, you can further improve the model using RLHF:

1. Collect human preferences (better/worse) on model outputs
2. Train a reward model to predict human preferences
3. Fine-tune the model with PPO to maximize the reward

This is an advanced technique but can significantly improve the quality and safety of generated responses.

## Synthetic Data Generation

If you lack sufficient training data, you can generate synthetic data:

```python
from transformers import pipeline

# Load a strong general model to generate examples
generator = pipeline("text-generation", model="gpt-j-6B")

# Generate synthetic training examples
prompts = [
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed?",
    "What are the side effects of aspirin?",
    # Add more medical questions
]

synthetic_data = []
for prompt in prompts:
    response = generator(f"Question: {prompt}\nAnswer:", max_length=200)
    synthetic_data.append({
        "question": prompt,
        "answer": response[0]["generated_text"].split("Answer:")[1].strip()
    })

# Review and filter the synthetic data before using it for training
```

## Model Merging and Distillation

To create a more efficient model:

1. **Model Merging**: Combine multiple LoRA adapters into a single model
2. **Knowledge Distillation**: Train a smaller model to mimic the larger one

These techniques can help create more efficient models for deployment.

## Conclusion

Training the Elara AI medical assistant involves careful data preparation, efficient fine-tuning with LoRA, and thorough evaluation. By following this guide, you can create specialized medical language models that provide accurate, helpful, and safe responses to medical queries.

Remember that medical AI requires ongoing maintenance and updates as medical knowledge evolves. Regularly update your training data and re-evaluate your models to ensure they provide the most current and accurate information.
