# Elara AI: LoRA Fine-tuning Guide

## Introduction to LoRA and Model Adaptation

Welcome to this comprehensive guide on LoRA fine-tuning for Elara AI! 🚀 This document explains how we adapt pre-trained language models to specialize in medical conversations using Low-Rank Adaptation (LoRA) techniques.

### What is LoRA?

LoRA (Low-Rank Adaptation) is a technique that allows us to fine-tune large language models efficiently. Instead of updating all parameters in a model (which could be billions), LoRA adds small "adapter" matrices to specific layers of the model. This approach has several advantages:

- **Memory Efficiency**: Requires significantly less GPU/CPU memory
- **Storage Efficiency**: Adapter weights are only a fraction of the full model size
- **Training Speed**: Faster training times since fewer parameters are updated
- **Modularity**: Multiple LoRA adapters can be swapped for different use cases

In Elara's current implementation, we're using LoRA to adapt DialoGPT-medium for medical conversations, but the same technique can be applied to larger models like Mistral 7B.

## Current Model: DialoGPT-medium

Our current implementation uses Microsoft's DialoGPT-medium (355M parameters) as the base model. DialoGPT was pre-trained on conversational data, making it well-suited for dialogue applications like Elara.

```python
# Loading the base model
model_name = "microsoft/DialoGPT-medium"
```

### Pros and Cons of the Current Model

**Advantages of DialoGPT-medium:**
- Fast inference and training on consumer hardware (including CPU)
- Naturally conversational
- Well-documented and stable

**Limitations:**
- Limited medical knowledge compared to larger models
- Restricted reasoning capabilities
- May produce incoherent text without careful prompt engineering

## Understanding LoRA Hyperparameters

Let's break down each hyperparameter in our `MedicalLoRAConfig` to understand its function and how to tune it:

```python
@dataclass
class MedicalLoRAConfig:
    # LoRA parameters
    lora_r: int = 16              # Rank (complexity)
    lora_alpha: int = 32          # Learning rate scaling
    lora_dropout: float = 0.1     # Prevent overfitting
    target_modules: List[str] = ["c_attn", "c_proj"]  # Which layers to adapt
    
    # Training parameters
    num_epochs: int = 3           # Training cycles
    batch_size: int = 4           # Memory efficiency
    learning_rate: float = 2e-4   # How fast to learn
    max_samples: int = 5000       # Training data size
```

### LoRA-specific Parameters

#### `lora_r: int = 16` (Rank)
The rank parameter determines the dimensionality of the low-rank matrices used in LoRA. In simple terms, it controls how much "expressivity" or "learning capacity" your adaptation has.

- **Higher values** (e.g., 16, 32, 64): More expressive, can learn more complex adaptations, but uses more memory and might overfit on small datasets
- **Lower values** (e.g., 4, 8): Less expressive, more memory efficient, less prone to overfitting

**How to choose:** Start with 8 for small datasets (hundreds of examples) and 16-32 for larger datasets (thousands of examples). For complex domains like medicine, higher ranks are often beneficial if you have sufficient data.

#### `lora_alpha: int = 32` (Scaling Factor)
Alpha controls how much influence the LoRA updates have relative to the original pre-trained weights. It's typically set to `lora_r * 2` by convention.

- **Higher values**: Stronger influence of fine-tuned content, more divergence from base model
- **Lower values**: More conservative updates, staying closer to the base model's behavior

**How to choose:** The rule of thumb is `lora_alpha = 2 * lora_r`, but you can experiment with higher values if you want the model to learn more from your fine-tuning data.

#### `lora_dropout: float = 0.1` (Dropout Rate)
Like standard dropout, this randomly zeros out some LoRA parameters during training to prevent overfitting.

- **Higher values** (e.g., 0.2, 0.3): More regularization, good for small datasets
- **Lower values** (e.g., 0.05, 0.0): Less regularization, better for large datasets

**How to choose:** Start with 0.1 and adjust based on validation performance. If the model is underfitting, reduce dropout; if overfitting, increase it.

#### `target_modules: List[str] = ["c_attn", "c_proj"]` (Target Layers)
This specifies which layers in the model receive LoRA adapters. Different models have different layer names:

- **DialoGPT/GPT-2**: `["c_attn", "c_proj"]` (attention layers)
- **LLaMA/Mistral**: `["q_proj", "k_proj", "v_proj", "o_proj"]` (query, key, value, output projections)
- **BLOOM**: `["query_key_value", "dense"]`

**How to choose:** Focus on attention layers as they're most influential for language understanding. For Mistral 7B, you would target the query, key, and value projection matrices.

### General Training Parameters

#### `num_epochs: int = 3` (Training Epochs)
The number of complete passes through the training dataset.

- **Higher values**: More learning, but risk of overfitting
- **Lower values**: Less chance to overfit, but might underfit

**How to choose:** Monitor validation loss. For small datasets (hundreds of examples), 3-5 epochs is common. For larger datasets, 1-3 epochs might be sufficient.

#### `batch_size: int = 4` (Batch Size)
The number of examples processed together in one forward/backward pass.

- **Higher values**: Faster training, more stable gradients, but requires more memory
- **Lower values**: Less memory required, but noisier updates and slower training

**How to choose:** For CPU training, 2-4 is practical. For GPUs, you can use 8-32 depending on VRAM. Adjust based on available memory.

#### `learning_rate: float = 2e-4` (Learning Rate)
Controls how large the parameter updates are during training.

- **Higher values** (e.g., 5e-4): Faster learning but risk of instability
- **Lower values** (e.g., 5e-5): More stable but slower convergence

**How to choose:** For LoRA fine-tuning, 1e-4 to 5e-4 is usually a good range. If loss is unstable, decrease the learning rate.

#### `max_samples: int = 5000` (Dataset Size)
Maximum number of training examples to use.

**How to choose:** Use as much high-quality data as you can process. For medical applications, quality and diversity are more important than sheer quantity.

## Migrating to Mistral 7B

Switching from DialoGPT-medium to Mistral 7B will substantially improve Elara's capabilities but requires more computational resources.

### Hardware Requirements for Mistral 7B

- **Training (full precision)**: 16+ GB VRAM GPU (A100, A6000, RTX 3090, etc.)
- **Training (8-bit quantized)**: 8+ GB VRAM GPU
- **Inference (4-bit quantized)**: Can run on M1 Mac with 16GB RAM using llama.cpp
- **CPU-only**: Not recommended for training but possible for inference with quantization

### Code Changes Required

#### 1. Model Loading

```python
# Replace this:
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# With this:
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "mistralai/Mistral-7B-v0.1"  # or another Mistral variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Enable quantization for memory efficiency
    device_map="auto"   # Automatically manage device placement
)
```

#### 2. LoRA Configuration

```python
@dataclass
class MistralLoRAConfig:
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Updated target modules for Mistral architecture
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Training parameters - adjust for larger model
    num_epochs: int = 2           # Fewer epochs for larger model
    batch_size: int = 1           # Smaller batch size due to memory constraints
    learning_rate: float = 1e-4   # Slightly lower learning rate for stability
    max_samples: int = 5000
```

#### 3. PEFT Configuration

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA for Mistral
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.target_modules,
    bias="none",
    task_type="CAUSAL_LM"
)

# Create PEFT model
model = get_peft_model(model, lora_config)
```

#### 4. Training Script Adjustments

- Add gradient accumulation to handle smaller batch sizes
- Use mixed precision training (fp16) to reduce memory usage
- Consider adding gradient checkpointing for memory efficiency

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./mistral-medical-lora",
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    learning_rate=config.learning_rate,
    num_train_epochs=config.num_epochs,
    weight_decay=0.01,
    fp16=True,  # Use mixed precision
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)
```

### Running Inference with Mistral+LoRA

For inference on hardware with limited resources (like M1 Mac), using llama.cpp is recommended:

```python
from llama_cpp import Llama

# Load the quantized model (GGUF format) with LoRA adapter
llm = Llama(
    model_path="mistral-7b-q4_0.gguf",     # Base quantized model
    lora_path="medical_lora.bin",          # LoRA weights
    n_gpu_layers=1,                        # Use 1+ GPU layers on M1
    n_ctx=2048,                            # Context window size
    n_threads=6                            # CPU thread count
)

# Generate text
output = llm(
    "USER: I have a headache and fever, what could it be? ASSISTANT:",
    max_tokens=512,
    temperature=0.15,
    repeat_penalty=1.3
)
```

## Creating Different LoRA Adapters for Different Use Cases

One of the major advantages of LoRA is the ability to create multiple adapters for different use cases. In Elara's case, you might want specialized adapters for different audiences:

1. **patient_lora**: Simplified, non-technical explanations for patients
2. **student_lora**: Educational content with more details for medical students
3. **researcher_lora**: Technical, research-oriented responses
4. **emergency_lora**: Concise, action-oriented guidance for emergency situations

### How to Create a New Adapter Type

#### 1. Prepare Specialized Training Data

Each adapter needs its own training dataset that reflects the desired communication style and content focus:

```python
def prepare_patient_dataset():
    """Prepare a dataset of patient-friendly medical explanations"""
    examples = [
        {
            "instruction": "Explain diabetes in simple terms",
            "response": "Diabetes is when your body has trouble using sugar properly. Your body needs a hormone called insulin to move sugar from your blood into your cells for energy. In diabetes, either your body doesn't make enough insulin or can't use it well, causing sugar to build up in your blood."
        },
        # Add more examples with patient-friendly explanations
    ]
    
    # Convert to dataset format
    return Dataset.from_dict({
        "instruction": [ex["instruction"] for ex in examples],
        "response": [ex["response"] for ex in examples]
    })
```

#### 2. Create a Specialized Configuration

Each adapter might benefit from different hyperparameters:

```python
@dataclass
class PatientLoRAConfig(MedicalLoRAConfig):
    # Override base parameters as needed
    lora_r: int = 8               # Lower rank for simpler language patterns
    lora_alpha: int = 16
    learning_rate: float = 3e-4   # Slightly higher to adapt more quickly
    
    # Add specialized parameters if needed
    max_explanation_length: int = 300  # Keep explanations shorter for patients
```

#### 3. Training Script

Create a dedicated training script for each adapter type:

```python
def train_patient_lora(model_name="microsoft/DialoGPT-medium", output_dir="patient_lora"):
    """Train a patient-friendly LoRA adapter"""
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure LoRA
    config = PatientLoRAConfig()
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Prepare data
    dataset = prepare_patient_dataset()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Train
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            # Add other training arguments
        ),
        train_dataset=tokenized_dataset,
        data_collator=DefaultDataCollator(),
    )
    
    trainer.train()
    
    # Save adapter
    model.save_pretrained(output_dir)
    return output_dir
```

#### 4. Using Multiple Adapters

You can load different adapters based on the context or user type:

```python
class ElaryAIManager:
    def __init__(self):
        # Load base model once
        self.base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        # Load different LoRA adapters
        self.adapters = {
            "doctor": PeftModel.from_pretrained(self.base_model, "medical_lora"),
            "patient": PeftModel.from_pretrained(self.base_model, "patient_lora"),
            "student": PeftModel.from_pretrained(self.base_model, "student_lora"),
            "researcher": PeftModel.from_pretrained(self.base_model, "researcher_lora"),
            "emergency": PeftModel.from_pretrained(self.base_model, "emergency_lora")
        }
    
    def process_query(self, query, user_type="patient"):
        """Process a query using the appropriate adapter"""
        # Select the right adapter for this user
        model = self.adapters.get(user_type, self.adapters["patient"])
        
        # Generate response
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            temperature=0.15,
            repetition_penalty=1.3
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Best Practices and Optimization Tips

### 1. Data Quality Over Quantity

For medical fine-tuning, prioritize:
- **Accuracy**: Have medical professionals review training data
- **Diversity**: Cover a wide range of conditions, symptoms, and scenarios
- **Ethical Considerations**: Ensure responses include appropriate disclaimers and cautions

### 2. Evaluation Metrics

Measure adapter quality with:
- **Medical Accuracy**: Rated by healthcare professionals
- **Readability**: Different scores for different audience types (Flesch-Kincaid for patients)
- **Response Relevance**: How directly the response addresses the query

### 3. Memory Optimization

If working with limited resources:
- **Gradient Checkpointing**: Trade computation for memory
- **Mixed Precision Training**: Use fp16 instead of fp32
- **Quantization**: Train in 8-bit with `bitsandbytes` library
- **Progressive Training**: Start with a smaller model, then transfer knowledge

### 4. Inference Optimization

For faster deployment:
- **Model Quantization**: Convert to 4-bit or 8-bit formats
- **Response Caching**: Cache common responses
- **Batched Inference**: Process multiple queries together when possible

## Troubleshooting Common Issues

### Training Issues

1. **Loss Doesn't Decrease**
   - Try increasing learning rate
   - Verify dataset formatting
   - Check for data quality issues

2. **Out of Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use quantization
   - Target fewer layers with LoRA

### Inference Issues

1. **Repetitive or Generic Responses**
   - Increase temperature (0.2-0.7)
   - Adjust repetition penalty (1.1-1.5)
   - Review training data quality

2. **Slow Generation**
   - Use quantized models
   - Reduce context length
   - Set max_new_tokens to a reasonable limit

## Conclusion

LoRA fine-tuning provides a powerful and efficient way to adapt large language models for specialized domains like medicine. By understanding the hyperparameters and how to create different adapters, you can maximize Elara's effectiveness for different user types and scenarios.

As you continue to develop Elara AI, consider progressively moving to larger models like Mistral 7B when resources permit, while maintaining the flexibility and efficiency of the LoRA approach.

Happy fine-tuning! 🧠🔬
