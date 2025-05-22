# Elara AI: Backend-Model Communication Architecture

## 🧠 Introduction

This document explains how the Elara AI backend system communicates with AI models to process medical queries. Understanding this architecture will help you run tests, make modifications, and extend the system.

The Elara AI system uses a modular design where the FastAPI backend acts as an orchestrator, coordinating the flow of data between the user interface and various AI models. These models include the medical reasoning model (Mistral with LoRA fine-tuning), translation services, and potentially speech recognition components.

## 📂 Code Structure Overview

Before diving into the communication flow, let's understand the key files involved:

```
backend/
├── main.py              # Main application entry point and FastAPI setup
├── routes.py            # API route definitions (endpoints)
├── models.py            # Model manager for loading and using AI models
├── config.py            # Configuration settings
├── schemas.py           # Pydantic data models for validation
└── rag_utils.py         # Retrieval Augmented Generation utilities
```

Each of these files has a specific role:

- **main.py**: Sets up the FastAPI application, middleware, and event handlers for startup/shutdown
- **routes.py**: Defines the API endpoints that clients can call
- **models.py**: Contains the `ModelManager` class that handles AI model loading and inference
- **config.py**: Houses all configuration settings in the `Settings` class
- **schemas.py**: Defines data structures using Pydantic for request/response validation
- **rag_utils.py**: Provides utilities for retrieving relevant medical context (knowledge base)

## 🔄 Backend-Model Communication Flow

### 1. Initialization Phase (Startup)

When the FastAPI server starts, the following sequence occurs:

```
main.py (startup_event)
   ↓
creates ModelManager
   ↓
ModelManager.initialize_models()
   ↓
_load_lightweight_models()
   ↓
Loads LoRA adapter from disk
```

Code walkthrough:

1. In `main.py`, the `startup_event` function is triggered when the server starts:

```python
@app.on_event("startup")
async def startup_event():
    global model_manager
    model_manager = ModelManager(settings)
    await model_manager.initialize_models()
```

2. The `ModelManager` is instantiated with settings and then initializes the models:

```python
# In models.py
async def initialize_models(self):
    print("🔄 Initializing AI models...")
    
    if not TORCH_AVAILABLE:
        await self._initialize_simulation_mode()
        return
    
    # Initialize the RAG system
    await medical_rag.initialize()
    
    # Try to load lightweight models first
    try:
        await self._load_lightweight_models()
        print("✅ Lightweight models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load AI models: {e}")
        await self._initialize_simulation_mode()
```

3. The `_load_lightweight_models()` method loads the LoRA adapter:

```python
async def _load_lightweight_models(self):
    # Path to your LoRA adapter
    lora_path = self.settings.models_dir / "medical_lora"
    
    if lora_path.exists():
        # Load the PEFT configuration
        peft_config = PeftConfig.from_pretrained(str(lora_path))
        
        # Load tokenizer, then base model
        tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Resize model embeddings if needed
        if len(tokenizer) != base_model.config.vocab_size:
            base_model.resize_token_embeddings(len(tokenizer))
        
        # Apply LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            str(lora_path),
            torch_dtype=torch.float32,
            is_trainable=False
        )
        
        # Store model and tokenizer
        self.models["medical_qa"] = {
            "model": model,
            "tokenizer": tokenizer,
            "type": "lora"
        }
        self.model_status["medical_qa"] = "loaded"
```

### 2. Query Processing Flow

When a user sends a query, the following sequence occurs:

```
Client request
   ↓
routes.py (ask_medical_question endpoint)
   ↓
ModelManager.detect_language
   ↓
ModelManager.translate_to_english (if needed)
   ↓
ModelManager.retrieve_medical_context (RAG)
   ↓
ModelManager.generate_medical_response
   ↓
ModelManager.translate_from_english (if needed)
   ↓
Response returned to client
```

Code walkthrough:

1. In `routes.py`, the `ask_medical_question` endpoint handles incoming requests:

```python
@chat_router.post("/ask", response_model=ChatResponse)
async def ask_medical_question(request: ChatRequest, http_request: Request):
    # Get model manager from request state
    model_manager = getattr(http_request.state, 'model_manager', None)
    
    # Step 1: Detect language
    detected_language = request.language or await model_manager.detect_language(request.question)
    
    # Step 2: Translate to English if needed
    english_question = request.question
    if detected_language != "en":
        english_question = await model_manager.translate_to_english(
            request.question, detected_language)
    
    # Step 3: Retrieve relevant medical context (RAG)
    relevant_context = await model_manager.retrieve_medical_context(
        english_question, max_sources=5)
    
    # Step 4: Generate AI response
    ai_response = await model_manager.generate_medical_response(
        question=english_question,
        context=relevant_context,
        user_type=request.user_type,
        include_sources=request.include_sources)
    
    # Step 5: Translate back to user's language
    final_response = ai_response["response"]
    if detected_language != "en":
        final_response = await model_manager.translate_from_english(
            ai_response["response"], detected_language)
    
    # Create response object and return
    response = ChatResponse(
        response=final_response,
        confidence=ai_response.get("confidence", 0.8),
        sources=ai_response.get("sources", []),
        language_detected=detected_language,
        processing_time=time.time() - start_time,
        conversation_id=request.conversation_id or str(uuid.uuid4()),
        safety_warning=safety_warning
    )
    
    return response
```

### 3. Model Inference (The Core of Communication)

The most important part is how the backend actually gets responses from the model. This happens in the `generate_medical_response` method:

```python
async def generate_medical_response(self, question, context, user_type, include_sources):
    # Construct prompt with RAG context
    prompt = f"Human: {question}\n"
    
    # Add any context from RAG
    if context_text:
        prompt += f"{context_text.strip()}\n"
    
    # Add the Assistant prefix for completion
    prompt += "Assistant:"
    
    # Check if we're using the LoRA model
    if isinstance(self.models["medical_qa"], dict) and "model" in self.models["medical_qa"]:
        # Using LoRA model
        model = self.models["medical_qa"]["model"]
        tokenizer = self.models["medical_qa"]["tokenizer"]
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2
            )
        
        # Decode the output
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
        
        # Extract only the assistant's response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        # Clean up special tokens
        response = response.replace("<USER>", "").replace("<ASSISTANT>", "")
                           .replace("<s>", "").replace("</s>", "")
                           .replace("<|endoftext|>", "").strip()
        
        # Add safety disclaimers
        if user_type == "patient":
            response += "\n\nPlease note: This information is for educational purposes only. Always consult a healthcare professional for personalized medical advice."
        
        return {
            "response": response,
            "confidence": 0.85,
            "sources": sources,
            "needs_professional_consultation": self._needs_professional_consultation(question, user_type)
        }
```

## 🔍 Key Model Communication Points

The following details explain how exactly the backend talks to the models:

### Model Loading

The models are loaded using Hugging Face's transformers and PEFT libraries:

1. **Base Model Loading**: 
   ```python
   base_model = AutoModelForCausalLM.from_pretrained(
       peft_config.base_model_name_or_path,
       torch_dtype=torch.float32,
       low_cpu_mem_usage=True
   )
   ```

2. **LoRA Adapter Application**:
   ```python
   model = PeftModel.from_pretrained(
       base_model,
       str(lora_path),
       torch_dtype=torch.float32,
       is_trainable=False
   )
   ```

### Model Inference

To get responses from the model:

1. **Tokenization**:
   ```python
   inputs = tokenizer(prompt, return_tensors="pt")
   ```

2. **Generation**:
   ```python
   output_sequences = model.generate(
       input_ids=inputs.input_ids,
       attention_mask=inputs.attention_mask,
       max_new_tokens=300,
       do_sample=True,
       temperature=0.7,
       top_p=0.95,
       repetition_penalty=1.1,
       no_repeat_ngram_size=2
   )
   ```

3. **Decoding**:
   ```python
   generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
   ```

### Parameter Explanation

Understanding the generation parameters helps you control the model's behavior:

- **max_new_tokens**: Maximum number of tokens to generate (longer values = longer responses)
- **do_sample**: If True, uses sampling; if False, uses greedy decoding
- **temperature**: Controls randomness (lower = more deterministic, higher = more creative)
- **top_p**: Nucleus sampling parameter (lower values = more focused on likely tokens)
- **repetition_penalty**: Penalizes repeating the same words/phrases
- **no_repeat_ngram_size**: Prevents repeating of n-grams of this size

## 📋 Testing the Backend-Model Communication

You can test the backend-model communication in several ways:

### 1. Using curl from the command line:

```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of diabetes?", "user_type": "patient"}'
```

### 2. Using the Swagger UI:

1. Open a browser and go to http://localhost:8000/docs
2. Find the POST /chat/ask endpoint
3. Click "Try it out"
4. Enter your test parameters in the request body
5. Click "Execute"

### 3. Creating a Python test script:

```python
import requests
import json

# Define the API endpoint
url = "http://localhost:8000/chat/ask"

# Create a test query
data = {
    "question": "What are the symptoms of diabetes?",
    "user_type": "patient",
    "include_sources": True
}

# Send the request
response = requests.post(url, json=data)

# Print the response
print(json.dumps(response.json(), indent=2))
```

Save this as `test_api.py` in the project root and run with `python test_api.py`.

## 🔧 Common Modification Scenarios

Here are some common scenarios for modifying the backend-model communication:

### 1. Changing the model's response style:

To make the model more or less verbose, adjust the generation parameters in `models.py`:

```python
output_sequences = model.generate(
    # ... other parameters ...
    max_new_tokens=500,  # Increase for longer responses
    temperature=0.5,     # Lower for more focused, deterministic responses
    # ... other parameters ...
)
```

### 2. Adding a new model type:

To add a new specialist model (e.g., for radiology), you would:

1. Add the model config in `config.py`:
```python
"radiology_qa": {
    "name": "radiology-lora",
    "path": self.models_dir / "radiology_lora",
    "type": "causal_lm",
    "base_model": "microsoft/DialoGPT-medium",
    "max_length": 1024,
    "temperature": 0.7,
    "enabled": True
}
```

2. Add loading logic in `models.py`:
```python
# Load radiology model
if self.settings.is_model_enabled("radiology_qa"):
    await self._load_radiology_model()
```

3. Add a method to use the model in `models.py`:
```python
async def analyze_radiology_image(self, image_path, question):
    # Model-specific code here
    # ...
```

4. Add a new endpoint in `routes.py`:
```python
@chat_router.post("/analyze_radiology")
async def analyze_radiology(request: RadiologyRequest, http_request: Request):
    # Endpoint-specific code here
    # ...
```

### 3. Modifying the RAG system:

To enhance the context retrieval, modify `rag_utils.py`:

```python
async def get_context_for_query(self, query: str, max_results: int = 3) -> str:
    # Modify the format or number of retrieved documents
    # ...
```

## 🧪 Troubleshooting

Common issues you might encounter and how to solve them:

### 1. Model not loading:

If you see errors about models not loading, check:
- Path to model files in `settings.models_dir`
- Whether the LoRA adapter exists in the expected location
- Python environment has all required packages installed

### 2. Slow responses:

If model responses are taking too long:
- Reduce `max_new_tokens` in the generation parameters
- Use a smaller model or more quantization (e.g., 4-bit instead of 8-bit)
- Check if your machine has enough memory and CPU/GPU resources

### 3. Import errors:

If you see import errors:
- Ensure your Python environment has all required packages
- Check the import paths (especially when running from different directories)
- Modify relative imports as needed (e.g., change `from backend.rag_utils` to `import rag_utils`)

## 🚀 Advanced Testing

For more advanced testing scenarios:

### 1. Testing with different user types:

```python
# Test as a doctor
doctor_data = {
    "question": "What are the latest treatment guidelines for type 2 diabetes?",
    "user_type": "doctor",
    "include_sources": True
}

# Test as a patient
patient_data = {
    "question": "How can I manage my diabetes symptoms?",
    "user_type": "patient",
    "include_sources": True
}

# Compare responses
doctor_response = requests.post(url, json=doctor_data).json()
patient_response = requests.post(url, json=patient_data).json()
```

### 2. Testing multilingual support:

```python
# Test with Spanish
spanish_data = {
    "question": "¿Cuáles son los síntomas de la diabetes?",
    "language": "es",
    "user_type": "patient"
}

response = requests.post(url, json=spanish_data).json()
print(f"Detected language: {response['language_detected']}")
print(f"Response: {response['response']}")
```

### 3. Benchmarking response times:

```python
import time
import statistics

# Prepare test queries
test_queries = [
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed?",
    "What should I know about COVID-19 prevention?",
    # Add more test queries...
]

# Measure response times
times = []
for query in test_queries:
    start_time = time.time()
    response = requests.post(url, json={"question": query, "user_type": "patient"})
    end_time = time.time()
    times.append(end_time - start_time)

# Calculate statistics
print(f"Average response time: {statistics.mean(times):.2f} seconds")
print(f"Median response time: {statistics.median(times):.2f} seconds")
print(f"Min response time: {min(times):.2f} seconds")
print(f"Max response time: {max(times):.2f} seconds")
```

## 📚 Further Resources

- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/
- **PEFT (Parameter-Efficient Fine-Tuning)**: https://huggingface.co/docs/peft/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **RAG (Retrieval Augmented Generation)**: https://www.pinecone.io/learn/retrieval-augmented-generation/

## 🔮 Conclusion

The Elara AI backend-model communication architecture is designed to be modular, scalable, and extensible. By understanding how the components interact, you can test, modify, and extend the system to meet your specific requirements.

Remember that the quality of responses depends on:
1. The base model size and capabilities
2. The quality and quantity of fine-tuning data
3. The retrieval mechanism (RAG) performance
4. The prompt engineering and generation parameters

By experimenting with these elements, you can optimize the performance of your Elara AI medical assistant for your specific use case.
