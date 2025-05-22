# Elara AI API Reference

This document provides a comprehensive reference for the Elara AI Medical Assistant API endpoints.

## Base URL

All API endpoints are relative to:

```
http://{host}:{port}
```

By default, the development server runs at `http://localhost:8000`.

## Authentication

Most endpoints do not require authentication in the development environment. In production, API key or token-based authentication can be enabled in the `config.py` file.

## Common Response Formats

Most responses follow this JSON structure:

```json
{
  "response": "AI-generated text",
  "confidence": 0.95,
  "sources": [
    {
      "title": "Source Title",
      "url": "https://source-url.com",
      "type": "medical_database",
      "confidence": 0.98,
      "relevance_score": 0.92
    }
  ],
  "language_detected": "en",
  "processing_time": 1.25,
  "conversation_id": "conv_abc123",
  "safety_warning": "This information is for educational purposes only."
}
```

## Endpoints

### Health Check

```
GET /health
```

Returns the system status, including loaded models.

**Response**:

```json
{
  "status": "healthy",
  "service": "Elara AI Medical Assistant",
  "version": "1.0.0",
  "models_loaded": true,
  "available_models": ["medical_qa"]
}
```

### Welcome Message

```
GET /
```

Returns a welcome message and basic information about the API.

**Response**:

```json
{
  "message": "Welcome to Elara AI Medical Assistant! 🏥🤖",
  "description": "Your multilingual AI companion for medical questions",
  "endpoints": {
    "health": "/health",
    "chat": "/chat",
    "docs": "/docs",
    "redoc": "/redoc"
  },
  "features": [
    "🧠 Medical Q&A with AI",
    "🌍 200+ languages supported",
    "🎙️ Voice input/output",
    "📊 Evidence-based responses",
    "🔒 Privacy-first design"
  ]
}
```

### Ask Medical Question

```
POST /chat/ask
```

The main endpoint for asking medical questions to the AI assistant.

**Request Body**:

```json
{
  "question": "What are the symptoms of diabetes?",
  "language": "en",
  "user_type": "patient",
  "conversation_id": null,
  "include_sources": true,
  "max_response_length": 500
}
```

**Parameters**:

- `question` (string, required): The medical question to ask
- `language` (string, optional): ISO language code (e.g., "en", "es", "fr"). If not provided, language will be auto-detected
- `user_type` (string, optional): Type of user asking the question, affects response style:
  - `patient`: Simplified language, more explanatory
  - `doctor`: Technical language, more concise
  - `medical_student`: Educational style with details
  - `researcher`: Research-focused with citations
  - `general_public`: Balanced general information
- `conversation_id` (string, optional): ID to continue a previous conversation
- `include_sources` (boolean, optional): Whether to include sources in response
- `max_response_length` (integer, optional): Maximum response length in words

**Response**:

```json
{
  "response": "Diabetes typically presents with symptoms like increased thirst (polydipsia), frequent urination (polyuria), unexplained weight loss, extreme hunger, blurry vision, fatigue, and slow-healing sores. Some patients may also experience tingling or numbness in the hands or feet. In Type 1 diabetes, symptoms often develop quickly, while in Type 2 diabetes, they may develop gradually or be mild enough to go unnoticed for years.",
  "confidence": 0.95,
  "sources": [
    {
      "title": "Diabetes Overview - Mayo Clinic",
      "url": "https://mayoclinic.org/diabetes",
      "type": "medical_website",
      "confidence": 0.98,
      "relevance_score": 0.92
    }
  ],
  "language_detected": "en",
  "processing_time": 1.25,
  "conversation_id": "conv_abc123",
  "safety_warning": "This information is for educational purposes only. Always consult a healthcare professional for personalized medical advice."
}
```

### Translate Text

```
POST /chat/translate
```

Translates text between languages.

**Request Body**:

```json
{
  "text": "What is diabetes?",
  "source_language": "en",
  "target_language": "es"
}
```

**Parameters**:

- `text` (string, required): Text to translate
- `source_language` (string, required): Source language code
- `target_language` (string, required): Target language code

**Response**:

```json
{
  "original_text": "What is diabetes?",
  "translated_text": "¿Qué es la diabetes?",
  "source_language": "en",
  "target_language": "es",
  "service": "Elara AI Translation"
}
```

### Get Conversation History

```
GET /chat/conversation/{conversation_id}
```

Retrieves the history of a conversation.

**Parameters**:

- `conversation_id` (string, path parameter): The ID of the conversation to retrieve

**Response**:

```json
{
  "conversation_id": "conv_abc123",
  "started_at": "2025-05-20T10:30:00Z",
  "last_updated": "2025-05-20T10:35:00Z",
  "messages": [
    {
      "type": "user",
      "content": "What is diabetes?",
      "timestamp": "2025-05-20T10:30:00Z",
      "language": "en"
    },
    {
      "type": "assistant",
      "content": "Diabetes is a condition...",
      "timestamp": "2025-05-20T10:30:05Z",
      "confidence": 0.95
    }
  ]
}
```

### Delete Conversation

```
DELETE /chat/conversation/{conversation_id}
```

Deletes a conversation and its history.

**Parameters**:

- `conversation_id` (string, path parameter): The ID of the conversation to delete

**Response**:

```json
{
  "message": "Conversation deleted successfully"
}
```

### Health Assessment

```
POST /chat/health-check
```

Analyzes symptoms and provides a health assessment.

**Request Body**:

```json
{
  "symptoms": ["headache", "fever", "nausea"],
  "age": 30,
  "gender": "male",
  "medical_history": ["hypertension"],
  "urgency_level": "normal"
}
```

**Parameters**:

- `symptoms` (array of strings, required): List of symptoms
- `age` (integer, optional): Patient age
- `gender` (string, optional): Patient gender
- `medical_history` (array of strings, optional): Relevant medical history
- `urgency_level` (string, optional): Perceived urgency level

**Response**:

```json
{
  "risk_level": "moderate",
  "possible_conditions": [
    {
      "name": "Common Cold",
      "probability": 0.7,
      "description": "Viral upper respiratory infection"
    },
    {
      "name": "Influenza",
      "probability": 0.5,
      "description": "Viral infection affecting respiratory system"
    }
  ],
  "recommendations": [
    "Rest and stay hydrated",
    "Monitor symptoms",
    "Consider seeing a doctor if symptoms worsen"
  ],
  "urgency": "routine",
  "disclaimer": {
    "warning": "This is NOT a medical diagnosis!",
    "advice": "Please consult a healthcare professional for proper medical evaluation.",
    "emergency": "If experiencing emergency symptoms, call emergency services immediately."
  },
  "confidence": 0.78
}
```

### Get Model Status

```
GET /chat/models/status
```

Returns the status of all AI models.

**Response**:

```json
{
  "models_loaded": true,
  "available_models": ["medical_qa", "translator", "language_detector"],
  "model_details": {
    "medical_qa": {
      "status": "loaded",
      "type": "neural_network",
      "memory_usage_mb": 4096.5,
      "capabilities": ["medical_reasoning", "question_answering", "health_information"]
    },
    "translator": {
      "status": "loaded",
      "type": "neural_network",
      "memory_usage_mb": 2048.3,
      "capabilities": ["translation", "multilingual_support"]
    }
  },
  "memory_usage": {
    "total_memory_mb": 6144.8,
    "memory_percent": 38.4,
    "available_memory_mb": 9855.2
  },
  "last_updated": "2025-05-20T15:30:45Z"
}
```

### Demo Response

```
GET /chat/demo
```

Returns a sample response for testing purposes.

**Response**:

```json
{
  "response": "Hello! I'm Elara AI, your medical assistant. This is a demo response. How can I help you with your health questions today?",
  "confidence": 1.0,
  "sources": ["Demo Mode"],
  "language_detected": "en",
  "processing_time": 0.001,
  "conversation_id": "demo-conversation",
  "safety_warning": "This is a demo response for testing purposes."
}
```

## Error Handling

When an error occurs, the API returns a JSON response with error details:

```json
{
  "error": "ValidationError",
  "message": "Invalid input provided",
  "details": {"field": "question", "issue": "cannot be empty"},
  "timestamp": "2025-05-20T10:30:00Z",
  "request_id": "req_abc123"
}
```

Common HTTP status codes:

- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Rate Limiting

The API implements rate limiting to prevent abuse. By default, this is set to 100 requests per minute per IP address. This can be configured in `config.py`.

## Example Usage with curl

### Ask a Medical Question

```bash
curl -X POST http://localhost:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of diabetes?",
    "user_type": "patient"
  }'
```

### Check System Health

```bash
curl http://localhost:8000/health
```

### Check Model Status

```bash
curl http://localhost:8000/chat/models/status
```
