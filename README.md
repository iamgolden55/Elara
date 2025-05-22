# Elara AI: Medical AI Assistant

Elara AI is a modular, multilingual medical AI assistant designed to provide accurate medical information in multiple languages, leveraging fine-tuned language models.

## 🌟 Features

- 🧠 Medical question answering with domain-specific LoRA models
- 🌍 Support for 200+ languages through translation models
- 🔍 Retrieval Augmented Generation (RAG) for up-to-date medical information
- 🧬 User-specific responses (patient, doctor, researcher, etc.)
- 🚀 Modular architecture for easy extension and deployment

## 📋 Requirements

- Python 3.10+ 
- PyTorch
- FastAPI
- Transformers & PEFT libraries
- Node.js 16+ (for frontend)
- Docker & Docker Compose (optional, for containerized deployment)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/elara-ai.git
cd elara-ai
```

### 2. Download model files

**Note:** Model files are not included in this repository due to their size.

Download the required model files:
- Download the medical LoRA adapter from [here](#) and place in `models_files/medical_lora/`
- Or use the training scripts to create your own models

### 3. Setup environment

```bash
# Backend setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup (optional)
cd frontend
npm install
cd ..
```

### 4. Run the backend server

```bash
cd backend
python main.py
```

The server will start at http://localhost:8000. API documentation is available at http://localhost:8000/docs.

### 5. Run the frontend (optional)

```bash
cd frontend
npm start
```

The frontend will be available at http://localhost:3000.

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Or run specific services
docker-compose up backend
```

## 🧠 Model Architecture

Elara AI uses multiple models:

1. **Medical Reasoning**: Fine-tuned Mistral 7B or DialoGPT-medium with LoRA adapters for medical Q&A
2. **Translation**: BLOOM 1.7B for multilingual support
3. **RAG**: FAISS vector database with medical text embeddings for retrieval

## 🛠️ Development

### Adding new LoRA adapters

To create a new specialized model adapter (e.g., for pediatrics):

1. Prepare training data in `training_data/pediatrics/`
2. Configure training parameters in `training_configs/pediatrics_config.json`
3. Run the training script:
   ```bash
   python scripts/train_lora.py --config training_configs/pediatrics_config.json
   ```
4. The new adapter will be saved to `models_files/pediatrics_lora/`

### API Endpoints

The main endpoints are:

- `POST /chat/ask` - Send a medical question and get an AI response
- `GET /health` - Check if the system is running
- `GET /chat/models/status` - Get status of loaded models

See the [API documentation](docs/api.md) for more details.

## 📚 Documentation

- [API Reference](docs/api.md)
- [Backend-Model Communication](docs/backend_model_communication.md)
- [RAG System](docs/rag_system.md)
- [Model Training Guide](docs/model_training.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

Elara AI is provided for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
