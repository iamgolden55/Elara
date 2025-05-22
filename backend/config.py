#!/usr/bin/env python3
"""
⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧
                            ELARA AI - CONFIGURATION
                        The Settings & Configuration! ⚙️
⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧⚙️🔧
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

class Settings:
    """
    🎛️ ELARA AI CONFIGURATION CENTER
    
    All settings and configuration options for the medical AI assistant.
    Think of this as the control panel for our AI restaurant!
    """
    
    def __init__(self):
        # 📁 Project paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models_files"
        self.backend_dir = self.project_root / "backend"
        
        # 🌐 API Configuration
        self.api_host = os.getenv("ELARA_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("ELARA_PORT", "8000"))
        self.api_workers = int(os.getenv("ELARA_WORKERS", "1"))
        self.debug_mode = os.getenv("ELARA_DEBUG", "true").lower() == "true"
        
        # 🤖 Model Configuration
        self.models_config = {
            "medical_qa": {
                "name": "medical-lora-diagpt",
                "path": self.models_dir / "medical_lora",
                "type": "causal_lm",
                "base_model": "microsoft/DialoGPT-medium",
                "max_length": 1024,
                "temperature": 0.7,
                "enabled": True
            },
            "translator": {
                "name": "bloom-1b7",
                "path": self.models_dir / "bloom", 
                "type": "translation",
                "max_length": 1024,
                "enabled": True
            },
            "language_detector": {
                "name": "language_detector",
                "type": "classification",
                "enabled": True
            },
            "symptom_analyzer": {
                "name": "symptom_analyzer",
                "type": "medical_reasoning",
                "enabled": True
            }
        }
        
        # Add Mistral model path for the LLM
        self.MISTRAL_MODEL_PATH = str(self.models_dir / "mistral")
        
        # 🧠 AI Generation Settings
        self.generation_config = {
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop_sequences": ["</response>", "[END]"]
        }
        
        # 🌍 Multilingual Settings
        self.languages_config = {
            "default_language": "en",
            "supported_languages": [
                "en",  # English
                "es",  # Spanish
                "fr",  # French
                "de",  # German
                "it",  # Italian
                "pt",  # Portuguese
                "nl",  # Dutch
                "ru",  # Russian
                "zh",  # Chinese
                "ja",  # Japanese
                "ko",  # Korean
                "ar",  # Arabic
                "hi",  # Hindi
                "sw",  # Swahili
                "yo",  # Yoruba
                "ha",  # Hausa
                "ig",  # Igbo
                "am",  # Amharic
                "zu",  # Zulu
                "xh",  # Xhosa
            ],
            "auto_detect_language": True,
            "translate_responses": True
        }
        
        # 🔒 Security Settings  
        self.security_config = {
            "cors_origins": [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:8000"
            ],
            "cors_methods": ["GET", "POST", "DELETE", "OPTIONS"],
            "cors_headers": ["*"],
            "rate_limit_requests": 100,  # Requests per minute
            "rate_limit_window": 60,     # Window in seconds
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "api_key_required": False,   # Set to True in production
            "log_user_queries": True,    # For analytics (anonymized)
        }
        
        # 🏥 Medical Settings
        self.medical_config = {
            "disclaimer_text": "This information is for educational purposes only and does not constitute medical advice. Please consult a healthcare professional for personalized medical guidance.",
            "emergency_keywords": [
                "emergency", "urgent", "chest pain", "difficulty breathing",
                "unconscious", "overdose", "suicidal", "heart attack", "stroke"
            ],
            "auto_add_disclaimers": True,
            "require_age_verification": False,
            "max_conversation_length": 20,  # Messages per conversation
            "enable_symptom_checker": True,
            "enable_drug_interactions": False,  # Disabled for legal safety
            "enable_diagnosis_suggestions": True
        }
        
        # 📊 Monitoring & Logging
        self.monitoring_config = {
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_file": self.project_root / "logs" / "elara.log",
            "enable_performance_monitoring": True,
            "track_user_sessions": True,
            "anonymous_analytics": True,
            "health_check_interval": 60,  # Seconds
            "memory_usage_threshold": 80,  # Percentage
            "response_time_threshold": 5.0  # Seconds
        }
        
        # 🗄️ Database Configuration (for future use)
        self.database_config = {
            "type": "sqlite",  # sqlite, postgresql, mysql
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "name": os.getenv("DB_NAME", "elara_ai"),
            "user": os.getenv("DB_USER", "elara"),
            "password": os.getenv("DB_PASSWORD", ""),
            "pool_size": 5,
            "max_overflow": 10
        }
        
        # 📈 Analytics Configuration
        self.analytics_config = {
            "enable_usage_tracking": True,
            "track_response_quality": True,
            "track_user_satisfaction": True,
            "export_analytics": True,
            "analytics_retention_days": 90
        }
        
        # 🔧 Development Settings
        self.dev_config = {
            "auto_reload": self.debug_mode,
            "show_detailed_errors": self.debug_mode,
            "enable_debug_routes": self.debug_mode,
            "simulate_models": os.getenv("SIMULATE_MODELS", "false").lower() == "true",
            "mock_external_apis": self.debug_mode
        }
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """📁 Create necessary directories if they don't exist"""
        directories = [
            self.data_dir,
            self.models_dir,
            self.project_root / "logs",
            self.data_dir / "training_data",
            self.data_dir / "processed",
            self.data_dir / "raw",
            self.models_dir / "mistral",
            self.models_dir / "bloom"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """🤖 Get configuration for a specific model"""
        return self.models_config.get(model_name, {})
    
    def is_model_enabled(self, model_name: str) -> bool:
        """✅ Check if a model is enabled"""
        return self.models_config.get(model_name, {}).get("enabled", False)
    
    def get_supported_languages(self) -> List[str]:
        """🌍 Get list of supported languages"""
        return self.languages_config["supported_languages"]
    
    def is_language_supported(self, language_code: str) -> bool:
        """🔍 Check if a language is supported"""
        return language_code in self.languages_config["supported_languages"]
    
    def get_cors_config(self) -> Dict[str, Any]:
        """🌐 Get CORS configuration for FastAPI"""
        return {
            "allow_origins": self.security_config["cors_origins"],
            "allow_methods": self.security_config["cors_methods"],
            "allow_headers": self.security_config["cors_headers"],
            "allow_credentials": True
        }
    
    def get_rate_limit_config(self) -> Dict[str, int]:
        """⏱️ Get rate limiting configuration"""
        return {
            "requests": self.security_config["rate_limit_requests"],
            "window": self.security_config["rate_limit_window"]
        }
    
    def should_log_query(self, user_id: Optional[str] = None) -> bool:
        """📝 Determine if a query should be logged"""
        # Implement privacy-respecting logging logic
        return self.security_config["log_user_queries"] and self.analytics_config["enable_usage_tracking"]
    
    def get_emergency_response(self) -> Dict[str, str]:
        """🚨 Get emergency response configuration"""
        return {
            "message": "If you are experiencing a medical emergency, please contact emergency services immediately (911 in the US, 112 in Europe, or your local emergency number).",
            "disclaimer": "This AI assistant cannot replace emergency medical services.",
            "emergency_numbers": {
                "US": "911",
                "EU": "112", 
                "UK": "999",
                "Canada": "911",
                "Australia": "000"
            }
        }
    
    def get_medical_disclaimer(self) -> str:
        """⚠️ Get medical disclaimer text"""
        return self.medical_config["disclaimer_text"]
    
    def is_emergency_query(self, query: str) -> bool:
        """🚨 Check if query contains emergency keywords"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.medical_config["emergency_keywords"])
    
    def get_generation_settings(self) -> Dict[str, Any]:
        """🧠 Get AI text generation settings"""
        return self.generation_config.copy()
    
    def get_log_config(self) -> Dict[str, Any]:
        """📝 Get logging configuration"""
        return {
            "level": self.monitoring_config["log_level"],
            "file": str(self.monitoring_config["log_file"]),
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "rotation": "500 MB",
            "retention": "30 days"
        }
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """⚡ Get performance monitoring thresholds"""
        return {
            "memory_usage_percent": self.monitoring_config["memory_usage_threshold"],
            "response_time_seconds": self.monitoring_config["response_time_threshold"]
        }
    
    def export_config(self) -> Dict[str, Any]:
        """📤 Export complete configuration as dictionary"""
        return {
            "api": {
                "host": self.api_host,
                "port": self.api_port,
                "workers": self.api_workers,
                "debug": self.debug_mode
            },
            "models": self.models_config,
            "generation": self.generation_config,
            "languages": self.languages_config,
            "security": self.security_config,
            "medical": self.medical_config,
            "monitoring": self.monitoring_config,
            "analytics": self.analytics_config,
            "development": self.dev_config
        }
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """🔧 Update a configuration value"""
        try:
            section_dict = getattr(self, f"{section}_config", None)
            if section_dict is not None:
                section_dict[key] = value
                return True
            return False
        except Exception:
            return False
    
    def validate_config(self) -> List[str]:
        """✅ Validate configuration and return any issues"""
        issues = []
        
        # Check if required directories exist
        if not self.data_dir.exists():
            issues.append(f"Data directory not found: {self.data_dir}")
        
        if not self.models_dir.exists():
            issues.append(f"Models directory not found: {self.models_dir}")
        
        # Check port availability
        if not (1024 <= self.api_port <= 65535):
            issues.append(f"Invalid API port: {self.api_port}")
        
        # Check language configuration
        if not self.languages_config["default_language"] in self.languages_config["supported_languages"]:
            issues.append("Default language not in supported languages list")
        
        # Check memory thresholds
        if not (0 <= self.monitoring_config["memory_usage_threshold"] <= 100):
            issues.append("Invalid memory usage threshold")
        
        return issues

# Global settings instance
settings = Settings()

# Environment-specific configurations
class DevelopmentSettings(Settings):
    """🛠️ Development-specific settings"""
    
    def __init__(self):
        super().__init__()
        self.debug_mode = True
        self.dev_config["auto_reload"] = True
        self.dev_config["show_detailed_errors"] = True
        self.dev_config["simulate_models"] = True
        self.security_config["log_user_queries"] = True

class ProductionSettings(Settings):
    """🏭 Production-specific settings"""
    
    def __init__(self):
        super().__init__()
        self.debug_mode = False
        self.dev_config["auto_reload"] = False
        self.dev_config["show_detailed_errors"] = False
        self.dev_config["simulate_models"] = False
        self.security_config["api_key_required"] = True
        self.security_config["rate_limit_requests"] = 60  # Stricter rate limiting

class TestingSettings(Settings):
    """🧪 Testing-specific settings"""
    
    def __init__(self):
        super().__init__()
        self.debug_mode = True
        self.dev_config["simulate_models"] = True
        self.security_config["log_user_queries"] = False
        self.analytics_config["enable_usage_tracking"] = False

# Factory function to get appropriate settings
def get_settings(environment: str = "development") -> Settings:
    """🏭 Get settings based on environment"""
    
    environment = environment.lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# Configuration validation on import
if __name__ == "__main__":
    # Validate configuration when run directly
    config = Settings()
    issues = config.validate_config()
    
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration validated successfully!")
    
    # Display summary
    print("\n🎛️ ELARA AI CONFIGURATION SUMMARY:")
    print(f"   🌐 API: {config.api_host}:{config.api_port}")
    print(f"   🤖 Models: {len(config.models_config)} configured")
    print(f"   🌍 Languages: {len(config.get_supported_languages())} supported")
    print(f"   🔒 Debug Mode: {'ON' if config.debug_mode else 'OFF'}")
    print(f"   📁 Data Directory: {config.data_dir}")
    print(f"   🎯 Ready to launch Elara AI!")
