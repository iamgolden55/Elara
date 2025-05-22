#!/usr/bin/env python3
"""
🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖
                    ELARA AI - LORA INTEGRATION TESTER
                Test if the medical LoRA adapter is working!
🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖🧪🤖
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path (absolute path to ensure we find it)
project_root = Path('/Users/new/elara_main')
if project_root.exists():
    print(f"✅ Project root found: {project_root}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ Added to Python path: {project_root}")
else:
    print(f"❌ Project root not found: {project_root}")
    
# Print Python path for debugging
print("🔍 Python path:")
for p in sys.path:
    print(f"  - {p}")

from backend.config import Settings
from backend.models import ModelManager

async def test_lora_response():
    """Test the LoRA model response"""
    print("🔍 TESTING ELARA AI LORA INTEGRATION")
    print("=" * 50)
    
    # Initialize settings and model manager
    settings = Settings()
    model_manager = ModelManager(settings)
    
    # Initialize models
    print("🔄 Initializing models...")
    await model_manager.initialize_models()
    
    # Check if models are loaded
    if model_manager.are_models_loaded():
        print("✅ Models loaded successfully!")
        print(f"📊 Available models: {model_manager.get_available_models()}")
        
        # Get model details
        model_details = model_manager.get_model_details()
        print("\n🤖 MODEL DETAILS:")
        for model_name, details in model_details.items():
            print(f"📋 {model_name}: {details['status']} ({details['type']})")
            print(f"   Memory usage: {details.get('memory_usage_mb', 'N/A')} MB")
            print(f"   Capabilities: {', '.join(details.get('capabilities', []))}")
        
        # Test a medical question
        test_questions = [
            "What are the symptoms of diabetes?",
            "How do I treat a sprained ankle?",
            "What are the risk factors for heart disease?", 
            "What's the difference between a cold and the flu?"
        ]
        
        for question in test_questions:
            print(f"\n🔍 TESTING QUESTION: \"{question}\"")
            
            # Get medical context
            context = await model_manager.retrieve_medical_context(question)
            
            # Generate response using medical LoRA
            response_data = await model_manager.generate_medical_response(
                question=question,
                context=context,
                user_type="patient"
            )
            
            print("\n🤖 MODEL RESPONSE:")
            print("-" * 50)
            print(response_data["response"])
            print("-" * 50)
            
            # Brief pause between questions
            await asyncio.sleep(1)
        
        # Clean up
        print("\n🧹 Cleaning up models...")
        await model_manager.cleanup()
        
        print("\n✅ TEST COMPLETE! If you see medical responses above, your LoRA is working!")
        return True
    else:
        print("❌ Models were not loaded correctly.")
        print("Check if the LoRA adapter exists at: /Users/new/elara_main/models_files/medical_lora")
        return False

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_lora_response())
