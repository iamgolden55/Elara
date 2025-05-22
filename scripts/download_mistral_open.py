#!/usr/bin/env python3
"""
Mistral 7B Alternative Model Downloader for Elara AI
Downloads open-source Mistral-style models for medical applications
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm

class MistralAlternativeDownloader:
    def __init__(self):
        self.model_dir = Path("/Users/new/elara_main/models_files/mistral")
        # Using an open alternative: NousResearch's Hermes 2 Pro (Mistral 7B based)
        self.model_name = "NousResearch/Hermes-2-Pro-Mistral-7B"
        
    def check_space(self):
        """Check if we have enough disk space (≈ 14GB)"""
        import shutil
        free_space = shutil.disk_usage(self.model_dir.parent).free
        required_space = 15 * 1024**3  # 15GB to be safe
        
        print(f"🔍 Checking disk space...")
        print(f"   Available: {free_space / (1024**3):.1f} GB")
        print(f"   Required:  {required_space / (1024**3):.1f} GB")
        
        if free_space < required_space:
            print(f"❌ Not enough space! Need at least 15GB free.")
            return False
        
        print(f"✅ Plenty of space available!")
        return True
    
    def check_existing(self):
        """Check if model already exists"""
        config_file = self.model_dir / "config.json"
        if config_file.exists():
            print(f"🎯 Mistral model already exists at {self.model_dir}")
            return True
        return False
    
    def download_model(self):
        """Download Mistral alternative with progress tracking"""
        print(f"🚀 Starting Hermes-2-Pro-Mistral-7B download...")
        print(f"📁 Saving to: {self.model_dir}")
        print(f"🌐 Source: {self.model_name}")
        print(f"✨ This is a Mistral 7B fine-tuned for instruction following!")
        
        try:
            # Create directory if it doesn't exist
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download the complete model repository
            print(f"\n⬇️  Downloading Hermes-2-Pro-Mistral-7B...")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=str(self.model_dir),
                resume_download=True,  # Resume if interrupted
            )
            
            print(f"\n✅ SUCCESS! Mistral model downloaded successfully!")
            print(f"📊 Model size: {self.get_model_size():.1f} GB")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            return False
    
    def get_model_size(self):
        """Calculate total model size in GB"""
        total_size = 0
        for file_path in self.model_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024**3)
    
    def verify_model(self):
        """Verify the downloaded model is complete"""
        required_files = [
            "config.json",
            "tokenizer_config.json", 
            "tokenizer.json",
        ]
        
        print(f"\n🔍 Verifying Mistral model download...")
        missing_files = []
        
        for file_name in required_files:
            file_path = self.model_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        print(f"✅ All required files present!")
        
        # Test if we can load the tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            print(f"✅ Tokenizer loads successfully!")
            print(f"📝 Vocab size: {tokenizer.vocab_size:,}")
            return True
        except Exception as e:
            print(f"❌ Tokenizer test failed: {e}")
            return False

def main():
    print("🔥 MISTRAL MODEL DOWNLOADER FOR ELARA AI 🔥")
    print("🎯 Using Hermes-2-Pro-Mistral-7B (fully open source!)")
    print("=" * 60)
    
    downloader = MistralAlternativeDownloader()
    
    # Check if already downloaded
    if downloader.check_existing():
        choice = input("Model exists. Re-download? (y/N): ").lower()
        if choice != 'y':
            print("Using existing model. Verifying...")
            if downloader.verify_model():
                print("🎉 Mistral model is ready to use!")
                return
    
    # Check disk space
    if not downloader.check_space():
        return
    
    # Download model
    print(f"\n🚨 This will download ~14GB. Continue? (Y/n): ", end="")
    choice = input().lower()
    if choice == 'n':
        print("Download cancelled.")
        return
    
    if downloader.download_model():
        if downloader.verify_model():
            print(f"\n🎉 MISTRAL MODEL READY FOR ELARA AI! 🎉")
            print(f"🔗 Model location: {downloader.model_dir}")
            print(f"✨ Hermes-2-Pro is excellent for medical reasoning!")
        else:
            print(f"❌ Verification failed. Please try re-downloading.")
    else:
        print(f"❌ Download failed. Check your internet connection.")

if __name__ == "__main__":
    main()
