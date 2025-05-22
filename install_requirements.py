#!/usr/bin/env python3
"""
📦✨ Elara AI - Installation Script

This script installs all required packages for the Elara AI medical assistant,
including the new features: LoRA training and RAG system.

It checks for existing installations and only installs missing packages.
"""

import sys
import subprocess
import pkg_resources
import os
from pathlib import Path

# Required packages for different components
BASE_REQUIREMENTS = [
    "fastapi>=0.95.0",
    "uvicorn>=0.20.0",
    "pydantic>=2.0.0",
    "transformers>=4.28.0",
    "torch>=2.0.0"
]

LORA_REQUIREMENTS = [
    "peft>=0.4.0",
    "datasets>=2.10.0",
    "accelerate>=0.18.0",
    "wandb>=0.15.0"  # Optional for tracking
]

RAG_REQUIREMENTS = [
    "sentence-transformers>=2.2.2",
    "faiss-cpu>=1.7.4",  # Use faiss-gpu for GPU support
    "tqdm>=4.65.0"
]

EXTRAS = [
    "openai-whisper>=20231101",  # For voice input
    "langchain>=0.0.200",        # Optional for advanced agents
    "matplotlib>=3.7.1",         # For visualizations
    "scikit-learn>=1.2.2"        # For evaluation metrics
]

ALL_REQUIREMENTS = BASE_REQUIREMENTS + LORA_REQUIREMENTS + RAG_REQUIREMENTS


def get_installed_packages():
    """Get a list of installed packages and their versions"""
    installed = {}
    for pkg in pkg_resources.working_set:
        installed[pkg.key] = pkg.version
    return installed


def check_package(package, installed_packages):
    """Check if a package is installed and meets version requirements"""
    package_name, version_spec = package.split('>=') if '>=' in package else (package, '')
    
    # Normalize package name for comparison
    package_name = package_name.lower().replace('-', '_')
    
    if package_name in installed_packages:
        if not version_spec:
            return True
            
        installed_version = installed_packages[package_name]
        required_version = version_spec
        
        try:
            # Simple version comparison (not full semver)
            if installed_version >= required_version:
                return True
            else:
                return False
        except Exception:
            # If version comparison fails, assume it's good enough
            return True
    
    return False


def install_packages(packages, upgrade=False, verbose=True):
    """Install a list of packages using pip"""
    if not packages:
        if verbose:
            print("No packages to install.")
        return True
    
    # Construct pip command
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if upgrade:
        cmd.append("--upgrade")
    
    cmd.extend(packages)
    
    if verbose:
        pkg_str = ", ".join(packages)
        print(f"Installing packages: {pkg_str}")
        print(f"Command: {' '.join(cmd)}")
    
    try:
        # Execute pip install
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if verbose:
            print(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False


def main():
    """Main function to install packages"""
    print("\n📦✨ ELARA AI - PACKAGE INSTALLATION")
    print("=" * 50)
    
    # Get installed packages
    installed_packages = get_installed_packages()
    
    # Check which packages need to be installed
    missing_base = [pkg for pkg in BASE_REQUIREMENTS if not check_package(pkg, installed_packages)]
    missing_lora = [pkg for pkg in LORA_REQUIREMENTS if not check_package(pkg, installed_packages)]
    missing_rag = [pkg for pkg in RAG_REQUIREMENTS if not check_package(pkg, installed_packages)]
    missing_extras = [pkg for pkg in EXTRAS if not check_package(pkg, installed_packages)]
    
    # Print summary
    print("\n📋 Installation Status:")
    print(f"  - Base Requirements: {len(BASE_REQUIREMENTS) - len(missing_base)}/{len(BASE_REQUIREMENTS)} installed")
    print(f"  - LoRA Requirements: {len(LORA_REQUIREMENTS) - len(missing_lora)}/{len(LORA_REQUIREMENTS)} installed")
    print(f"  - RAG Requirements: {len(RAG_REQUIREMENTS) - len(missing_rag)}/{len(RAG_REQUIREMENTS)} installed")
    print(f"  - Extra Packages: {len(EXTRAS) - len(missing_extras)}/{len(EXTRAS)} installed")
    
    # Check if running in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("\n⚠️  Warning: Not running in a virtual environment!")
        print("   It's recommended to install packages in a virtual environment.")
        proceed = input("Do you want to continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Ask which components to install
    print("\n🔍 Select packages to install:")
    print("  1) Base requirements only")
    print("  2) Base + LoRA (for model training)")
    print("  3) Base + RAG (for retrieval)")
    print("  4) All requirements (recommended)")
    print("  5) Extras only")
    print("  6) Custom selection")
    print("  0) Exit without installing")
    
    choice = input("\nEnter your choice (0-6): ")
    
    to_install = []
    
    if choice == '0':
        print("Installation cancelled.")
        return
    elif choice == '1':
        to_install = missing_base
    elif choice == '2':
        to_install = missing_base + missing_lora
    elif choice == '3':
        to_install = missing_base + missing_rag
    elif choice == '4':
        to_install = missing_base + missing_lora + missing_rag
    elif choice == '5':
        to_install = missing_extras
    elif choice == '6':
        # Custom selection
        print("\n📋 Available packages:")
        
        all_missing = missing_base + missing_lora + missing_rag + missing_extras
        for i, pkg in enumerate(all_missing):
            print(f"  {i+1}) {pkg}")
        
        selected = input("\nEnter package numbers to install (comma-separated): ")
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in selected.split(',')]
            to_install = [all_missing[idx] for idx in selected_indices if 0 <= idx < len(all_missing)]
        except ValueError:
            print("Invalid selection. Please enter comma-separated numbers.")
            return
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Confirm installation
    if to_install:
        print(f"\n🔍 Ready to install {len(to_install)} packages:")
        for pkg in to_install:
            print(f"  - {pkg}")
        
        confirm = input("\nProceed with installation? (y/n): ")
        if confirm.lower() != 'y':
            print("Installation cancelled.")
            return
        
        # Install the packages
        print("\n📥 Installing packages...")
        success = install_packages(to_install)
        
        if success:
            print("\n✅ Installation completed successfully!")
        else:
            print("\n❌ Installation completed with errors.")
    else:
        print("\n✅ All selected packages are already installed!")
    
    # Final instructions
    print("\n🚀 Next steps:")
    print("  1. Use generate_training_data.py to create training examples")
    print("  2. Train a LoRA model with train_medical_lora.py")
    print("  3. Build the RAG index with build_medical_rag.py")
    print("  4. Run the Elara AI backend with uvicorn")


if __name__ == "__main__":
    main()
