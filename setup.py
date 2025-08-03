#!/usr/bin/env python3
"""
Setup script for LISA改 (LISA-Kai) - Integration of Qwen2.5-VL-3B and SAM2.1
"""
import subprocess
import sys
import os


def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install Flash Attention 2 if GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            print("\nGPU detected. Installing Flash Attention 2...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-U", "flash-attn", "--no-build-isolation"
            ])
    except Exception as e:
        print(f"Warning: Could not install Flash Attention 2: {e}")
        print("Continuing without Flash Attention 2...")


def verify_installation():
    """Verify that all required packages are installed correctly"""
    print("\nVerifying installation...")
    
    try:
        import torch
        import transformers
        import sam2
        import peft
        
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ Transformers version: {transformers.__version__}")
        print(f"✓ SAM2 installed successfully")
        print(f"✓ PEFT version: {peft.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - running on CPU")
            
        # Test model availability
        from transformers import AutoTokenizer
        print("\nTesting model access...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        print("✓ Successfully accessed Qwen2.5-VL-3B-Instruct")
        
    except Exception as e:
        print(f"✗ Installation verification failed: {e}")
        return False
    
    return True


def create_project_structure():
    """Create necessary directories and __init__.py files"""
    directories = [
        "src",
        "src/models",
        "src/utils",
        "tests",
        "data",
        "checkpoints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src"):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write("")
    
    print("✓ Project structure created")


if __name__ == "__main__":
    print("Setting up LISA改 (LISA-Kai) project...\n")
    
    # Create project structure
    create_project_structure()
    
    # Install requirements
    install_requirements()
    
    # Verify installation
    if verify_installation():
        print("\n✅ Setup completed successfully!")
        print("\nYou can now start implementing the LISA改 model.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)