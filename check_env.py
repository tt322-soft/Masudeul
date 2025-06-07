import os
import torch
import sys

def check_environment():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
    
    print("\nChecking LoRA model:")
    lora_path = "lora_model"
    if os.path.exists(lora_path):
        print(f"LoRA path exists: {os.path.abspath(lora_path)}")
        print("Contents:", os.listdir(lora_path))
    else:
        print(f"LoRA path not found: {os.path.abspath(lora_path)}")
    
    print("\nCurrent working directory:", os.getcwd())
    print("Python path:", sys.path)

if __name__ == "__main__":
    check_environment()