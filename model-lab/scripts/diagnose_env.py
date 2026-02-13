import sys
import os

print(f"Python: {sys.version.split()[0]}")
print(f"Venv: {os.environ.get('VIRTUAL_ENV', 'NONE')}")

def check_import(name):
    try:
        mod = __import__(name)
        print(f"{name.capitalize()}: {mod.__version__}")
    except ImportError:
        print(f"{name.capitalize()}: MISSING")
    except Exception as e:
        print(f"{name.capitalize()}: ERROR ({e})")

try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    print(f"  MPS: {torch.backends.mps.is_available()}")
except ImportError:
    print("Torch: MISSING")

check_import("torchvision")
check_import("torchaudio")
check_import("transformers")
