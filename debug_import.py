import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

print("Checking imports...", flush=True)

try:
    print("Importing transformers...", flush=True)
    import transformers
    print(f"Transformers version: {transformers.__version__}", flush=True)
except Exception as e:
    print(f"Failed to import transformers: {e}", flush=True)

try:
    print("Importing peft...", flush=True)
    import peft
    print(f"Peft version: {peft.__version__}", flush=True)
except Exception as e:
    print(f"Failed to import peft: {e}", flush=True)

try:
    print("Importing torch...", flush=True)
    import torch
    print(f"Torch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"Failed to import torch: {e}", flush=True)

try:
    print("Importing src.core.engine...", flush=True)
    from src.core import engine
    print(f"Engine imported: {engine}", flush=True)
except Exception as e:
    print(f"Failed to import engine: {e}", flush=True)
    import traceback
    traceback.print_exc()
