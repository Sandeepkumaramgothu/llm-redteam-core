"""
Target Models Module
Wrappers for target LLMs (Llama, GPT, etc.)
"""

# ✅ FIX #3: Graceful imports with fallback
__all__ = []

try:
    from .target_model import TargetModel
    __all__.append("TargetModel")
except ImportError as e:
    print(f"⚠️  Could not import TargetModel: {e}")
    TargetModel = None

try:
    from .red_model import RedModel
    __all__.append("RedModel")
except ImportError as e:
    print(f"⚠️  Could not import RedModel: {e}")
    RedModel = None

# Summary
if TargetModel and RedModel:
    print("✅ Models module: Both TargetModel and RedModel loaded")
elif TargetModel:
    print("⚠️  Models module: Only TargetModel loaded (RedModel missing)")
elif RedModel:
    print("⚠️  Models module: Only RedModel loaded (TargetModel missing)")
else:
    print("❌ Models module: No models loaded")
