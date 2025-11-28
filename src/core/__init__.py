"""
Core attack logic module
Handles seeds, strategies, and attack engine
"""

# ✅ FIX #3: Import in correct order to avoid circular dependencies
__all__ = []

try:
    from . import seeds
    __all__.append("seeds")
except ImportError as e:
    print(f"⚠️  Could not import seeds: {e}")
    seeds = None

try:
    from . import strategies
    __all__.append("strategies")
except ImportError as e:
    print(f"⚠️  Could not import strategies: {e}")
    strategies = None

print("Attempting to import engine...", flush=True)
try:
    from . import engine
    __all__.append("engine")
    print("Successfully imported engine module", flush=True)
except ImportError as e:
    print(f"⚠️  Could not import engine: {e}", flush=True)
    with open("import_error.log", "w") as f:
        f.write(f"ImportError: {e}\n")
        import traceback
        traceback.print_exc(file=f)
    engine = None

# Summary
loaded = [name for name in ["seeds", "strategies", "engine"]
          if name in __all__]
if loaded:
    print(f"✅ Core module loaded: {', '.join(loaded)}")
else:
    print("❌ Core module: Nothing loaded")

# ✅ FIX #3: Add few_shot_loader to core module
try:
    from . import few_shot_loader
    __all__.append("few_shot_loader")
    print("✅ Core module: few_shot_loader loaded")
except ImportError as e:
    print(f"⚠️  Could not import few_shot_loader: {e}")
