"""
Utilities Module
Scoring, metrics, visualization, and post-processing
"""

# ✅ FIX #3: Graceful imports
__all__ = []

try:
    from . import scoring
    __all__.append("scoring")
except ImportError as e:
    print(f"⚠️  Could not import scoring: {e}")
    scoring = None

try:
    from . import metrics
    __all__.append("metrics")
except ImportError as e:
    print(f"⚠️  Could not import metrics: {e}")
    metrics = None

try:
    from . import visualization
    __all__.append("visualization")
except ImportError as e:
    print(f"⚠️  Could not import visualization: {e}")
    visualization = None

try:
    from . import postrun
    __all__.append("postrun")
except ImportError as e:
    print(f"⚠️  Could not import postrun: {e}")
    postrun = None

# Summary
loaded_count = len(__all__)
if loaded_count == 4:
    print("✅ Utils module: All 4 utilities loaded")
elif loaded_count > 0:
    print(f"⚠️  Utils module: {loaded_count}/4 utilities loaded")
else:
    print("❌ Utils module: Nothing loaded")
