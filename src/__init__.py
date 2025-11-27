"""
AdversaFlow Red-Teaming Framework
Multi-level adversarial flow analysis for LLM security
"""

__version__ = "1.0.0"
__author__ = "AdversaFlow Team"

# ✅ FIX #3: Lazy imports to avoid circular dependencies
__all__ = ["core", "models", "utils"]

def _lazy_import():
    """Lazy import to avoid circular dependencies."""
    import sys
    from pathlib import Path

    # Ensure project root is in path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import submodules
    try:
        from . import core
        print("✅ Imported: core")
    except ImportError as e:
        print(f"⚠️  Failed to import core: {e}")

    try:
        from . import models
        print("✅ Imported: models")
    except ImportError as e:
        print(f"⚠️  Failed to import models: {e}")

    try:
        from . import utils
        print("✅ Imported: utils")
    except ImportError as e:
        print(f"⚠️  Failed to import utils: {e}")

# Don't auto-import (let user import explicitly)
# _lazy_import()

print(f"AdversaFlow v{__version__} initialized")
