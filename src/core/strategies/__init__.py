"""
Attack Strategies Module
Contains all red-teaming attack strategies

Available Strategies:
- PyRITCrescendoStrategy: Multi-turn escalation with PyRIT
- ManualCrescendoStrategy: Crescendo fallback without PyRIT
- OpenAIBaselineStrategy: Single-turn OpenAI/HF baseline
- JailbreakStrategy: Template-based jailbreak prompts
- TemplateStrategy: Direct seed testing
"""

# ✅ FIX #3: Import factory first, then strategies
__all__ = []

# Import factory (depends only on base)
try:
    from .base import AttackStrategyBase
    __all__.append("AttackStrategyBase")
except ImportError as e:
    print(f"⚠️  Could not import AttackStrategyBase: {e}")
    AttackStrategyBase = None

# Import individual strategies
strategies_to_import = [
    ("PyRITCrescendoStrategy", "pyrit_crescendo"),
    ("ManualCrescendoStrategy", "manual_crescendo"),
    ("OpenAIBaselineStrategy", "openai_baseline"),
    ("JailbreakStrategy", "jailbreak"),
    ("TemplateStrategy", "template"),
]

for strategy_name, module_name in strategies_to_import:
    try:
        module = __import__(f"src.core.strategies.{module_name}",
                            fromlist=[strategy_name])
        globals()[strategy_name] = getattr(module, strategy_name)
        __all__.append(strategy_name)
    except (ImportError, AttributeError) as e:
        print(f"⚠️  Could not import {strategy_name}: {e}")

# Import factory last (it needs the strategies)
try:
    from .factory import AttackStrategyFactory
    __all__.append("AttackStrategyFactory")
except ImportError as e:
    print(f"⚠️  Could not import AttackStrategyFactory: {e}")
    AttackStrategyFactory = None

# Summary
if AttackStrategyFactory:
    print(f"✅ Strategies module loaded: {len(__all__)-1} strategies + factory")
else:
    print(f"⚠️  Strategies module partially loaded: {len(__all__)} components")
