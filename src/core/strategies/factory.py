# Cell 27
"""
Strategy Factory - Creates appropriate attack strategy based on configuration
Supports flexible model (OpenAI/HF/local) for both red and target; CLI print; model sweep ready.
"""
from typing import Dict, Any
from .base import AttackStrategyBase

from .pyrit_crescendo import PyRITCrescendoStrategy
from .manual_crescendo import ManualCrescendoStrategy
from .openai_baseline import OpenAIBaselineStrategy
from .jailbreak import JailbreakStrategy
from .template import TemplateStrategy

class AttackStrategyFactory:
    """
    Factory for creating attack strategy instances.
    Always pass config, red_model, and target_model to enable flexible permuted evaluation.
    """
    STRATEGIES = {
        "pyrit_crescendo": PyRITCrescendoStrategy,
        "manual_crescendo": ManualCrescendoStrategy,
        "openai_baseline": OpenAIBaselineStrategy,
        "gpt4o-mini": OpenAIBaselineStrategy,   # Alias
        "jailbreak": JailbreakStrategy,
        "pyrit_jailbreak": JailbreakStrategy,   # Alias
        "template": TemplateStrategy,
        "direct": TemplateStrategy,             # Alias
    }
    @staticmethod
    def create(strategy_name: str, config: Dict[str, Any], red_model, target_model) -> AttackStrategyBase:
        """
        Create and return an attack strategy instance.
        Args:
            strategy_name: Name of strategy to create
            config: Configuration dict
            red_model: Red model wrapper (can be OpenAI, HF, or local)
            target_model: Target model wrapper (can be OpenAI, HF, or local)
        Returns:
            AttackStrategy instance
        """
        print(f"\n🏭 Factory creating strategy: {strategy_name}")
        strategy_class = AttackStrategyFactory.STRATEGIES.get(strategy_name)
        if strategy_class is None:
            print(f"⚠️ Unknown strategy '{strategy_name}', using ManualCrescendo")
            strategy_class = ManualCrescendoStrategy
        # CUSTOMIZATION: always pass both models (for permutation sweep flexibility)
        return strategy_class(config, red_model, target_model)

    @staticmethod
    def list_strategies() -> list:
        """Return list of available strategy names."""
        return list(AttackStrategyFactory.STRATEGIES.keys())

    @staticmethod
    def print_strategies():
        """Print formatted list of available strategies."""
        print("\n📋 Available Attack Strategies:")
        print("="*60)
        print("✅ OpenAI/HF/Local Supported:")
        print("   • pyrit_crescendo    - Multi-turn with PyRIT")
        print("   • manual_crescendo   - Multi-turn without PyRIT")
        print("   • openai_baseline    - Single turn baseline (OpenAI/HF)")
        print("\n✅ Template-Based:")
        print("   • jailbreak          - Multiple jailbreak templates")
        print("   • template           - Direct (no modification)")
        print("="*60)
