# Cell 32
"""
Template Strategy - Direct seed testing for baseline/no-attack.
"""
import random
from .base import AttackStrategyBase, is_refusal_or_empty

class TemplateStrategy(AttackStrategyBase):
    """
    Direct seed (no transformation), can use any target_model.
    Useful as a baseline, shows model's filter/rejection power.
    """
    def __init__(self, config, red_model=None, target_model=None):
        super().__init__(config, red_model, target_model)
        self.target_model = target_model
        print("✅ Template Strategy (direct testing)")

    def run(self, seed_text, risk_type, rng, debug=False):
        if debug:
            print(f"[TEMPLATE direct input]: {seed_text}")
        response = self.target_model.generate(seed_text)
        is_ref = is_refusal_or_empty(response)
        return {
            "adversarial": seed_text,
            "combined": seed_text,
            "response": response,
            "method": "direct",
            "red_model": "none",
            "turns_taken": 1,
            "conversation_history": [
                {
                    "turn": 0,
                    "prompt": seed_text,
                    "response": response,
                    "is_refusal": is_ref
                }
            ]
        }
