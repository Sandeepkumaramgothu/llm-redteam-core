\
from __future__ import annotations
from typing import List, Dict, Any
import json, random, pathlib

# NOTE: Harmless placeholders for study. Replace with your dataset later (JSONL).
# Each item has a "text" and a "risk" label.
_BUILTIN = [
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Weapons]", "risk": "Weapons"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Crime]", "risk": "Crime"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Violence]", "risk": "Violence"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Drugs]", "risk": "Drugs"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Hate]", "risk": "Hate"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Harassment]", "risk": "Harassment"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Privacy]", "risk": "Privacy"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Extremism]", "risk": "Extremism"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Financial]", "risk": "Financial"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Self-harm]", "risk": "Self-harm"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=Sexual]", "risk": "Sexual"},
  {"text": "[SEED_PLACEHOLDER about risky idea: type=InfoHazards]", "risk": "InfoHazards"},
  {"text": "[SEED_PLACEHOLDER generic risky topic]", "risk": "Other"},
]

def load_seeds(cfg_data: dict, k: int, rng: random.Random) -> List[Dict[str, Any]]:
    src = cfg_data.get("seed_source","builtin")
    if src == "builtin":
        pool = _BUILTIN.copy()
    else:
        path = cfg_data.get("local_seed_path","")
        p = pathlib.Path(path)
        assert p.exists(), f"local_seed_path not found: {path}"
        pool = []
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                pool.append({"text": row["text"], "risk": row.get("risk","Other")})
    rng.shuffle(pool)
    return pool[:k]
