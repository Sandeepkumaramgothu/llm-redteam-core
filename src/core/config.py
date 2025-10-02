from __future__ import annotations
import yaml
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
