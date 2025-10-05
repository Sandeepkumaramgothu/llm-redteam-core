from __future__ import annotations
from typing import List, Dict
import torch
try:
    from detoxify import Detoxify
except Exception:
    Detoxify = None

class DetoxifyAux:
    def __init__(self, device: str|None=None):
        if Detoxify is None:
            self.model = None
            return
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Detoxify('original', device=device)

    def score_batch(self, texts: List[str]) -> List[Dict[str,float]]:
        """Returns dict of category probabilities per text (0..1)."""
        if self.model is None:
            return [dict() for _ in texts]
        out = self.model.predict(texts)
        # out is a dict of lists; zip them into per-text dicts
        keys = list(out.keys())
        rows = []
        n = len(out[keys[0]]) if keys else 0
        for i in range(n):
            rows.append({k: float(out[k][i]) for k in keys})
        return rows
