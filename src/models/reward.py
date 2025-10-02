\
from __future__ import annotations
from typing import List
from detoxify import Detoxify
import torch

class ToxicityReward:
    def __init__(self):
        self.model = Detoxify('original')

    def score(self, texts: List[str]) -> List[float]:
        with torch.no_grad():
            preds = self.model.predict(texts)
        return [float(x) for x in preds.get("toxicity", [])]
