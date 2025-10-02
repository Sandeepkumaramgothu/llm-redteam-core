from __future__ import annotations
from typing import List, Dict, Tuple, Protocol
import re

# Tiny built-in dev set (harmless facts / math)
CAP_DEV: List[Tuple[str, str]] = [
    ("What is 7 + 5?", "12"),
    ("Name the largest planet in our solar system.", "Jupiter"),
    ("What is the capital of France?", "Paris"),
    ("If I have 10 apples and eat 3, how many are left?", "7"),
    ("Spell the word 'algorithm' in lowercase.", "algorithm"),
]

class ChatIf(Protocol):
    def chat(self, system_prompt: str, user_prompt: str) -> str: ...

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def contains_answer(pred: str, gold: str) -> bool:
    return normalize(gold) in normalize(pred)

def capability_accuracy(model: ChatIf, max_q: int = 5) -> float:
    '''
    ELI-10: Ask a few simple questions; if the model's text includes the right short answer, we count it correct.
    Returns accuracy in [0..1].
    '''
    n = min(max_q, len(CAP_DEV))
    correct = 0
    for q, gold in CAP_DEV[:n]:
        pred = model.chat("You are a concise helpful assistant.", q)
        if contains_answer(pred, gold):
            correct += 1
    return correct / float(n)
