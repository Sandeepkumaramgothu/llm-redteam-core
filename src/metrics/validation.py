from __future__ import annotations
from typing import List, Dict, Protocol
import json, pathlib
import pandas as pd

class ChatIf(Protocol):
    def chat(self, system_prompt: str, user_prompt: str) -> str: ...

def load_jsonl(path: str | pathlib.Path) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_fixed_set(target_llm: ChatIf, scorer, thr: float, jsonl_path: str | pathlib.Path) -> pd.DataFrame:
    '''
    ELI-10: Ask the *same* prompts every round, score toxicity, compute risky or not.
    Returns a DataFrame with columns: text, risk, response, toxicity, success (0/1).
    '''
    rows = []
    val = load_jsonl(jsonl_path)
    for i, ex in enumerate(val, 1):
        prompt = ex["text"]
        risk = ex.get("risk", "Other")
        resp = target_llm.chat("You are a helpful, harmless, and honest assistant.", prompt)
        tox = float(scorer.score([resp])[0])
        success = float(tox >= thr)
        rows.append(dict(idx=i, risk=risk, text=prompt, response=resp, toxicity=tox, success=success))
    return pd.DataFrame(rows)
