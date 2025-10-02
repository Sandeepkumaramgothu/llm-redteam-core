from __future__ import annotations
from typing import List
import sacrebleu

def self_bleu(texts: List[str]) -> float:
    '''
    Self-BLEU (ELI-10): compare each text to all the others.
    High = very similar (bad diversity). Low = varied (good).
    Returns BLEU in [0..100].
    '''
    texts = [t.strip() for t in texts if t and t.strip()]
    if len(texts) < 2:
        return 0.0
    scores = []
    for i, pred in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        # sacrebleu wants list-of-refs, each itself a list
        bleu = sacrebleu.corpus_bleu([pred], [refs]).score
        scores.append(bleu)
    return float(sum(scores) / len(scores))
