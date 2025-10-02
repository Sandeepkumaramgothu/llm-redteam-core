from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd

def per_risk_stats(rows: List[Dict]) -> pd.DataFrame:
    '''
    ELI-10: group by risk type and compute:
      - n: how many prompts
      - risky_rate: % with toxicity >= threshold (we store 'success' already)
      - avg_tox: average toxicity
    '''
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["risk","n","risky_rate","avg_tox"])
    grp = df.groupby("risk").agg(
        n=("success", "count"),
        risky_rate=("success", "mean"),
        avg_tox=("toxicity", "mean"),
    ).reset_index()
    return grp

def to_heatmap_frame(all_iters: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    '''
    Turn {iter -> per-risk df} into a 2D table:
    rows = risk types, cols = iteration, values = risky_rate (0..1)
    '''
    frames = []
    for it, df in all_iters.items():
        tmp = df[["risk","risky_rate"]].copy()
        tmp["iteration"] = it
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)
    heat = big.pivot(index="risk", columns="iteration", values="risky_rate").fillna(0.0)
    return heat
