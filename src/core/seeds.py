
from __future__ import annotations
import json, pathlib, random
from typing import List, Dict, Any

def load_all(seeds_file: str) -> list[dict]:
 p = pathlib.Path(seeds_file)
 assert p.exists(), f"Seeds file not found: {p} — put your Do-Not-Answer JSONL there."
 rows = []
 with p.open("r", encoding="utf-8") as f:
     for li, line in enumerate(f, start=1):
         line=line.strip()
         if not line: continue
         obj = json.loads(line)
         seed_id = (obj.get("seed_id") or obj.get("id") or obj.get("idx") or
                    obj.get("uid") or li)
         seed = obj.get("seed") or obj.get("prompt") or obj.get("text") or ""
         risk = obj.get("risk") or obj.get("category") or obj.get("tag") or "unknown"
         rows.append({"seed_id": seed_id, "seed": seed, "risk": risk})
 return rows

def sample_seeds(all_seeds: list[dict], k: int, rng: random.Random) -> list[dict]:
 if k <= len(all_seeds):
     return rng.sample(all_seeds, k)
 return [rng.choice(all_seeds) for _ in range(k)]
