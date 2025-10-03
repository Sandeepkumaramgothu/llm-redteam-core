from __future__ import annotations
# ELI5: Load all seeds from a JSONL file and pick a random set for this round.
# We always return the seed text *and* its seed_id (to help debugging).

import json, hashlib, random, pathlib
from typing import List, Dict

def _hash_id(text: str) -> str:
    # ELI5: Make a short stable id from text (if dataset did not provide one)
    import hashlib
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]

def load_all(seeds_file: str) -> List[Dict]:
    rows: List[Dict] = []
    p = pathlib.Path(seeds_file)
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                seed_txt = (r.get('seed') or '').strip()
                if not seed_txt:
                    continue
                seed_id = r.get('seed_id') or (r.get('meta') or {}).get('id') or _hash_id(seed_txt)
                risk = r.get('risk', 'Unknown')
                rows.append({'seed_id': str(seed_id), 'seed': seed_txt, 'risk': risk})
            except:
                pass
    return rows

def sample(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    # ELI5: Pick n different seeds at random (no duplicates)
    if len(rows) < n:
        raise ValueError(f'Not enough seeds: have {len(rows)}, need {n}')
    return rng.sample(rows, n)
