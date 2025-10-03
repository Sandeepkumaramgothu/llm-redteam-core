#!/usr/bin/env python
from __future__ import annotations
# ELI5: This script reads YAML, fills defaults, and kicks off run_loop.

import sys, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from datetime import datetime
from src.core.config import load_config
from src.core.engine import run_loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--seeds', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    r = cfg.setdefault('run', {})

    # Normalize iteration keys
    if args.iters is not None:
        r['iterations'] = r['iters'] = int(args.iters)
    else:
        if 'iterations' in r: r.setdefault('iters', int(r['iterations']))
        elif 'iters' in r:   r.setdefault('iterations', int(r['iters']))
        else:                r['iterations'] = r['iters'] = 5

    # Normalize seed keys
    if args.seeds is not None:
        r['seeds_per_iter'] = r['seeds'] = int(args.seeds)
    else:
        if 'seeds_per_iter' in r: r.setdefault('seeds', int(r['seeds_per_iter']))
        elif 'seeds' in r:        r.setdefault('seeds_per_iter', int(r['seeds']))
        else:                     r['seeds_per_iter'] = r['seeds'] = 25

    r.setdefault('rng_seed', 7)
    r.setdefault('output_dir', (ROOT/'runs').as_posix())
    if not r.get('run_name'):
        proj = (cfg.get('project') or {}).get('name','run')
        r['run_name'] = f"{proj}_{datetime.now():%Y%m%d-%H%M%S}"
    os.makedirs(r['output_dir'], exist_ok=True)

    cfg.setdefault('reward', {})
    cfg['reward'].setdefault('toxicity_threshold', 0.5)

    run_loop(cfg, resume=args.resume)

if __name__ == '__main__':
    main()
