\
#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml
from src.core.config import load_config
from src.core.engine import run_loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", action="store_true", help="resume last run folder for this run_name")
    ap.add_argument("--iters", type=int, default=None, help="override iterations")
    ap.add_argument("--seeds", type=int, default=None, help="override seeds_per_iter")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.iters is not None:
        cfg["run"]["iterations"] = int(args.iters)
    if args.seeds is not None:
        cfg["run"]["seeds_per_iter"] = int(args.seeds)
    run_loop(cfg, resume=args.resume)

if __name__ == "__main__":
    main()
