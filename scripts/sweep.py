
#!/usr/bin/env python3
import sys, os, json, time, subprocess, pathlib, datetime
import argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run_one(config_path: str, iters: int, seeds: int, note: str = "") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    env = os.environ.copy()
    # Just pass overrides as CLI flags your run.py already supports
    cmd = [
        sys.executable, str(ROOT / "scripts/run.py"),
        "--config", config_path,
        "--iters", str(iters),
        "--seeds", str(seeds),
    ]
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, env=env)

    # Find newest run folder (created by run.py)
    runs = sorted([p for p in (ROOT/"runs").glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    run_dir = str(runs[-1])
    print("✓ finished:", run_dir)
    return run_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/openai_red_llama3.yaml")
    ap.add_argument("--jobs", nargs="+", required=True,
                    help="Pairs of iters:seeds, e.g. 4:30 5:40 6:50")
    args = ap.parse_args()

    produced = []
    for pair in args.jobs:
        it, sd = pair.split(":")
        run_dir = run_one(args.config, int(it), int(sd))
        produced.append(run_dir)
        # Build metrics and report
        subprocess.check_call([sys.executable, "scripts/paper_metrics.py", "--run-dir", run_dir], cwd=ROOT)
    print("All done.")
    for r in produced:
        print(" -", r)

if __name__ == "__main__":
    main()
