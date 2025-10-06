
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, subprocess, glob, pathlib

def stream(cmd, cwd=None, env=None, heartbeat=15):
 env = {**os.environ, **(env or {})}
 if cmd[0].endswith("python") or cmd[0].endswith("python3"):
     if "-u" not in cmd: cmd.insert(1, "-u")
 print("\n▶", " ".join(cmd), flush=True)
 p = subprocess.Popen(
     cmd, cwd=cwd, env=env,
     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
     text=True, bufsize=1
 )
 last = time.time()
 try:
     while True:
         line = p.stdout.readline()
         if line:
             print(line, end="")
             last = time.time()
         elif p.poll() is not None:
             break
         else:
             if time.time() - last > heartbeat:
                 print(" … (still working)", flush=True)
                 last = time.time()
             time.sleep(0.3)
 finally:
     ret = p.wait()
 if ret != 0:
     raise subprocess.CalledProcessError(ret, cmd)
 return ret

def newest_run():
 runs = sorted([p for p in glob.glob("runs/*") if pathlib.Path(p).is_dir()],
               key=lambda p: pathlib.Path(p).stat().st_mtime)
 return runs[-1] if runs else None

def main():
 import argparse
 ap = argparse.ArgumentParser()
 ap.add_argument("--config", required=False, default="configs/openai_red_llama3_fp16.yaml")
 ap.add_argument("--iters", default="2")
 ap.add_argument("--seeds", default="20")
 args = ap.parse_args()

 os.environ["PYTHONUNBUFFERED"] = "1"
 os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "1"
 os.environ["TRANSFORMERS_VERBOSITY"] = "info"

 stream([sys.executable, "scripts/run.py", "--config", args.config, "--iters", args.iters, "--seeds", args.seeds])
 rd = newest_run()
 if rd is None:
     print("No run directory found.")
     return
 print("📦 run dir:", rd)
 stream([sys.executable, "scripts/rescore.py", "--run-dir", rd])
 stream([sys.executable, "scripts/paper_metrics.py", "--run-dir", rd])
 print("\\n✅ done. files:")
 print(" •", rd + "/interactions_full.csv")
 print(" •", rd + "/metrics.csv")
 print(" •", rd + "/metrics_summary.json")

if __name__ == "__main__":
 main()
