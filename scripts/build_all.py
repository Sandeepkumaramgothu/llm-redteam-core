#!/usr/bin/env python
# ELI5: Helper to build the paper metrics page (and any other scripts you have).

import argparse, pathlib, sys, subprocess

def pick_run_dir(run_dir_arg: str|None) -> str:
    if run_dir_arg and run_dir_arg != 'latest':
        return run_dir_arg
    runs = sorted([p for p in pathlib.Path('runs').glob('*') if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for d in runs:
        if list(d.glob('iter_*')):
            return d.as_posix()
    raise SystemExit('No runs with iter_* found.')

def call(pyfile, *args):
    cmd = [sys.executable, pyfile, *args]
    print('>', ' '.join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='latest')
    args = ap.parse_args()
    run_dir = pick_run_dir(args.run_dir)

    # Build paper metrics (CSVs + HTML)
    call('scripts/paper_metrics.py', '--run-dir', run_dir)

    print('\nOpen:')
    print(run_dir + '/paper_metrics/paper_report.html')

if __name__ == '__main__':
    main()
