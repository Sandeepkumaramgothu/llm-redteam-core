#!/usr/bin/env python3
"""
Report Generation Script
========================
Generate HTML reports from completed runs.

Usage:
    python generate_report.py --run-dir runs/20241016_123456
    python generate_report.py --latest
"""

import sys
import argparse
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import postrun


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not run_dirs:
        raise ValueError(f"No runs found in {runs_dir}")
    
    return run_dirs[0]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate HTML report from completed run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for specific run
  python generate_report.py --run-dir runs/20241016_123456
  
  # Generate report for latest run
  python generate_report.py --latest
        """
    )
    
    parser.add_argument(
        '--run-dir',
        type=str,
        help='Path to run directory'
    )
    
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use the latest run'
    )
    
    args = parser.parse_args()
    
    # Determine run directory
    if args.latest:
        runs_dir = PROJECT_ROOT / "runs"
        run_dir = find_latest_run(runs_dir)
        print(f"Using latest run: {run_dir}")
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            return 1
    else:
        print("Error: Specify --run-dir or --latest")
        parser.print_help()
        return 1
    
    # Generate report
    try:
        metrics = postrun.analyze_and_visualize(str(run_dir))
        
        print(f"\n✅ Report generation complete!")
        print(f"   Run: {run_dir}")
        print(f"   ASR: {metrics['attack_success_rate']:.2f}%")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
