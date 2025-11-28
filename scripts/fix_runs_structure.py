import os
import shutil
from pathlib import Path

def fix_runs_structure():
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("Runs directory not found.")
        return

    print(f"Scanning {runs_dir}...")
    
    # List all subdirectories
    subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    for d in subdirs:
        # Check if it's a nested structure (contains another dir with the same name or similar)
        # Or just check if it contains a subdirectory that looks like a run
        
        # In this case, the outer dir is like "20251115...-20251128..."
        # And inner dir is "20251115..."
        
        inner_dirs = [sub for sub in d.iterdir() if sub.is_dir()]
        
        for inner in inner_dirs:
            # Check if inner dir has results.csv
            if (inner / "results.csv").exists():
                print(f"Found nested run: {inner}")
                
                target_path = runs_dir / inner.name
                
                if target_path.exists():
                    print(f"⚠️ Target {target_path} already exists. Skipping to avoid overwrite.")
                    continue
                
                print(f"Moving {inner} to {target_path}...")
                try:
                    shutil.move(str(inner), str(target_path))
                    print("✅ Moved successfully.")
                    
                    # Check if outer dir is empty (ignoring zip files)
                    remaining = [x for x in d.iterdir() if x.name != inner.name]
                    if not remaining:
                        print(f"Removing empty outer dir: {d}")
                        d.rmdir()
                    else:
                        print(f"Outer dir {d} not empty, keeping it. Contents: {[x.name for x in remaining]}")
                        
                except Exception as e:
                    print(f"❌ Failed to move: {e}")

if __name__ == "__main__":
    fix_runs_structure()
