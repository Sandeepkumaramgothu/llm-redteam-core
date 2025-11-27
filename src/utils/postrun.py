"""Post-Run Analysis with Enhanced Visualizations and CSV Summaries"""
from pathlib import Path
import pandas as pd
import json
from src.utils import metrics, visualization

def safe_print(*args, **kwargs):
    """Print with proper Unicode encoding handling."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode('utf-8', 'surrogateescape').decode('utf-8', 'ignore'))
            else:
                safe_args.append(arg)
        print(*safe_args, **kwargs)
    except Exception:
        pass

def analyze_and_visualize(run_dir: str):
    """
    Post-experiment analysis pipeline:
      - Loads results, computes metrics
      - Saves metrics.json, visual dashboards, CSV summaries
      - Generates all paper-quality/interpretable visualizations and main HTML data explorer
      - Outputs iteration and global summary CSVs for record-keeping
    """
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        safe_print(f"❌ Results not found: {results_csv}")
        return

    safe_print(f"Loading results from {results_csv}")
    df = pd.read_csv(results_csv)
    safe_print(f"✓ Loaded {len(df)} records\n")

    # Compute all summary metrics
    safe_print("Computing metrics...")
    computed_metrics = metrics.compute_metrics(df)
    metrics.print_metrics_summary(computed_metrics)

    # Save metrics to JSON
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(computed_metrics, f, indent=2)
    safe_print(f"✓ Metrics saved: {metrics_file}\n")

    # Generate all major visualizations
    charts = visualization.create_all_visualizations(df, run_dir)

    # Generate and save main HTML dashboard
    report_path = run_dir / "report.html"
    visualization.generate_html_report(df, computed_metrics, charts, report_path)

    # Generate interactive data explorer table
    data_table_path = run_dir / "data_explorer.html"
    visualization.generate_interactive_data_table(df, data_table_path)

    # === NEW: Export iteration and global summaries as CSVs ===
    iter_csv = run_dir / "iteration_summary.csv"
    global_csv = run_dir / "global_summary.csv"
    visualization.save_iteration_summary_csv(df, iter_csv)
    visualization.save_global_summary_csv(computed_metrics, global_csv)

    safe_print(f"\n{'='*70}")
    safe_print(f"✅ ANALYSIS COMPLETE")
    safe_print(f"📊 Main Dashboard: {report_path}")
    safe_print(f"🔍 Data Explorer:  {data_table_path}")
    safe_print(f"📑 Iteration CSV:  {iter_csv}")
    safe_print(f"📈 Global Summary: {global_csv}")
    safe_print(f"{'='*70}\n")

    return str(report_path)
