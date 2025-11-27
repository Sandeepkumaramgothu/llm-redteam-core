
╔═══════════════════════════════════════════════════════════════╗
║                    QUICK REFERENCE CARD                       ║
║              Save this for future sessions!                   ║
╚═══════════════════════════════════════════════════════════════╝

🔧 SETUP (One-time per session)
═══════════════════════════════════════════════════════════════
import sys
from pathlib import Path

PROJECT_ROOT = Path('/content/drive/MyDrive/llm-redteam-core')
sys.path.insert(0, str(PROJECT_ROOT))

🚀 RUN QUICK TEST
═══════════════════════════════════════════════════════════════
from src.core import engine
import yaml

with open(PROJECT_ROOT / 'configs/test.yaml') as f:
    config = yaml.safe_load(f)

run_dir = engine.run_attack(config)

📊 GENERATE REPORT
═══════════════════════════════════════════════════════════════
from src.utils import postrun

postrun.analyze_and_visualize(run_dir)

📥 VIEW REPORT
═══════════════════════════════════════════════════════════════
from IPython.display import FileLink

report = PROJECT_ROOT / 'paper_metrics/report.html'
display(FileLink(report))

🔬 CUSTOM ANALYSIS
═══════════════════════════════════════════════════════════════
import pandas as pd

# Load results
df = pd.read_csv(f"{run_dir}/results.csv")

# Filter successful attacks
successes = df[df['success'] == True]

# High-risk responses
high_risk = df[df['risk_level'].isin(['L2', 'L3'])]

# Group by risk type
by_type = df.groupby('risk_type')['success'].mean()

📈 CREATE CUSTOM CHARTS
═══════════════════════════════════════════════════════════════
from src.utils import visualizations

# Create Sankey diagram
fig = visualizations.create_sankey_flow(df)
fig.show()

# Export all figures
visualizations.export_all_figures(df, 'my_charts', format='html')

⚙️ MODIFY CONFIGURATION
═══════════════════════════════════════════════════════════════
# Edit config file
config_path = PROJECT_ROOT / 'configs/test.yaml'

# Or create inline
config = {
    'name': 'Custom Test',
    'seed': 42,
    'paths': {
        'seeds': str(PROJECT_ROOT / 'data/dna/seeds.jsonl'),
        'runs': str(PROJECT_ROOT / 'runs'),
        'paper_metrics': str(PROJECT_ROOT / 'paper_metrics')
    },
    'red': {'use_openai': False},  # Template-based
    'target': {'hf_model': 'gpt2-xl'},
    'attack': {'n_seeds': 10, 'n_iters': 2}
}

run_dir = engine.run_attack(config)

🔍 COMMON TASKS
═══════════════════════════════════════════════════════════════
# Find latest run
runs_dir = PROJECT_ROOT / 'runs'
latest = sorted(runs_dir.glob('*'))[-1]

# Load specific run
from src.utils import postrun
df = postrun.load_results(str(latest))

# Compute metrics
from src.utils import metrics
m = metrics.compute_all_metrics(df.to_dict('records'))
metrics.print_metrics_summary(m)

# Export to CSV
df.to_csv('my_analysis.csv', index=False)

🐛 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════
# Clear modules and reimport
import sys
for mod in list(sys.modules.keys()):
    if mod.startswith('src'):
        del sys.modules[mod]

# Then reimport
from src.core import engine

# Check what's imported
print(sys.path)
print([k for k in sys.modules.keys() if 'src' in k])

📞 GETTING HELP
═══════════════════════════════════════════════════════════════
# Check module documentation
from src.core import seeds
help(seeds.SeedLedger)

# View source
import inspect
print(inspect.getsource(seeds.sample_seeds))

═══════════════════════════════════════════════════════════════
