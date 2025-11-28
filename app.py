import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

# Page Config
st.set_page_config(
    page_title="AdversaFlow Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #764ba2;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_data(run_path):
    """Load results and metrics for a specific run."""
    run_path = Path(run_path)
    results_csv = run_path / "results.csv"
    metrics_json = run_path / "metrics.json"
    
    df = pd.DataFrame()
    metrics = {}
    
    if results_csv.exists():
        df = pd.read_csv(results_csv)
    
    if metrics_json.exists():
        with open(metrics_json, 'r') as f:
            metrics = json.load(f)
            
    return df, metrics

def get_all_runs(runs_dir):
    """Get list of all available runs."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    return sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)

# Main App Logic
def main():
    st.markdown('<div class="main-header">🛡️ AdversaFlow Red Teaming Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar: Run Selection
    st.sidebar.title("Experiment Selection")
    
    # Default runs directory
    # Try to find runs dir relative to current script or in common locations
    possible_run_dirs = [
        Path("runs"),
        Path("c:/Users/sande/Documents/Anti_Gravity_Git_Repo/llm-redteam-core/runs"),
        Path("/content/drive/MyDrive/llm-redteam-core/runs") # Fallback for colab paths in config
    ]
    
    runs_dir = None
    for d in possible_run_dirs:
        if d.exists():
            runs_dir = d
            break
            
    if not runs_dir:
        st.error("❌ Could not find 'runs' directory. Please run an experiment first.")
        st.info("Tip: Run `python run_experiment.py configs/your_config.yaml` to generate data.")
        return

    all_runs = get_all_runs(runs_dir)
    
    if not all_runs:
        st.warning(f"⚠️ No runs found in `{runs_dir}`.")
        return

    selected_run_name = st.sidebar.selectbox(
        "Select Run ID",
        options=[r.name for r in all_runs]
    )
    
    selected_run_path = runs_dir / selected_run_name
    
    # Load Data
    with st.spinner(f"Loading data for {selected_run_name}..."):
        df, metrics = load_data(selected_run_path)

    if df.empty:
        st.error("❌ No results data found for this run.")
        return

    # --- Dashboard Overview ---
    st.header("📊 Overview")
    
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    asr = metrics.get('attack_success_rate', 0) * 100 if metrics else (df['success'].mean() * 100 if 'success' in df.columns else 0)
    avg_tox = metrics.get('toxicity', {}).get('avg', 0) if metrics else (df['toxicity'].mean() if 'toxicity' in df.columns else 0)
    total_queries = len(df)
    severe_risk = df[df['risk_level'].isin(['L2', 'L3'])].shape[0] if 'risk_level' in df.columns else 0

    c1.metric("Attack Success Rate", f"{asr:.1f}%")
    c2.metric("Avg Toxicity", f"{avg_tox:.2f}")
    c3.metric("Total Queries", f"{total_queries}")
    c4.metric("Severe Risks (L2/L3)", f"{severe_risk}")

    # --- Visualizations ---
    st.divider()
    
    col_charts_1, col_charts_2 = st.columns(2)
    
    with col_charts_1:
        st.subheader("Risk Distribution")
        if 'risk_level' in df.columns:
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            fig_risk = px.bar(risk_counts, x='Risk Level', y='Count', color='Risk Level', 
                              color_discrete_map={'L0': '#2ecc71', 'L1': '#f1c40f', 'L2': '#e67e22', 'L3': '#e74c3c'})
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("Risk level data not available.")

    with col_charts_2:
        st.subheader("ASR by Iteration")
        if 'iter' in df.columns:
            iter_stats = df.groupby('iter')['success'].mean().reset_index()
            iter_stats['success'] *= 100
            fig_iter = px.line(iter_stats, x='iter', y='success', markers=True, 
                               title="Attack Success Rate over Iterations",
                               labels={'success': 'ASR (%)', 'iter': 'Iteration'})
            st.plotly_chart(fig_iter, use_container_width=True)
        else:
            st.info("Iteration data not available.")

    # --- Chat Explorer ---
    st.divider()
    st.header("💬 Conversation Explorer")
    
    # Filters
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_only_success = st.checkbox("Show only successful attacks", value=True)
    with filter_col2:
        if 'risk_type' in df.columns:
            selected_risk_type = st.selectbox("Filter by Risk Type", ["All"] + list(df['risk_type'].unique()))
        else:
            selected_risk_type = "All"

    # Apply filters
    filtered_df = df.copy()
    if show_only_success:
        filtered_df = filtered_df[filtered_df['success'] == True]
    if selected_risk_type != "All":
        filtered_df = filtered_df[filtered_df['risk_type'] == selected_risk_type]

    if filtered_df.empty:
        st.info("No conversations match the current filters.")
    else:
        # Pagination / Selection
        total_convs = len(filtered_df)
        st.write(f"Showing {total_convs} conversations")
        
        # Select specific conversation index
        selected_idx = st.number_input("Select Conversation Index", min_value=0, max_value=max(0, total_convs-1), value=0)
        
        if total_convs > 0:
            row = filtered_df.iloc[selected_idx]
            
            with st.container():
                st.markdown(f"**Risk Type:** `{row.get('risk_type', 'N/A')}` | **Risk Level:** `{row.get('risk_level', 'N/A')}`")
                
                # Display Conversation History if available
                history = row.get('conversation_history')
                if isinstance(history, str):
                    try:
                        history = json.loads(history.replace("'", '"')) # Basic cleanup attempt if it's a string repr
                    except:
                        history = None
                
                if isinstance(history, list) and history:
                    for turn in history:
                        st.chat_message("user", avatar="🔴").write(turn.get('prompt', ''))
                        st.chat_message("assistant", avatar="🛡️").write(turn.get('response', ''))
                else:
                    # Fallback to simple prompt/response
                    st.chat_message("user", avatar="🔴").write(row.get('combined', row.get('adversarial', '')))
                    st.chat_message("assistant", avatar="🛡️").write(row.get('response', ''))

    # --- Data Table ---
    st.divider()
    st.header("📋 Raw Data")
    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
