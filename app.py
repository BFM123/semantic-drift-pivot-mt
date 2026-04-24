"""
Semantic Drift Evaluation Dashboard
Interactive visualization of translation quality improvements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="Semantic Drift Evaluation Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .improvement-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .title-section {
        text-align: center;
        margin-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Set base path - handle both local and Streamlit environments
script_dir = Path(__file__).parent
BASE_DIR = script_dir if script_dir.name != ".streamlit" else script_dir.parent
RESULTS_DIR = BASE_DIR / "results" / "metrics"

# Debug: Display path information in sidebar
with st.sidebar:
    st.markdown("### 🔧 Debug Info")
    st.code(f"Base Dir: {BASE_DIR}", language="text")
    st.code(f"Results Dir: {RESULTS_DIR}", language="text")
    
    # Check if directories exist
    if RESULTS_DIR.exists():
        st.success(f"✅ Results directory exists")
        files = list(RESULTS_DIR.glob("*.csv"))
        st.write(f"Found {len(files)} CSV files:")
        for f in files:
            st.write(f"  • {f.name}")
    else:
        st.error(f"❌ Results directory not found: {RESULTS_DIR}")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Load data with better error handling
def load_data():
    semantic_drift_file = RESULTS_DIR / "semantic_drift_results.csv"
    metrics_file = RESULTS_DIR / "metrics_summary.csv"
    
    errors = []
    
    if not semantic_drift_file.exists():
        errors.append(f"Missing: {semantic_drift_file}")
    if not metrics_file.exists():
        errors.append(f"Missing: {metrics_file}")
    
    if errors:
        return None, None, errors
    
    try:
        semantic_df = pd.read_csv(semantic_drift_file)
        metrics_df = pd.read_csv(metrics_file)
        return semantic_df, metrics_df, []
    except Exception as e:
        return None, None, [str(e)]

semantic_df, metrics_df, load_errors = load_data()

# Title section
st.markdown("""
<div class='title-section'>
    <h1>🌐 Semantic Drift Evaluation Dashboard</h1>
    <p>Analyzing translation quality improvements with augmentation</p>
</div>
""", unsafe_allow_html=True)

# Check if data exists
if semantic_df is None or metrics_df is None:
    st.error("❌ Results not found")
    st.markdown("### Error Details:")
    for error in load_errors:
        st.code(error, language="text")
    
    st.info("📋 To generate results, run: `python scripts/05_evaluate_semantic_drift.py`")
    st.stop()
else:
    # Show success message and loading details
    st.sidebar.success("✅ Results loaded successfully!")
    st.sidebar.info(f"📊 Samples loaded: {len(semantic_df)}")
    st.sidebar.divider()
    
    # Extract metrics
    baseline_drift = metrics_df['baseline_drift'].values[0]
    augmented_drift = metrics_df['augmented_drift'].values[0]
    improvement_pct = metrics_df['drift_reduction_percent'].values[0]
    bert_baseline = metrics_df['bert_baseline'].values[0]
    bert_augmented = metrics_df['bert_augmented'].values[0]
    
    # ============ KEY METRICS ============
    st.markdown("### 📊 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Baseline Drift",
            value=f"{baseline_drift:.4f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Augmented Drift",
            value=f"{augmented_drift:.4f}",
            delta=f"-{baseline_drift - augmented_drift:.4f}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Drift Reduction",
            value=f"{improvement_pct:.2f}%",
            delta=None,
            delta_color="normal"
        )
    
    with col4:
        bert_improvement = ((bert_augmented - bert_baseline) / bert_baseline * 100)
        st.metric(
            label="BERTScore Improvement",
            value=f"{bert_improvement:.2f}%",
            delta=f"+{bert_improvement:.2f}%"
        )
    
    # ============ VISUALIZATIONS ============
    st.markdown("---")
    
    col_viz1, col_viz2 = st.columns(2)
    
    # Drift Comparison
    with col_viz1:
        st.markdown("### 📉 Semantic Drift Comparison")
        
        drift_comparison = go.Figure(data=[
            go.Bar(
                x=['Baseline', 'Augmented'],
                y=[baseline_drift, augmented_drift],
                marker_color=['#667eea', '#f5576c'],
                text=[f'{baseline_drift:.4f}', f'{augmented_drift:.4f}'],
                textposition='auto',
            )
        ])
        
        drift_comparison.update_layout(
            yaxis_title="Semantic Drift (↓ better)",
            xaxis_title="Pipeline Version",
            height=400,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(drift_comparison, use_container_width=True)
    
    # BERTScore Comparison
    with col_viz2:
        st.markdown("### 🧠 BERTScore Comparison")
        
        bert_comparison = go.Figure(data=[
            go.Bar(
                x=['Baseline', 'Augmented'],
                y=[bert_baseline, bert_augmented],
                marker_color=['#667eea', '#f093fb'],
                text=[f'{bert_baseline:.4f}', f'{bert_augmented:.4f}'],
                textposition='auto',
            )
        ])
        
        bert_comparison.update_layout(
            yaxis_title="BERTScore F1 (↑ better)",
            xaxis_title="Pipeline Version",
            height=400,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(bert_comparison, use_container_width=True)
    
    # ============ DRIFT DISTRIBUTION ============
    st.markdown("---")
    st.markdown("### 📊 Drift Distribution Across Samples")
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=semantic_df['drift_baseline'],
            name='Baseline',
            opacity=0.7,
            marker_color='#667eea',
            nbinsx=30
        ))
        
        fig_hist.add_trace(go.Histogram(
            x=semantic_df['drift_augmented'],
            name='Augmented',
            opacity=0.7,
            marker_color='#f5576c',
            nbinsx=30
        ))
        
        fig_hist.update_layout(
            title="Drift Distribution",
            xaxis_title="Semantic Drift",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400,
            hovermode='x'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_dist2:
        drift_improvement = semantic_df['drift_baseline'] - semantic_df['drift_augmented']
        
        fig_improvement = go.Figure(data=[
            go.Histogram(
                x=drift_improvement,
                marker_color='#f093fb',
                nbinsx=30,
            )
        ])
        
        fig_improvement.update_layout(
            title="Per-Sample Drift Improvement",
            xaxis_title="Drift Reduction (Baseline - Augmented)",
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
            hovermode='x'
        )
        
        st.plotly_chart(fig_improvement, use_container_width=True)
    
    # ============ INTERACTIVE EXPLORER ============
    st.markdown("---")
    st.markdown("### 🔍 Translation Samples Explorer")
    
    # Sample selection
    sample_idx = st.slider(
        "Select a sample to view:",
        0,
        len(semantic_df) - 1,
        0
    )
    
    # Display selected sample
    sample = semantic_df.iloc[sample_idx]
    
    col_sample1, col_sample2 = st.columns(2)
    
    with col_sample1:
        st.markdown("#### Source Language (Chichewa)")
        st.info(sample['chichewa'])
        
        st.markdown("#### Baseline English")
        st.write(sample['en_baseline'])
        
        st.markdown("#### Baseline Hindi")
        st.success(sample['hi_baseline'])
        
        st.metric("Baseline Drift", f"{sample['drift_baseline']:.4f}")
    
    with col_sample2:
        st.markdown("#### Augmented English")
        st.write(sample['en_augmented'])
        
        st.markdown("#### Augmented Hindi")
        st.success(sample['hi_augmented'])
        
        st.metric("Augmented Drift", f"{sample['drift_augmented']:.4f}")
        
        improvement_value = sample['drift_baseline'] - sample['drift_augmented']
        if improvement_value > 0:
            st.success(f"✅ Improved by {improvement_value:.4f}")
        else:
            st.warning(f"⚠️ Worsened by {abs(improvement_value):.4f}")
    
    # ============ STATISTICS ============
    st.markdown("---")
    st.markdown("### 📈 Detailed Statistics")
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.markdown("#### Baseline Drift")
        stats_base = {
            "Mean": semantic_df['drift_baseline'].mean(),
            "Median": semantic_df['drift_baseline'].median(),
            "Std Dev": semantic_df['drift_baseline'].std(),
            "Min": semantic_df['drift_baseline'].min(),
            "Max": semantic_df['drift_baseline'].max()
        }
        for key, val in stats_base.items():
            st.metric(key, f"{val:.4f}")
    
    with col_stats2:
        st.markdown("#### Augmented Drift")
        stats_aug = {
            "Mean": semantic_df['drift_augmented'].mean(),
            "Median": semantic_df['drift_augmented'].median(),
            "Std Dev": semantic_df['drift_augmented'].std(),
            "Min": semantic_df['drift_augmented'].min(),
            "Max": semantic_df['drift_augmented'].max()
        }
        for key, val in stats_aug.items():
            st.metric(key, f"{val:.4f}")
    
    with col_stats3:
        st.markdown("#### Improvement Stats")
        improvement_vals = semantic_df['drift_baseline'] - semantic_df['drift_augmented']
        improvement_stats = {
            "Mean Improvement": improvement_vals.mean(),
            "Samples Improved": (improvement_vals > 0).sum(),
            "Improvement Rate": f"{(improvement_vals > 0).sum() / len(improvement_vals) * 100:.1f}%",
            "Best Improvement": improvement_vals.max(),
            "Worst Case": improvement_vals.min()
        }
        for key, val in improvement_stats.items():
            st.metric(key, str(val))
    
    # ============ SCATTER PLOT ============
    st.markdown("---")
    st.markdown("### 🎯 Baseline vs Augmented Drift")
    
    scatter_fig = go.Figure(data=[
        go.Scatter(
            x=semantic_df['drift_baseline'],
            y=semantic_df['drift_augmented'],
            mode='markers',
            marker=dict(
                size=6,
                color=semantic_df['drift_baseline'] - semantic_df['drift_augmented'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Improvement")
            ),
            text=[
                f"Sample {i}<br>" +
                f"Baseline: {row['drift_baseline']:.4f}<br>" +
                f"Augmented: {row['drift_augmented']:.4f}<br>" +
                f"Improvement: {row['drift_baseline'] - row['drift_augmented']:.4f}"
                for i, (idx, row) in enumerate(semantic_df.iterrows())
            ],
            hoverinfo='text'
        )
    ])
    
    scatter_fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=1, y1=1,
        line=dict(color="gray", width=2, dash="dash"),
        xref="x", yref="y"
    )
    
    scatter_fig.update_layout(
        title="Baseline vs Augmented Drift (diagonal = no improvement)",
        xaxis_title="Baseline Drift",
        yaxis_title="Augmented Drift",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # ============ DATA TABLE ============
    st.markdown("---")
    st.markdown("### 📋 Full Results Table")
    
    if st.checkbox("Show full results table"):
        st.dataframe(
            semantic_df,
            use_container_width=True,
            height=400
        )
    
    # ============ SUMMARY ============
    st.markdown("---")
    st.markdown("### 📝 Summary")
    
    summary_text = f"""
    **Project:** Semantic Drift Evaluation with Augmentation
    
    **Pipeline:** Chichewa → English → Hindi
    
    **Key Findings:**
    - **Baseline Semantic Drift:** {baseline_drift:.4f}
    - **Augmented Semantic Drift:** {augmented_drift:.4f}
    - **Drift Reduction:** {improvement_pct:.2f}%
    - **Samples Analyzed:** {len(semantic_df)}
    - **Success Rate:** {(semantic_df['drift_baseline'] > semantic_df['drift_augmented']).sum() / len(semantic_df) * 100:.1f}%
    
    **Augmentation Techniques Applied:**
    1. Back-translation (EN → NY → EN)
    2. Phrase stabilization
    3. Semantic normalization
    
    **Validation Metrics:**
    - BERTScore (Baseline): {bert_baseline:.4f}
    - BERTScore (Augmented): {bert_augmented:.4f}
    """
    
    st.info(summary_text)
    
    # Download button
    st.markdown("---")
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        csv_data = semantic_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name="semantic_drift_results.csv",
            mime="text/csv"
        )
    
    with col_download2:
        metrics_csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics Summary",
            data=metrics_csv,
            file_name="metrics_summary.csv",
            mime="text/csv"
        )
