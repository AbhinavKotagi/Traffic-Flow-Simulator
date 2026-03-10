"""
============================================================
 PART 4 & 5 — STREAMLIT INTERACTIVE APP
============================================================
AI-Based Traffic Policy Simulation System
Decision-Support Tool for Traffic Authorities

Features:
  • Interactive sliders for traffic inputs
  • Policy selection dropdown
  • Before vs After bar chart
  • Trend graph (congestion vs vehicle count)
  • Scenario history with session state
  • Side-by-side comparison of current vs previous scenario

Run:
    streamlit run app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import time
import joblib

from simulator import simulate_policy, simulate_trend, POLICIES, format_result_summary
from feature_extraction import generate_synthetic_dataset

# ── Page configuration ────────────────────────────────────
st.set_page_config(
    page_title="Traffic Policy Simulator",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a40);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9e9e9e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Result banner */
    .result-banner {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-left: 5px solid #4caf50;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 12px 0;
    }
    .result-banner-red {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        border-left: 5px solid #ef5350;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 12px 0;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    .result-subtitle {
        font-size: 0.9rem;
        color: #c8e6c9;
    }

    /* Scenario history item */
    .history-item {
        background: #1e2130;
        border: 1px solid #3d4166;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.85rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4fc3f7;
        border-bottom: 2px solid #4fc3f7;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    
    .stSlider > div > div > div > div {
        background: #4fc3f7;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════════════
if "scenario_history" not in st.session_state:
    st.session_state.scenario_history = []

if "model" not in st.session_state:
    st.session_state.model  = None
    st.session_state.scaler = None

if "dataset" not in st.session_state:
    st.session_state.dataset = None


# ════════════════════════════════════════════════════════════
#  MODEL LOADER
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained model and scaler; train from scratch if needed."""
    try:
        if os.path.exists("traffic_model.pkl") and os.path.exists("scaler.pkl"):
            model  = joblib.load("traffic_model.pkl")
            scaler = joblib.load("scaler.pkl")
            return model, scaler
    except Exception:
        pass

    # Auto-train if model not found
    st.info("🔄 Training model for the first time … (one-time setup)")
    from feature_extraction import generate_synthetic_dataset
    from train_model import preprocess, train_model, save_artifacts

    if not os.path.exists("dataset.csv"):
        generate_synthetic_dataset("dataset.csv")

    df = pd.read_csv("dataset.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    model = train_model(X_train, y_train)
    save_artifacts(model, scaler)
    return model, scaler


# ════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ════════════════════════════════════════════════════════════

def build_bar_chart(result: dict) -> go.Figure:
    """Before vs After bar chart for current simulation."""
    policy_color = POLICIES[result["policy_key"]]["color"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Baseline"],
        y=[result["baseline_score"]],
        name="Baseline",
        marker_color="#ef5350",
        marker_line_color="#ff8a80",
        marker_line_width=1.5,
        width=0.4,
        text=[f"{result['baseline_score']:.1f}"],
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))

    fig.add_trace(go.Bar(
        x=["After Policy"],
        y=[result["modified_score"]],
        name="After Policy",
        marker_color=policy_color,
        marker_line_color="#ffffff",
        marker_line_width=1.5,
        width=0.4,
        text=[f"{result['modified_score']:.1f}"],
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))

    fig.update_layout(
        title=dict(text=f"Congestion Score: Before vs After<br><sup>{result['policy_label']}</sup>",
                   font=dict(size=16, color="white")),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        yaxis=dict(
            title="Congestion Score",
            gridcolor="#2a2d3e",
            zerolinecolor="#3d4166",
            range=[0, max(result["baseline_score"], 1) * 1.3],
        ),
        xaxis=dict(gridcolor="#2a2d3e"),
        showlegend=True,
        legend=dict(bgcolor="#1e2130", bordercolor="#3d4166", borderwidth=1),
        height=380,
        bargap=0.4,
    )
    return fig


def build_trend_chart(result: dict, model, scaler) -> go.Figure:
    """Congestion vs VehicleCount trend lines (baseline and after policy)."""
    trend_df = simulate_trend(
        wrong_parked=result["inputs"]["WrongParked"],
        stopped=result["inputs"]["Stopped"],
        policy_key=result["policy_key"],
        model=model,
        scaler=scaler,
    )
    policy_color = POLICIES[result["policy_key"]]["color"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["VehicleCount"], y=trend_df["Baseline"],
        name="Baseline Trend",
        mode="lines",
        line=dict(color="#ef5350", width=2.5),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["VehicleCount"], y=trend_df["AfterPolicy"],
        name=f"After: {result['policy_label']}",
        mode="lines",
        line=dict(color=policy_color, width=2.5, dash="dash"),
        fill="tozeroy", fillcolor=f"rgba(79,195,247,0.08)",
    ))

    # Mark current vehicle count
    fig.add_vline(
        x=result["inputs"]["VehicleCount"],
        line_dash="dot", line_color="#ffeb3b", line_width=1.5,
        annotation_text=f" Current: {result['inputs']['VehicleCount']}",
        annotation_font_color="#ffeb3b",
    )

    fig.update_layout(
        title=dict(text="Congestion Score vs Vehicle Count (Trend Analysis)",
                   font=dict(size=16, color="white")),
        xaxis=dict(title="Vehicle Count", gridcolor="#2a2d3e"),
        yaxis=dict(title="Congestion Score", gridcolor="#2a2d3e"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1e2130", bordercolor="#3d4166", borderwidth=1),
        height=380,
    )
    return fig


def build_comparison_chart(current: dict, previous: dict) -> go.Figure:
    """Side-by-side comparison of current vs previous scenario."""
    categories  = ["VehicleCount", "WrongParked", "Stopped", "CongestionScore"]
    curr_values = [
        current["inputs"]["VehicleCount"],
        current["inputs"]["WrongParked"],
        current["inputs"]["Stopped"],
        current["modified_score"],
    ]
    prev_values = [
        previous["inputs"]["VehicleCount"],
        previous["inputs"]["WrongParked"],
        previous["inputs"]["Stopped"],
        previous["modified_score"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Previous: {previous['policy_label']}",
        x=categories, y=prev_values,
        marker_color="#78909c",
        text=[f"{v:.1f}" for v in prev_values],
        textposition="outside", textfont=dict(size=11, color="white"),
    ))
    fig.add_trace(go.Bar(
        name=f"Current: {current['policy_label']}",
        x=categories, y=curr_values,
        marker_color=POLICIES[current["policy_key"]]["color"],
        text=[f"{v:.1f}" for v in curr_values],
        textposition="outside", textfont=dict(size=11, color="white"),
    ))

    fig.update_layout(
        title=dict(text="Scenario Comparison: Current vs Previous",
                   font=dict(size=16, color="white")),
        barmode="group",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        yaxis=dict(title="Value", gridcolor="#2a2d3e"),
        xaxis=dict(gridcolor="#2a2d3e"),
        legend=dict(bgcolor="#1e2130", bordercolor="#3d4166", borderwidth=1),
        height=400,
        bargap=0.2,
        bargroupgap=0.1,
    )
    return fig


def build_feature_modified_chart(result: dict) -> go.Figure:
    """Show original vs modified feature values after policy."""
    features = ["VehicleCount", "WrongParked", "Stopped"]
    original = [result["inputs"][f]           for f in features]
    modified = [result["modified_features"][f] for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original", x=features, y=original,
        marker_color="#ef9a9a",
        text=[str(int(v)) for v in original], textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.add_trace(go.Bar(
        name="After Policy", x=features, y=modified,
        marker_color=POLICIES[result["policy_key"]]["color"],
        text=[f"{v:.1f}" for v in modified], textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        title=dict(text="Feature Changes Applied by Policy", font=dict(size=14, color="white")),
        barmode="group",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        yaxis=dict(gridcolor="#2a2d3e"),
        legend=dict(bgcolor="#1e2130"),
        height=300,
        bargroupgap=0.15,
    )
    return fig


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.title("🚦 Traffic Policy\nSimulator")
    st.markdown("---")

    st.markdown('<div class="section-header">📥 Input Parameters</div>', unsafe_allow_html=True)

    vehicle_count = st.slider(
        "🚗 Vehicle Count",
        min_value=0, max_value=60, value=30, step=1,
        help="Total number of vehicles detected in frame"
    )
    wrong_parked = st.slider(
        "🚫 Wrong Parked Vehicles",
        min_value=0, max_value=15, value=4, step=1,
        help="Stationary vehicles near road edges (illegal parking)"
    )
    stopped = st.slider(
        "⛔ Stopped Vehicles (In Lane)",
        min_value=0, max_value=20, value=6, step=1,
        help="Stationary vehicles blocking active lanes"
    )

    st.markdown("---")
    st.markdown('<div class="section-header">📋 Select Policy</div>', unsafe_allow_html=True)

    policy_options = {key: val["label"] for key, val in POLICIES.items()}
    policy_key = st.selectbox(
        "Traffic Policy",
        options=list(policy_options.keys()),
        format_func=lambda k: policy_options[k],
    )

    st.info(POLICIES[policy_key]["description"])
    st.markdown("---")

    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📂 Scenario History</div>', unsafe_allow_html=True)

    if st.session_state.scenario_history:
        st.caption(f"{len(st.session_state.scenario_history)} scenario(s) saved")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.scenario_history = []
            st.rerun()
    else:
        st.caption("No scenarios yet. Run a simulation!")


# ════════════════════════════════════════════════════════════
#  MAIN PANEL — HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center; color:#4fc3f7; font-size:2.2rem; margin-bottom:4px;'>
    🚦 AI-Based Traffic Policy Simulation System
</h1>
<p style='text-align:center; color:#9e9e9e; font-size:0.95rem; margin-bottom:24px;'>
    Decision Support Tool for Traffic Authorities | Powered by YOLOv8 + RandomForest
</p>
""", unsafe_allow_html=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════════
with st.spinner("Loading AI model …"):
    model, scaler = load_model()
    st.session_state.model  = model
    st.session_state.scaler = scaler


# ════════════════════════════════════════════════════════════
#  LIVE METRIC PREVIEW (always shown)
# ════════════════════════════════════════════════════════════
live_score = (0.5 * vehicle_count) + (2 * wrong_parked) + (2 * stopped)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{vehicle_count}</div>
        <div class="metric-label">🚗 Vehicle Count</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{wrong_parked}</div>
        <div class="metric-label">🚫 Wrong Parked</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{stopped}</div>
        <div class="metric-label">⛔ Stopped</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#ffb74d;">{live_score:.1f}</div>
        <div class="metric-label">📊 Live Congestion Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════
#  RUN SIMULATION
# ════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Running simulation …"):
        result = simulate_policy(
            vehicle_count=vehicle_count,
            wrong_parked=wrong_parked,
            stopped=stopped,
            policy_key=policy_key,
            model=model,
            scaler=scaler,
        )
        # Add timestamp
        result["timestamp"] = time.strftime("%H:%M:%S")

        # Save to session history
        st.session_state.scenario_history.append(result)

    # ── Result Banner ──────────────────────────────────────
    reduced = result["reduction_pct"] > 0
    banner_class = "result-banner" if reduced else "result-banner-red"
    arrow = "↓" if reduced else "↑"
    st.markdown(f"""
    <div class="{banner_class}">
        <div class="result-title">
            ✅ Simulation Complete — {result['policy_label']}
        </div>
        <div class="result-subtitle" style="margin-top:8px; font-size:1rem;">
            Baseline Score: <strong>{result['baseline_score']}</strong> &nbsp;→&nbsp;
            After Policy: <strong>{result['modified_score']}</strong> &nbsp;|&nbsp;
            Congestion {arrow} <strong>{abs(result['reduction_pct']):.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts Row 1 ──────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_bar_chart(result), use_container_width=True)
    with c2:
        st.plotly_chart(build_trend_chart(result, model, scaler), use_container_width=True)

    # ── Feature modification chart ─────────────────────────
    st.plotly_chart(build_feature_modified_chart(result), use_container_width=True)

    # ── Comparison with previous ───────────────────────────
    if len(st.session_state.scenario_history) >= 2:
        st.markdown("---")
        st.markdown("### 🔄 Scenario Comparison: Current vs Previous")
        previous = st.session_state.scenario_history[-2]
        st.plotly_chart(build_comparison_chart(result, previous), use_container_width=True)

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            delta = result["reduction_pct"] - previous["reduction_pct"]
            st.metric(
                label=f"Current Reduction ({result['policy_label']})",
                value=f"{result['reduction_pct']:.1f}%",
                delta=f"{delta:+.1f}% vs previous",
            )
        with comp_col2:
            st.metric(
                label=f"Previous Reduction ({previous['policy_label']})",
                value=f"{previous['reduction_pct']:.1f}%",
            )

    st.success(f"✅ {format_result_summary(result)}")


# ════════════════════════════════════════════════════════════
#  SCENARIO HISTORY TABLE
# ════════════════════════════════════════════════════════════
if st.session_state.scenario_history:
    st.markdown("---")
    st.markdown("### 📋 Simulation History")

    history_rows = []
    for i, s in enumerate(reversed(st.session_state.scenario_history)):
        history_rows.append({
            "#"              : len(st.session_state.scenario_history) - i,
            "Time"           : s.get("timestamp", "—"),
            "Policy"         : s["policy_label"],
            "VehicleCount"   : s["inputs"]["VehicleCount"],
            "WrongParked"    : s["inputs"]["WrongParked"],
            "Stopped"        : s["inputs"]["Stopped"],
            "Baseline Score" : s["baseline_score"],
            "After Policy"   : s["modified_score"],
            "Reduction %"    : f"{s['reduction_pct']:.1f}%",
        })

    history_df = pd.DataFrame(history_rows)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── Multi-scenario trend ───────────────────────────────
    if len(st.session_state.scenario_history) >= 2:
        st.markdown("#### 📈 Reduction % Across All Scenarios")
        fig_hist = go.Figure()
        labels = [f"#{i+1} {s['policy_label'][:12]}" for i, s in enumerate(st.session_state.scenario_history)]
        values = [s["reduction_pct"] for s in st.session_state.scenario_history]
        colors = [POLICIES[s["policy_key"]]["color"] for s in st.session_state.scenario_history]

        fig_hist.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig_hist.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            yaxis=dict(title="Reduction %", gridcolor="#2a2d3e"),
            xaxis=dict(gridcolor="#2a2d3e"),
            height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  DATA EXPLORER (expandable)
# ════════════════════════════════════════════════════════════
with st.expander("📊 Dataset Explorer"):
    if st.button("Load / Regenerate Dataset"):
        generate_synthetic_dataset("dataset.csv")
        st.session_state.dataset = pd.read_csv("dataset.csv")

    if st.session_state.dataset is None and os.path.exists("dataset.csv"):
        st.session_state.dataset = pd.read_csv("dataset.csv")

    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        st.write(f"**Rows:** {len(df)} | **Columns:** {list(df.columns)}")
        st.dataframe(df.head(100), use_container_width=True)

        fig_ds = px.line(
            df, x="Timestamp", y="CongestionScore",
            title="Congestion Score Over Time",
            template="plotly_dark",
            color_discrete_sequence=["#4fc3f7"],
        )
        st.plotly_chart(fig_ds, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:0.8rem; padding:12px;'>
    🚦 AI Traffic Policy Simulation System &nbsp;|&nbsp;
    YOLOv8 + OpenCV + RandomForest + Streamlit &nbsp;|&nbsp;
    Final Year Project
</div>
""", unsafe_allow_html=True)
