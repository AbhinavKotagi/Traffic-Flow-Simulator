"""
============================================================
  MAIN APP — STREAMLIT INTERACTIVE DASHBOARD
  Traffic Policy Simulator | app.py
============================================================
Changes in this version:
  1. Groq API (groq.com) instead of xAI/Grok
  2. API key set in road_classifier.py source — not in UI
  3. No KPI cards / road profile on front page
  4. Video upload first → progress bar → then results
  5. AI-generated graph conclusions after each chart

Run:
    streamlit run app.py
============================================================
"""

import os, time, tempfile, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

from feature_extraction import generate_synthetic_dataset, aggregate_features
from road_classifier    import classify_road, generate_graph_conclusion, GROQ_API_KEY
from train_model        import (
    load_data, preprocess, train_model as _train,
    save_artifacts, run_full_pipeline,
    MODEL_PATH, SCALER_PATH, FEATURE_DEFAULTS, predict_congestion
)
from simulator import simulate_policy, simulate_trend, POLICIES, format_result

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficIQ — Policy Simulator",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  CSS — Industrial dark theme
# ─────────────────────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
    --bg-base:   #08090d;
    --bg-card:   #0f1117;
    --bg-raised: #161921;
    --border:    #252a3a;
    --accent:    #f0c040;
    --green:     #34d399;
    --red:       #f87171;
    --blue:      #60a5fa;
    --purple:    #a78bfa;
    --text-hi:   #f0f2f8;
    --text-mid:  #8b92a8;
    --text-lo:   #454d62;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}
html, body, .main, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    font-family: var(--font-head) !important;
    color: var(--text-hi) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
#MainMenu, footer { display:none !important; }
[data-testid="stDecoration"] { display:none !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width:100% !important; }
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
}
h1,h2,h3,h4 { font-family: var(--font-head) !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700 !important;
    font-family: var(--font-head) !important;
    border: none !important;
    border-radius: 6px !important;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button[kind="secondary"] {
    background: var(--bg-raised) !important;
    color: var(--text-mid) !important;
    border: 1px solid var(--border) !important;
}

/* Sidebar */
.sidebar-logo {
    display:flex; align-items:center; gap:10px;
    padding: 1rem 1.1rem 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.8rem;
}
.sidebar-logo-text { font-size:1.2rem; font-weight:800; color:var(--accent); }
.sidebar-logo-sub  { font-size:0.65rem; color:var(--text-mid); font-family:var(--font-mono); }
.sidebar-section {
    font-size:0.62rem; font-family:var(--font-mono);
    color:var(--text-lo); letter-spacing:2px; text-transform:uppercase;
    padding: 0.5rem 1.1rem 0.2rem;
    border-top: 1px solid var(--border);
    margin-top: 0.6rem;
}

/* Section headers */
.sec-header {
    font-size:0.62rem; font-family:var(--font-mono);
    color:var(--text-lo); letter-spacing:2.5px; text-transform:uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom:0.35rem; margin: 1.2rem 0 0.8rem;
}

/* Result banner */
.result-banner {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-left: 4px solid var(--green);
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin: 0.8rem 0;
}
.result-banner.warn { border-left-color: var(--red); }
.result-title { font-size:1.25rem; font-weight:800; color:var(--text-hi); }
.result-meta  { font-size:0.82rem; color:var(--text-mid); font-family:var(--font-mono); margin-top:0.3rem; }

/* Conclusion box */
.conclusion-box {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-top: 0.6rem;
}
.conclusion-label {
    font-size:0.6rem; font-family:var(--font-mono);
    color:var(--accent); letter-spacing:2px;
    text-transform:uppercase; margin-bottom:0.5rem;
}

/* History item */
.hist-item {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius:6px; padding:0.55rem 0.85rem;
    font-size:0.76rem; font-family:var(--font-mono);
    color:var(--text-mid); margin-bottom:0.4rem;
}
.hist-item .tag {
    display:inline-block; background:var(--bg-base);
    border-radius:3px; padding:1px 5px;
    color:var(--accent); font-size:0.65rem; margin-right:4px;
}

/* Selectbox / slider overrides */
.stSelectbox > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius:6px !important;
}
[data-testid="stSlider"] > div > div > div { background: var(--border) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; }

/* Upload zone */
.stFileUploader > div {
    background: var(--bg-raised) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
</style>
""")

# ─────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────
for k, v in {
    "scenario_history" : [],
    "model"            : None,
    "scaler"           : None,
    "video_features"   : None,
    "road_profile"     : None,
    "dataset"          : None,
    "last_result"      : None,
    "extraction_done"  : False,
    "conclusions"      : {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────
#  CHART LAYOUT DEFAULTS
# ─────────────────────────────────────────────────────────
_CL = dict(
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    font=dict(family="JetBrains Mono, monospace", color="#8b92a8", size=11),
    margin=dict(l=50, r=20, t=50, b=40),
    xaxis=dict(gridcolor="#1d2130", zerolinecolor="#252a3a"),
    yaxis=dict(gridcolor="#1d2130", zerolinecolor="#252a3a"),
    legend=dict(bgcolor="rgba(15,17,23,0.9)", bordercolor="#252a3a", borderwidth=1),
    hoverlabel=dict(bgcolor="#161921", bordercolor="#334", font_color="#f0f2f8"),
)

# ─────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_or_train_model():
    """Load saved model or run full dual-source pipeline on first launch."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        except Exception:
            pass
    # First launch: train on knowledge dataset (video data added later)
    m, sc, _ = run_full_pipeline("road_knowledge.csv", "dataset.csv")
    return m, sc

# ─────────────────────────────────────────────────────────
#  CHART BUILDERS
# ─────────────────────────────────────────────────────────
def chart_before_after(result):
    pc = result["policy_color"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline", x=["Baseline"], y=[result["baseline_score"]],
        marker=dict(color="#f87171", line=dict(color="#fca5a5", width=1.5)),
        width=0.4,
        text=[f"{result['baseline_score']:.1f}"],
        textposition="outside", textfont=dict(color="#f0f2f8", size=13),
    ))
    fig.add_trace(go.Bar(
        name="After Policy", x=["After Policy"], y=[result["modified_score"]],
        marker=dict(color=pc),
        width=0.4,
        text=[f"{result['modified_score']:.1f}"],
        textposition="outside", textfont=dict(color="#f0f2f8", size=13),
    ))
    fig.update_layout(
        **_CL,
        title=dict(text=f"<b>Congestion — Before vs After</b><br><sup>{result['policy_label']}</sup>",
                   font=dict(color="#f0f2f8", size=14, family="Syne")),
        height=380, showlegend=True, bargap=0.4,
    )
    fig.update_yaxes(title_text="Congestion Score",
                     range=[0, max(result["baseline_score"], 1) * 1.35])
    return fig


def chart_trend(result, model, scaler):
    df = simulate_trend(result["inputs"], result["policy_key"], model, scaler)
    pc = result["policy_color"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["VehicleCount"], y=df["Baseline"],
        name="Baseline", mode="lines",
        line=dict(color="#f87171", width=2.5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=df["VehicleCount"], y=df["AfterPolicy"],
        name=result["policy_label"], mode="lines",
        line=dict(color=pc, width=2.5, dash="dash"),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.05)",
    ))
    cur_vc = result["inputs"].get("VehicleCount", 30)
    fig.add_vline(x=cur_vc, line_dash="dot", line_color="#f0c040", line_width=1.5,
                  annotation_text=f" VC={int(cur_vc)}",
                  annotation_font=dict(color="#f0c040", size=10))
    fig.update_layout(
        **_CL,
        title=dict(text="<b>Congestion Score vs Vehicle Count — Trend</b>",
                   font=dict(color="#f0f2f8", size=14, family="Syne")),
        height=380,
    )
    fig.update_xaxes(title_text="Vehicle Count")
    fig.update_yaxes(title_text="Congestion Score")
    return fig


def chart_features_modified(result):
    feats  = ["VehicleCount", "Stopped", "WrongParked", "EstSpeed"]
    labels = ["Vehicle Count", "Stopped", "Wrong Parked", "Est. Speed (km/h)"]
    orig   = [result["inputs"].get(f, 0) for f in feats]
    modf   = [result["modified_features"].get(f, orig[i]) for i, f in enumerate(feats)]
    pc     = result["policy_color"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original", x=labels, y=orig, marker=dict(color="#64748b"),
        text=[f"{v:.1f}" for v in orig], textposition="outside",
        textfont=dict(color="#8b92a8", size=10),
    ))
    fig.add_trace(go.Bar(
        name="After Policy", x=labels, y=modf, marker=dict(color=pc, opacity=0.9),
        text=[f"{v:.1f}" for v in modf], textposition="outside",
        textfont=dict(color="#f0f2f8", size=10),
    ))
    fig.update_layout(
        **_CL,
        title=dict(text="<b>Feature Changes Applied by Policy</b>",
                   font=dict(color="#f0f2f8", size=14, family="Syne")),
        barmode="group", height=350, bargroupgap=0.12,
    )
    return fig


def chart_scenario_comparison(current, previous):
    cats   = ["Congestion Before", "Congestion After", "Speed Before", "Speed After"]
    prev_v = [previous["baseline_score"], previous["modified_score"],
               previous["baseline_speed"],  previous["modified_speed"]]
    curr_v = [current["baseline_score"],  current["modified_score"],
               current["baseline_speed"],   current["modified_speed"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Previous: {previous['policy_label']}", x=cats, y=prev_v,
        marker=dict(color="#475569"),
        text=[f"{v:.1f}" for v in prev_v], textposition="outside",
        textfont=dict(color="#94a3b8", size=10),
    ))
    fig.add_trace(go.Bar(
        name=f"Current: {current['policy_label']}", x=cats, y=curr_v,
        marker=dict(color=current["policy_color"]),
        text=[f"{v:.1f}" for v in curr_v], textposition="outside",
        textfont=dict(color="#f0f2f8", size=10),
    ))
    fig.update_layout(
        **_CL,
        title=dict(text="<b>Scenario Comparison — Current vs Previous</b>",
                   font=dict(color="#f0f2f8", size=14, family="Syne")),
        barmode="group", height=400, bargroupgap=0.1,
    )
    return fig


def chart_history_reduction(history):
    labels = [f"#{i+1} {s['policy_label'][:14]}" for i, s in enumerate(history)]
    values = [s["reduction_pct"] for s in history]
    colors = [POLICIES[s["policy_key"]]["color"] for s in history]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside", textfont=dict(color="#f0f2f8", size=11),
    ))
    fig.update_layout(
        **_CL,
        title=dict(text="<b>Congestion Reduction % — All Scenarios</b>",
                   font=dict(color="#f0f2f8", size=14, family="Syne")),
        height=300,
    )
    fig.update_yaxes(title_text="Reduction %")
    return fig


# ─────────────────────────────────────────────────────────
#  CONCLUSION RENDERER
# ─────────────────────────────────────────────────────────
def render_conclusion(graph_type: str, result: dict, cache_key: str):
    """Render AI conclusion box below a chart, with caching."""
    key = f"{cache_key}_{graph_type}"
    if key not in st.session_state.conclusions:
        groq_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        with st.spinner("Generating AI analysis …"):
            text = generate_graph_conclusion(graph_type, result, groq_key)
        st.session_state.conclusions[key] = text

    text = st.session_state.conclusions[key]
    # Render bullet points as styled markdown inside a box
    st.markdown(f"""
    <div class="conclusion-box">
        <div class="conclusion-label">📊 AI Analysis & Conclusion</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(text)


# ═════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span style="font-size:1.5rem;">🚦</span>
        <div>
            <div class="sidebar-logo-text">TrafficIQ</div>
            <div class="sidebar-logo-sub">POLICY SIMULATION SYSTEM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🎛 Simulation Inputs</div>', unsafe_allow_html=True)

    defaults = st.session_state.video_features or {}
    vehicle_count = st.slider("🚗 Vehicle Count",          0, 80,  int(defaults.get("VehicleCount", 30)), 1)
    stopped       = st.slider("⛔ Stopped In Lane",         0, 25,  int(defaults.get("Stopped",       6)), 1)
    wrong_parked  = st.slider("🚫 Wrong Parked",            0, 15,  int(defaults.get("WrongParked",    3)), 1)
    est_speed     = st.slider("⚡ Est. Speed (km/h)",       5, 80,  int(defaults.get("EstSpeed",      25)), 1)
    lanes         = st.slider("🛣  Lanes",                  1, 6,   int(defaults.get("Lanes",          3)), 1)

    st.markdown('<div class="sidebar-section">📋 Policy</div>', unsafe_allow_html=True)
    policy_key = st.selectbox(
        "Traffic Policy",
        list(POLICIES.keys()),
        format_func=lambda k: f"{POLICIES[k]['icon']}  {POLICIES[k]['label']}"
    )
    st.info(POLICIES[policy_key]["description"], icon="ℹ️")

    simulate_btn = st.button("▶  Run Simulation", use_container_width=True, type="primary")

    st.markdown('<div class="sidebar-section">📂 Scenarios</div>', unsafe_allow_html=True)
    n_hist = len(st.session_state.scenario_history)
    if n_hist:
        st.caption(f"{n_hist} scenario(s) saved")
        for i, s in enumerate(reversed(st.session_state.scenario_history[-4:])):
            idx = n_hist - i
            st.markdown(f"""<div class="hist-item">
                <span class="tag">#{idx}</span>{s['policy_label'][:18]}<br>
                <span style="color:#34d399;">▼{s['reduction_pct']:.1f}%</span>
                &nbsp;|&nbsp;VC={int(s['inputs']['VehicleCount'])}
            </div>""", unsafe_allow_html=True)
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.scenario_history = []
            st.session_state.conclusions = {}
            st.rerun()
    else:
        st.caption("No scenarios yet.")

# ═════════════════════════════════════════════════════════
#  LOAD MODEL (silently)
# ═════════════════════════════════════════════════════════
model, scaler = _load_or_train_model()
st.session_state.model  = model
st.session_state.scaler = scaler

# ═════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════
st.markdown("""
<div style="border-bottom:1px solid #252a3a; padding-bottom:0.8rem; margin-bottom:1.4rem;">
    <h1 style="font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800;
               color:#f0f2f8; margin:0; letter-spacing:-1px;">
        🚦 Traffic Policy Simulation System
    </h1>
    <p style="color:#8b92a8; font-family:'JetBrains Mono',monospace;
              font-size:0.75rem; margin:0.25rem 0 0;">
        YOLOv8 + OpenCV  →  Groq AI Classifier  →  RandomForest  →  Policy Engine
    </p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
#  STEP 1 — VIDEO UPLOAD & EXTRACTION
# ═════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">01 — UPLOAD & ANALYZE VIDEO</div>', unsafe_allow_html=True)

upload_col, info_col = st.columns([2, 1])

with upload_col:
    uploaded_video = st.file_uploader(
        "Upload a traffic video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Recommended: 2–3 minutes, static camera angle"
    )
    use_demo = st.checkbox(
        "Use synthetic demo data instead (no video needed)",
        value=not bool(uploaded_video),
        help="Generates realistic time-of-day traffic patterns"
    )

with info_col:
    st.markdown("""
    <div style="background:#161921;border:1px solid #252a3a;border-radius:8px;
         padding:1rem;font-size:0.78rem;color:#8b92a8;font-family:'JetBrains Mono',monospace;
         margin-top:0.5rem;">
        <b style="color:#f0c040;">📌 What happens:</b><br><br>
        1. YOLOv8 detects vehicles<br>
        2. Optical flow estimates speed<br>
        3. Hough lines count lanes<br>
        4. Groq AI classifies road<br>
        5. ML model trains on data<br>
        6. Simulation engine ready
    </div>
    """, unsafe_allow_html=True)

extract_btn = st.button("🔍 Extract Features & Analyze", type="primary")

# ── Processing with progress bar ──────────────────────────
if extract_btn:
    if not uploaded_video and not use_demo:
        st.warning("⚠️ Please upload a video or enable demo mode.")
    else:
        progress_bar = st.progress(0)
        status_text  = st.empty()

        try:
            if uploaded_video:
                # ── Step 1: Save temp file ─────────────────
                status_text.markdown(
                    "<span style='font-family:JetBrains Mono;font-size:0.82rem;color:#8b92a8;'>"
                    "⟳ Saving uploaded video …</span>", unsafe_allow_html=True)
                progress_bar.progress(5)
                suffix = os.path.splitext(uploaded_video.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_video.read())
                    tmp_path = tmp.name
                size_mb = os.path.getsize(tmp_path) / 1024 / 1024
                progress_bar.progress(15)

                # ── Step 2: YOLO extraction ────────────────
                status_text.markdown(
                    f"<span style='font-family:JetBrains Mono;font-size:0.82rem;color:#8b92a8;'>"
                    f"⟳ Running YOLOv8 on {size_mb:.1f} MB video …</span>", unsafe_allow_html=True)

                try:
                    from feature_extraction import extract_features_from_video

                    # Patch extraction to report progress
                    import cv2
                    cap   = cv2.VideoCapture(tmp_path)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    cap.release()

                    # Run in a way that we can update progress
                    # We do batched processing simulation via progress updates
                    for pct in range(15, 75, 5):
                        progress_bar.progress(pct)
                        time.sleep(0.05)

                    df = extract_features_from_video(tmp_path, "dataset.csv")
                    os.unlink(tmp_path)
                except Exception as e:
                    st.warning(f"YOLOv8 not available ({e}). Using synthetic data fallback.")
                    df = generate_synthetic_dataset("dataset.csv")

            else:
                # ── Demo mode ─────────────────────────────
                status_text.markdown(
                    "<span style='font-family:JetBrains Mono;font-size:0.82rem;color:#8b92a8;'>"
                    "⟳ Generating synthetic traffic dataset …</span>", unsafe_allow_html=True)
                for pct in range(5, 70, 8):
                    progress_bar.progress(pct)
                    time.sleep(0.04)
                df = generate_synthetic_dataset("dataset.csv")

            progress_bar.progress(75)

            # ── Step 3: Aggregate features ─────────────────
            status_text.markdown(
                "<span style='font-family:JetBrains Mono;font-size:0.82rem;color:#8b92a8;'>"
                "⟳ Aggregating traffic features …</span>", unsafe_allow_html=True)
            agg = aggregate_features(df)
            st.session_state.dataset        = df
            st.session_state.video_features  = agg
            progress_bar.progress(82)

            # ── Step 4: Road classification ────────────────
            status_text.markdown(
                "<span style='font-family:JetBrains Mono;font-size:0.82rem;color:#8b92a8;'>"
                "⟳ Classifying road type with Groq AI …</span>", unsafe_allow_html=True)
            groq_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
            profile  = classify_road(agg, groq_key or None)
            st.session_state.road_profile   = profile
            progress_bar.progress(92)

            # ── Step 5: Dual-source training pipeline ────
            status_text.markdown(
                "<span style='font-family:JetBrains Mono;font-size:0.82rem;color:#8b92a8;'>"
                "⟳ Training ensemble model (knowledge + video data) …</span>",
                unsafe_allow_html=True)
            m, sc, metrics = run_full_pipeline(
                knowledge_csv="road_knowledge.csv",
                video_csv="dataset.csv"
            )
            st.session_state.model   = m
            st.session_state.scaler  = sc
            st.session_state.metrics = metrics
            progress_bar.progress(100)

            status_text.markdown(
                "<span style='font-family:JetBrains Mono;font-size:0.82rem;color:#34d399;'>"
                f"✅ Done — {len(df)} frame records processed.</span>", unsafe_allow_html=True)
            st.session_state.extraction_done = True

        except Exception as e:
            progress_bar.progress(0)
            st.error(f"❌ Processing error: {e}")
            st.session_state.extraction_done = False


# ═════════════════════════════════════════════════════════
#  STEP 2 — RESULTS (only shown after extraction)
# ═════════════════════════════════════════════════════════
if st.session_state.extraction_done:

    st.markdown('<div class="sec-header">02 — SIMULATION</div>', unsafe_allow_html=True)

    # ── Run simulation ────────────────────────────────────
    if simulate_btn:
        base = dict(st.session_state.video_features or {})
        base.update({
            "VehicleCount": vehicle_count,
            "Stopped"     : stopped,
            "WrongParked" : wrong_parked,
            "EstSpeed"    : est_speed,
            "Lanes"       : lanes,
            "Density"     : round(vehicle_count / max(lanes, 1), 2),
        })
        result = simulate_policy(base, policy_key,
                                 st.session_state.model,
                                 st.session_state.scaler)
        result["timestamp"] = time.strftime("%H:%M:%S")
        st.session_state.scenario_history.append(result)
        st.session_state.last_result = result

    result = st.session_state.last_result

    if result is None:
        st.info("👈 Adjust sliders in the sidebar and click **▶ Run Simulation** to begin.")
    else:
        # ── Result Banner ──────────────────────────────────
        reduced = result["reduction_pct"] > 0
        banner_class = "result-banner" if reduced else "result-banner warn"
        arrow = "▼" if reduced else "▲"
        st.markdown(f"""
        <div class="{banner_class}">
            <div class="result-title">
                {result['policy_icon']} &nbsp; {result['policy_label']}
            </div>
            <div class="result-meta" style="margin-top:0.4rem;">
                Congestion: <b>{result['baseline_score']:.1f}</b> → <b>{result['modified_score']:.1f}</b>
                &nbsp;|&nbsp; {arrow} <b>{abs(result['reduction_pct']):.1f}%</b> reduction
                &nbsp;|&nbsp; Speed: {result['baseline_speed']:.0f} → {result['modified_speed']:.0f} km/h
                &nbsp;|&nbsp; ↑ <b>{result['speed_improvement_pct']:.1f}%</b> speed gain
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Cache key for conclusions (unique per scenario)
        cache_key = f"{result['policy_key']}_{result['baseline_score']}_{result['modified_score']}"

        # ── Simulation output code block ───────────────────
        with st.expander("📋 Raw Simulation Output"):
            st.code(f"""Policy            : {result['policy_label']}
Congestion Before : {result['baseline_score']:.2f}
Congestion After  : {result['modified_score']:.2f}
Congestion Reduced: {result['reduction_pct']:.1f}%
Speed Before      : {result['baseline_speed']:.1f} km/h
Speed After       : {result['modified_speed']:.1f} km/h
Speed Improved    : {result['speed_improvement_pct']:.1f}%
Modified Features : {result['modified_features']}""", language="yaml")

        # ═══════════════════════════════════════════════════
        #  VISUALISATIONS — each chart + conclusion below it
        # ═══════════════════════════════════════════════════
        st.markdown('<div class="sec-header">03 — VISUALISATIONS & AI ANALYSIS</div>',
                    unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Before vs After",
            "📈 Trend Graph",
            "🔧 Feature Changes",
            "🔄 Scenario Compare",
        ])

        # ── Tab 1: Before vs After ─────────────────────────
        with tab1:
            st.plotly_chart(chart_before_after(result), use_container_width=True, key=f"chart_before_after_{result['policy_key']}")
            render_conclusion("before_after", result, cache_key)

        # ── Tab 2: Trend Graph ─────────────────────────────
        with tab2:
            st.plotly_chart(chart_trend(result, st.session_state.model,
                                        st.session_state.scaler), use_container_width=True, key=f"chart_trend_{result['policy_key']}")
            render_conclusion("trend", result, cache_key)

        # ── Tab 3: Feature Changes ─────────────────────────
        with tab3:
            st.plotly_chart(chart_features_modified(result), use_container_width=True, key=f"chart_features_{result['policy_key']}")
            render_conclusion("feature_changes", result, cache_key)

        # ── Tab 4: Scenario Comparison ────────────────────
        with tab4:
            history = st.session_state.scenario_history
            if len(history) >= 2:
                prev = history[-2]
                st.plotly_chart(chart_scenario_comparison(result, prev),
                                use_container_width=True, key=f"chart_compare_latest_{result['policy_key']}")
                render_conclusion("scenario_compare", result, cache_key)

                col_a, col_b = st.columns(2)
                with col_a:
                    delta = result["reduction_pct"] - prev["reduction_pct"]
                    st.metric(f"Current — {result['policy_label'][:22]}",
                              f"{result['reduction_pct']:.1f}%",
                              delta=f"{delta:+.1f}% vs previous")
                with col_b:
                    st.metric(f"Previous — {prev['policy_label'][:22]}",
                              f"{prev['reduction_pct']:.1f}%")
            else:
                st.info("Run at least **2 simulations** to see a scenario comparison.")

        # ═══════════════════════════════════════════════════
        #  HISTORY
        # ═══════════════════════════════════════════════════
        history = st.session_state.scenario_history
        if len(history) >= 1:
            st.markdown('<div class="sec-header">04 — SCENARIO HISTORY</div>',
                        unsafe_allow_html=True)
            rows = []
            for i, s in enumerate(history):
                rows.append({
                    "#"              : i + 1,
                    "Time"           : s.get("timestamp", "—"),
                    "Policy"         : s["policy_label"],
                    "VehicleCount"   : int(s["inputs"]["VehicleCount"]),
                    "Stopped"        : int(s["inputs"]["Stopped"]),
                    "Baseline Score" : s["baseline_score"],
                    "After Policy"   : s["modified_score"],
                    "Congestion ▼ %" : f"{s['reduction_pct']:.1f}%",
                    "Speed ↑ %"      : f"{s['speed_improvement_pct']:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if len(history) >= 2:
                st.plotly_chart(chart_history_reduction(history), use_container_width=True, key="chart_history")

                st.markdown("##### 🔍 Custom Scenario Comparison")
                labels = [f"#{i+1} — {s['policy_label']} (VC={int(s['inputs']['VehicleCount'])})"
                          for i, s in enumerate(history)]
                col_a, col_b = st.columns(2)
                with col_a:
                    idx_a = st.selectbox("Scenario A", range(len(history)),
                                         index=len(history)-1,
                                         format_func=lambda i: labels[i], key="cmp_a")
                with col_b:
                    idx_b = st.selectbox("Scenario B", range(len(history)),
                                         index=max(0, len(history)-2),
                                         format_func=lambda i: labels[i], key="cmp_b")
                if idx_a != idx_b:
                    st.plotly_chart(
                        chart_scenario_comparison(history[idx_a], history[idx_b]),
                        use_container_width=True,
                        key=f"chart_custom_{idx_a}_{idx_b}"
                    )
                else:
                    st.warning("Select two different scenarios.")

        # ── Dataset explorer ──────────────────────────────
        with st.expander("📊 Raw Dataset Explorer"):
            ds = st.session_state.dataset
            if ds is not None:
                st.caption(f"{len(ds)} rows | {len(ds.columns)} columns")
                st.dataframe(ds.head(200), use_container_width=True)
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=ds["Timestamp"], y=ds["CongestionScore"],
                    mode="lines", line=dict(color="#f0c040", width=1.5),
                    fill="tozeroy", fillcolor="rgba(240,192,64,0.06)",
                    name="CongestionScore",
                ))
                fig_ts.update_layout(
                    **_CL, height=280,
                    title=dict(text="<b>Congestion Score Over Time</b>",
                               font=dict(color="#f0f2f8", size=13, family="Syne")),
                )
                fig_ts.update_xaxes(title_text="Timestamp (hours)")
                fig_ts.update_yaxes(title_text="Congestion Score")
                st.plotly_chart(fig_ts, use_container_width=True, key="chart_timeseries")

# ═════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════
st.markdown("""
<div style="border-top:1px solid #252a3a; margin-top:2rem; padding-top:1rem;
     text-align:center; font-family:'JetBrains Mono'; font-size:0.68rem; color:#454d62;">
    TrafficIQ &nbsp;·&nbsp; YOLOv8 + OpenCV + Groq AI + RandomForest + Streamlit
    &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp; Traffic Decision-Support System
</div>
""", unsafe_allow_html=True)
