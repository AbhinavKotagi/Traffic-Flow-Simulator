"""
============================================================
  TRAFFICIQ BANGALORE — Policy Simulation Dashboard
  app.py | Streamlit UI
============================================================
Location-aware traffic policy simulator for Bangalore.
Powered by real Bangalore traffic data (2022–2024).

Run:  streamlit run app.py
============================================================
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

from data_processor  import (
    load_and_engineer, get_location_stats, get_area_summary,
    get_historical_trend, get_weather_impact,
    AREA_LIST, AREA_ROADS, ROAD_TYPE_MAP, ROAD_TYPE_LABELS,
    FEATURE_COLS, TARGET_COL, FEATURE_DEFAULTS, DATA_PATH,
    WEATHER_CODES, WEATHER_IMPACT,
)
from train_model     import (
    run_full_pipeline, load_artifacts, predict_congestion,
    MODEL_PATH, SCALER_PATH,
)
from simulator       import (
    simulate_policy, simulate_trend, simulate_combined, POLICIES,
)
from road_classifier import generate_conclusion

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficIQ Bangalore",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────
import streamlit.components.v1 as _stc
_stc.html("""
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
.block-container { padding: 1.4rem 2rem 3rem !important; max-width:100% !important; }
[data-testid="collapsedControl"] { display:flex !important; visibility:visible !important; }
h1,h2,h3,h4 { font-family: var(--font-head) !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important; color: #000 !important;
    font-weight: 700 !important; font-family: var(--font-head) !important;
    border: none !important; border-radius: 6px !important;
}
.stButton > button:hover { opacity:0.85 !important; }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:0.7rem; margin:0.8rem 0; }
.kpi-card {
    background: var(--bg-raised); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.85rem 1rem;
}
.kpi-val  { font-size:1.6rem; font-weight:800; font-family:var(--font-head); }
.kpi-lbl  { font-size:0.65rem; font-family:var(--font-mono); color:var(--text-mid);
            letter-spacing:1.5px; text-transform:uppercase; margin-top:0.15rem; }
.kpi-sub  { font-size:0.62rem; font-family:var(--font-mono); color:var(--text-lo); margin-top:0.1rem; }
.c-green  { color: var(--green); }
.c-red    { color: var(--red); }
.c-blue   { color: var(--blue); }
.c-yellow { color: var(--accent); }
.c-purple { color: var(--purple); }
.c-white  { color: var(--text-hi); }

/* Section headers */
.sec-header {
    font-size:0.6rem; font-family:var(--font-mono); color:var(--text-lo);
    letter-spacing:2.5px; text-transform:uppercase;
    border-bottom:1px solid var(--border); padding-bottom:0.3rem; margin:1.1rem 0 0.7rem;
}

/* Location badge */
.loc-badge {
    display:inline-flex; align-items:center; gap:8px;
    background: var(--bg-raised); border:1px solid var(--border);
    border-left:3px solid var(--accent); border-radius:6px;
    padding:0.55rem 1rem; font-size:0.8rem; margin-bottom:0.8rem;
}
.loc-area { color: var(--accent); font-weight:700; }
.loc-road { color: var(--text-hi); }
.loc-type { color: var(--text-mid); font-family:var(--font-mono); font-size:0.68rem; }

/* Result banners */
.result-banner {
    background: var(--bg-raised); border:1px solid var(--border);
    border-left:4px solid var(--green); border-radius:8px;
    padding:1rem 1.3rem; margin:0.7rem 0;
}
.result-banner.warn { border-left-color: var(--red); }
.result-title { font-size:1.2rem; font-weight:800; }
.result-meta  { font-size:0.8rem; font-family:var(--font-mono); color:var(--text-mid); margin-top:0.3rem; }

/* Conclusion box */
.conclusion-box {
    background: var(--bg-raised); border:1px solid var(--border);
    border-left:3px solid var(--accent); border-radius:6px;
    padding:0.9rem 1.1rem; margin-top:0.5rem;
}
.conclusion-lbl {
    font-size:0.58rem; font-family:var(--font-mono); color:var(--accent);
    letter-spacing:2px; text-transform:uppercase; margin-bottom:0.4rem;
}

/* Info cards */
.info-card {
    background: var(--bg-raised); border:1px solid var(--border);
    border-radius:8px; padding:0.9rem 1rem; font-size:0.78rem;
    color: var(--text-mid); font-family:var(--font-mono);
}
.info-card b { color:var(--accent); }

/* Sidebar */
.sb-section {
    font-size:0.58rem; font-family:var(--font-mono); color:var(--text-lo);
    letter-spacing:2px; text-transform:uppercase;
    border-top:1px solid var(--border); padding:0.5rem 0 0.15rem;
    margin-top:0.5rem;
}
</style>
""", height=0)

# ─────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────
for k, v in {
    "model"        : None,
    "scaler"       : None,
    "df"           : None,
    "last_results" : None,
    "conclusions"  : {},
    "history"      : [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────
#  CHART LAYOUT
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
def _load_model_and_data():
    df = load_and_engineer(DATA_PATH)
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model, scaler = load_artifacts()
            return model, scaler, df
        except Exception:
            pass
    model, scaler, _, _ = run_full_pipeline(DATA_PATH)
    return model, scaler, df

# ─────────────────────────────────────────────────────────
#  CONCLUSION RENDERER
# ─────────────────────────────────────────────────────────
def render_conclusion(graph_type, result, area, road, stats, cache_key):
    key = f"{cache_key}_{graph_type}"
    if key not in st.session_state.conclusions:
        with st.spinner("Generating AI analysis …"):
            text = generate_conclusion(graph_type, result, area, road, stats)
        st.session_state.conclusions[key] = text

    raw  = st.session_state.conclusions[key]

    # Split on bullet markers and render each point on its own line
    import re as _re
    # Normalise various bullet styles (•, -, *, 1.) into a clean list
    lines = _re.split(r'\n+', raw.strip())
    bullets = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # Strip leading bullet/number markers
        ln = _re.sub(r'^[•\-\*\d\.]+\s*', '', ln).strip()
        if ln:
            bullets.append(ln)

    st.markdown('<div class="conclusion-box"><div class="conclusion-lbl">📊 AI Analysis</div>',
                unsafe_allow_html=True)
    for b in bullets:
        st.markdown(f"• {b}")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────
def chart_policy_compare(all_results):
    labels   = [v["policy_icon"] + " " + v["policy_label"] for v in all_results.values()]
    baselines= [v["baseline_score"]  for v in all_results.values()]
    afters   = [v["modified_score"]  for v in all_results.values()]
    colors   = [v["policy_color"]    for v in all_results.values()]
    reductions=[v["reduction_pct"]   for v in all_results.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Baseline", x=labels, y=baselines,
                         marker_color="#475569",
                         text=[f"{v:.1f}" for v in baselines],
                         textposition="outside", textfont=dict(color="#94a3b8", size=10)))
    fig.add_trace(go.Bar(name="After Policy", x=labels, y=afters,
                         marker_color=colors,
                         text=[f"{v:.1f}" for v in afters],
                         textposition="outside", textfont=dict(color="#f0f2f8", size=10)))
    fig.update_layout(**_CL, barmode="group", height=380,
                      title=dict(text="<b>All 4 Policies — Congestion Before vs After</b>",
                                 font=dict(color="#f0f2f8", size=14, family="Syne")))
    fig.update_yaxes(title_text="Congestion Level (%)", range=[0, 115])
    return fig


def chart_reduction_bar(all_results):
    labels = [v["policy_icon"] + " " + v["policy_label"] for v in all_results.values()]
    reds   = [v["reduction_pct"]  for v in all_results.values()]
    speeds = [v["speed_gain_pct"] for v in all_results.values()]
    colors = [v["policy_color"]   for v in all_results.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Congestion Reduction %", x=labels, y=reds,
                         marker_color=colors,
                         text=[f"▼{v:.1f}%" for v in reds],
                         textposition="outside", textfont=dict(color="#f0f2f8", size=11)))
    fig.add_trace(go.Scatter(name="Speed Gain %", x=labels, y=speeds,
                             mode="lines+markers",
                             line=dict(color="#f0c040", width=2.5),
                             marker=dict(size=8, color="#f0c040"),
                             yaxis="y2"))
    fig.update_layout(**_CL, height=360,
                      title=dict(text="<b>Congestion Reduction % & Speed Gain by Policy</b>",
                                 font=dict(color="#f0f2f8", size=14, family="Syne")),
                      yaxis2=dict(overlaying="y", side="right",
                                  title="Speed Gain %",
                                  gridcolor="#1d2130", color="#8b92a8"))
    fig.update_yaxes(title_text="Congestion Reduction %")
    return fig


def chart_trend(result, model, scaler, road_name):
    df_t   = simulate_trend(result["inputs"], result["policy_key"], model, scaler, road_name)
    pc     = result["policy_color"]
    cur_vol= result["inputs"].get("TrafficVolume", 28000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t["TrafficVolume"], y=df_t["Baseline"],
                             name="Baseline", mode="lines",
                             line=dict(color="#f87171", width=2.5),
                             fill="tozeroy", fillcolor="rgba(248,113,113,0.05)"))
    fig.add_trace(go.Scatter(x=df_t["TrafficVolume"], y=df_t["AfterPolicy"],
                             name=result["policy_label"], mode="lines",
                             line=dict(color=pc, width=2.5, dash="dash"),
                             fill="tozeroy", fillcolor="rgba(96,165,250,0.04)"))
    fig.add_vline(x=cur_vol, line_dash="dot", line_color="#f0c040", line_width=1.5,
                  annotation_text=f" Current vol={int(cur_vol):,}",
                  annotation_font=dict(color="#f0c040", size=10))
    fig.update_layout(**_CL, height=370,
                      title=dict(text="<b>Congestion vs Traffic Volume — Policy Trend</b>",
                                 font=dict(color="#f0f2f8", size=14, family="Syne")))
    fig.update_xaxes(title_text="Traffic Volume (vehicles/day)")
    fig.update_yaxes(title_text="Congestion Level (%)")
    return fig


def chart_historical(trend_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["CongestionLevel"],
                             mode="lines", name="Congestion %",
                             line=dict(color="#f0c040", width=1.5),
                             fill="tozeroy", fillcolor="rgba(240,192,64,0.06)"))
    fig.update_layout(**_CL, height=320,
                      title=dict(text="<b>Historical Congestion Trend</b>",
                                 font=dict(color="#f0f2f8", size=14, family="Syne")))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Congestion Level (%)", range=[0, 110])
    return fig


def chart_weather(weather_data: dict):
    weather_order = ["Clear", "Overcast", "Windy", "Fog", "Rain"]
    labels  = [w for w in weather_order if w in weather_data]
    values  = [weather_data[w] for w in labels]
    w_colors= {"Clear": "#34d399", "Overcast": "#8b92a8",
                "Windy": "#60a5fa", "Fog": "#a78bfa", "Rain": "#f87171"}
    colors  = [w_colors.get(w, "#8b92a8") for w in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=[f"{v:.1f}%" for v in values],
                           textposition="outside",
                           textfont=dict(color="#f0f2f8", size=11)))
    fig.update_layout(**_CL, height=310,
                      title=dict(text="<b>Avg Congestion by Weather Condition</b>",
                                 font=dict(color="#f0f2f8", size=14, family="Syne")))
    fig.update_yaxes(title_text="Avg Congestion Level (%)", range=[0, 115])
    return fig


def chart_area_heatmap(df):
    pivot = (df.groupby(["AreaName", "RoadName"])["CongestionLevel"]
               .mean().round(1).reset_index()
               .pivot(index="AreaName", columns="RoadName", values="CongestionLevel"))
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                    text_auto=True, aspect="auto",
                    labels=dict(color="Congestion %"))
    fig.update_layout(**_CL, height=380,
                      title=dict(text="<b>Bangalore — Congestion Heatmap by Area & Road</b>",
                                 font=dict(color="#f0f2f8", size=14, family="Syne")),
                      coloraxis_colorbar=dict(tickfont=dict(color="#8b92a8")))
    fig.update_xaxes(tickangle=-30)
    return fig


def chart_feature_impact(result):
    feats  = ["TrafficVolume", "AverageSpeed", "RoadCapacityUtil",
              "ParkingUsage", "SignalCompliance", "PublicTransportUsage"]
    labels = ["Traffic\nVolume", "Avg Speed\n(km/h)", "Capacity\nUtil %",
              "Parking\nUsage %", "Signal\nCompliance %", "Public\nTransport %"]
    orig   = [result["inputs"].get(f, FEATURE_DEFAULTS.get(f, 0)) for f in feats]
    modf   = [result["modified_features"].get(f, orig[i]) for i, f in enumerate(feats)]
    pc     = result["policy_color"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=labels, y=orig, marker_color="#64748b",
                         text=[f"{v:.0f}" for v in orig], textposition="outside",
                         textfont=dict(color="#8b92a8", size=9)))
    fig.add_trace(go.Bar(name="After", x=labels, y=modf, marker_color=pc,
                         text=[f"{v:.0f}" for v in modf], textposition="outside",
                         textfont=dict(color="#f0f2f8", size=9)))
    fig.update_layout(**_CL, barmode="group", height=360,
                      title=dict(text=f"<b>Feature Changes — {result['policy_label']}</b>",
                                 font=dict(color="#f0f2f8", size=14, family="Syne")))
    return fig


def chart_combined_vs_single(features, all_results, model, scaler, road_name):
    labels, reds = [], []
    for k, v in all_results.items():
        labels.append(v["policy_icon"] + " " + v["policy_label"])
        reds.append(v["reduction_pct"])
    # Combined
    from simulator import simulate_combined
    comb = simulate_combined(features, list(POLICIES.keys()), model, scaler, road_name)
    labels.append("🔗 All Combined")
    reds.append(comb["reduction_pct"])
    colors = [v["policy_color"] for v in all_results.values()] + ["#f0c040"]
    fig = go.Figure(go.Bar(x=labels, y=reds, marker_color=colors,
                           text=[f"{v:.1f}%" for v in reds],
                           textposition="outside",
                           textfont=dict(color="#f0f2f8", size=11)))
    fig.update_layout(**_CL, height=360,
                      title=dict(text="<b>Individual vs Combined Policy — Reduction %</b>",
                                 font=dict(color="#f0f2f8", size=14, family="Syne")))
    fig.update_yaxes(title_text="Congestion Reduction (%)")
    return fig, comb


# ═════════════════════════════════════════════════════════
#  LOAD MODEL & DATA
# ═════════════════════════════════════════════════════════
with st.spinner("Loading Bangalore traffic data & model …"):
    model, scaler, df = _load_model_and_data()
    st.session_state.model  = model
    st.session_state.scaler = scaler
    st.session_state.df     = df

# ═════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.9rem 0 0.6rem; border-bottom:1px solid #252a3a; margin-bottom:0.6rem;">
        <div style="font-size:1.25rem;font-weight:800;color:#f0c040;">🚦 TrafficIQ</div>
        <div style="font-size:0.62rem;font-family:'JetBrains Mono';color:#454d62;
                    letter-spacing:2px;text-transform:uppercase;">Bangalore Policy Simulator</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Location ─────────────────────────────────────────
    st.markdown('<div class="sb-section">📍 Location</div>', unsafe_allow_html=True)
    area = st.selectbox("Area", AREA_LIST,
                        index=AREA_LIST.index("Koramangala"))
    road_options = AREA_ROADS.get(area, [])
    road = st.selectbox("Road / Intersection", road_options)

    road_type      = ROAD_TYPE_MAP.get(road, "arterial_main")
    road_type_label= ROAD_TYPE_LABELS.get(road_type, "Main Road")

    # Load location stats to pre-fill sliders
    loc_stats = get_location_stats(df, area, road)

    # ── Conditions ────────────────────────────────────────
    st.markdown('<div class="sb-section">🌤 Conditions</div>', unsafe_allow_html=True)
    weather = st.selectbox("Weather", ["Clear", "Overcast", "Windy", "Fog", "Rain"])
    roadwork = st.checkbox("Active Roadwork / Construction",
                           value=bool(loc_stats.get("RoadworkActive", 0) > 0.3))
    is_weekend = st.checkbox("Weekend",
                             value=bool(loc_stats.get("IsWeekend", 0) > 0.3))

    # ── Traffic Inputs ────────────────────────────────────
    st.markdown('<div class="sb-section">🚗 Traffic Inputs</div>', unsafe_allow_html=True)
    traffic_vol = st.slider("Traffic Volume (vehicles/day)", 4000, 72000,
                            int(loc_stats.get("TrafficVolume", 28000)), 500)
    avg_speed   = st.slider("Average Speed (km/h)", 5, 90,
                            int(loc_stats.get("AverageSpeed", 38)), 1)
    capacity    = st.slider("Road Capacity Utilization %", 10, 100,
                            int(loc_stats.get("RoadCapacityUtil", 85)), 1)
    parking     = st.slider("Parking Usage %", 0, 100,
                            int(loc_stats.get("ParkingUsage", 65)), 1)
    signal_comp = st.slider("Signal Compliance %", 10, 100,
                            int(loc_stats.get("SignalCompliance", 75)), 1)
    pt_usage    = st.slider("Public Transport Usage %", 5, 100,
                            int(loc_stats.get("PublicTransportUsage", 50)), 1)
    incidents   = st.slider("Incident Reports", 0, 10,
                            int(loc_stats.get("IncidentReports", 0)), 1)
    pedestrians = st.slider("Pedestrian & Cyclist Count", 50, 250,
                            int(loc_stats.get("PedestrianCount", 110)), 5)

    st.markdown('<div class="sb-section">📋 Policy</div>', unsafe_allow_html=True)
    selected_policy = st.selectbox(
        "Select Policy to Simulate",
        list(POLICIES.keys()),
        format_func=lambda k: f"{POLICIES[k]['icon']}  {POLICIES[k]['label']}",
        key="selected_policy"
    )
    st.markdown(
        f"<div style='font-size:0.7rem;font-family:JetBrains Mono;color:#8b92a8;"
        f"background:#0f1117;border:1px solid #252a3a;border-radius:5px;"
        f"padding:0.5rem 0.7rem;margin-top:0.3rem;'>"
        f"{POLICIES[selected_policy]['description']}</div>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-section">▶ Simulate</div>', unsafe_allow_html=True)
    simulate_btn = st.button("▶  Run Simulation", type="primary",
                             use_container_width=True)

    # History
    if st.session_state.history:
        st.markdown('<div class="sb-section">📂 History</div>', unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.history)} run(s)")
        for i, h in enumerate(reversed(st.session_state.history[-4:])):
            r = h["result"]
            st.markdown(f"""<div style="background:#161921;border:1px solid #252a3a;
                border-radius:5px;padding:0.45rem 0.7rem;font-size:0.72rem;
                font-family:'JetBrains Mono';color:#8b92a8;margin-bottom:0.3rem;">
                {r['policy_icon']} {h['road']} · {h['timestamp']}<br>
                <span style="color:#34d399;">▼{r['reduction_pct']:.1f}% · {r['policy_label']}</span>
            </div>""", unsafe_allow_html=True)
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.conclusions = {}
            st.rerun()

# ═════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════
st.markdown("""
<div style="border-bottom:1px solid #252a3a; padding-bottom:0.75rem; margin-bottom:1.2rem;">
    <h1 style="font-family:'Syne',sans-serif; font-size:1.85rem; font-weight:800;
               color:#f0f2f8; margin:0; letter-spacing:-1px;">
        🚦 TrafficIQ Bangalore
    </h1>
    <p style="color:#8b92a8; font-family:'JetBrains Mono';font-size:0.72rem; margin:0.2rem 0 0;">
        Real Bangalore Traffic Data (2022–2024) · ML-Powered Policy Simulation · 8 Areas · 16 Roads
    </p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
#  LOCATION BANNER
# ═════════════════════════════════════════════════════════
st.markdown(f"""
<div class="loc-badge">
    <span style="font-size:1.2rem;">📍</span>
    <div>
        <div>
            <span class="loc-road">{road}</span>
            <span style="color:#454d62;"> · </span>
            <span class="loc-area">{area}</span>
            <span style="color:#454d62;"> · Bangalore</span>
        </div>
        <div class="loc-type">{road_type_label} &nbsp;·&nbsp;
            {loc_stats.get('RecordCount', 0)} historical records &nbsp;·&nbsp;
            Avg congestion: {loc_stats.get('AvgCongestion', 0):.1f}%
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
#  SECTION 01 — LOCATION KPIs
# ═════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">01 — LOCATION SNAPSHOT</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
def _kpi(col, val, label, sub, cls="c-white"):
    col.markdown(f"""<div class="kpi-card">
        <div class="kpi-val {cls}">{val}</div>
        <div class="kpi-lbl">{label}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

avg_c  = loc_stats.get("AvgCongestion",  0)
max_c  = loc_stats.get("MaxCongestion",  0)
wkd_c  = loc_stats.get("WeekendCongestion", 0)
pk_c   = loc_stats.get("PeakCongestion", 0)
spd_v  = loc_stats.get("AverageSpeed",   38)

congestion_cls = "c-red" if avg_c > 75 else "c-yellow" if avg_c > 50 else "c-green"
_kpi(c1, f"{avg_c:.1f}%",        "Avg Congestion",    "historical mean",     congestion_cls)
_kpi(c2, f"{max_c:.1f}%",        "Peak Congestion",   "historical max",      "c-red")
_kpi(c3, f"{pk_c:.1f}%",         "Weekday Peak",      "Mon–Fri avg",         "c-yellow")
_kpi(c4, f"{wkd_c:.1f}%",        "Weekend",           "Sat–Sun avg",         "c-blue")
_kpi(c5, f"{spd_v:.1f} km/h",    "Avg Speed",         "observed mean",       "c-green")

# ═════════════════════════════════════════════════════════
#  SECTION 02 — HISTORICAL & WEATHER CHARTS
# ═════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">02 — HISTORICAL ANALYSIS</div>', unsafe_allow_html=True)

col_h, col_w = st.columns([3, 2])

with col_h:
    trend_df = get_historical_trend(df, area, road)
    if not trend_df.empty:
        st.plotly_chart(chart_historical(trend_df),
                        use_container_width=True, key="chart_hist")
    else:
        st.info("No historical data for this selection.")

with col_w:
    weather_data = get_weather_impact(df, area, road)
    if weather_data:
        st.plotly_chart(chart_weather(weather_data),
                        use_container_width=True, key="chart_weather")

# ═════════════════════════════════════════════════════════
#  SECTION 03 — CITY-WIDE HEATMAP
# ═════════════════════════════════════════════════════════
with st.expander("🗺️  Bangalore City-Wide Congestion Heatmap"):
    st.plotly_chart(chart_area_heatmap(df),
                    use_container_width=True, key="chart_heatmap")

# ═════════════════════════════════════════════════════════
#  BUILD FEATURE DICT FROM SIDEBAR INPUTS
# ═════════════════════════════════════════════════════════
# cache_key defined here so it's available to all sections below
cache_key = f"{area}_{road}_{traffic_vol}_{avg_speed}"

features = {
    "TrafficVolume"       : traffic_vol,
    "AverageSpeed"        : avg_speed,
    "TravelTimeIndex"     : round(1.0 + (100 - avg_speed) / 100, 3),
    "RoadCapacityUtil"    : capacity,
    "IncidentReports"     : incidents,
    "EnvironmentalImpact" : round(traffic_vol * 0.003, 2),
    "PublicTransportUsage": pt_usage,
    "SignalCompliance"    : signal_comp,
    "ParkingUsage"        : parking,
    "PedestrianCount"     : pedestrians,
    "WeatherCode"         : WEATHER_CODES.get(weather, 0),
    "RoadworkActive"      : int(roadwork),
    "RoadTypeCode"        : {"arterial_main": 0, "signal_junction": 1,
                              "crossroad": 2, "expressway_ramp": 3,
                              "residential": 4}.get(road_type, 0),
    "AreaCode"            : {"Electronic City":0,"Hebbal":1,"Indiranagar":2,
                              "Jayanagar":3,"Koramangala":4,"M.G. Road":5,
                              "Whitefield":6,"Yeshwanthpur":7}.get(area, 0),
    "DayOfWeek"           : 5 if is_weekend else 2,
    "IsWeekend"           : int(is_weekend),
}

# Live prediction
live_congestion = predict_congestion(features, model, scaler)

# ── Live congestion meter ─────────────────────────────────
st.markdown('<div class="sec-header">03 — CURRENT CONDITIONS</div>', unsafe_allow_html=True)
cong_color = "#f87171" if live_congestion > 75 else "#f0c040" if live_congestion > 50 else "#34d399"
cond_label = "Severe" if live_congestion > 80 else "High" if live_congestion > 60 else "Moderate" if live_congestion > 40 else "Low"

ci1, ci2, ci3, ci4 = st.columns(4)
ci1.markdown(f"""<div class="kpi-card">
    <div class="kpi-val" style="color:{cong_color}; font-size:2rem;">{live_congestion:.1f}%</div>
    <div class="kpi-lbl">Predicted Congestion</div>
    <div class="kpi-sub">{cond_label} · {road}, {area}</div>
</div>""", unsafe_allow_html=True)

weather_impact_val = WEATHER_IMPACT.get(weather, 0)
ci2.markdown(f"""<div class="kpi-card">
    <div class="kpi-val c-blue">{weather} {'⚠' if weather_impact_val > 0.3 else '✓'}</div>
    <div class="kpi-lbl">Weather</div>
    <div class="kpi-sub">Impact: +{weather_impact_val*100:.0f}% congestion</div>
</div>""", unsafe_allow_html=True)

ci3.markdown(f"""<div class="kpi-card">
    <div class="kpi-val {'c-red' if roadwork else 'c-green'}">{'Active ⚠' if roadwork else 'None ✓'}</div>
    <div class="kpi-lbl">Roadwork</div>
    <div class="kpi-sub">{'+5% congestion penalty' if roadwork else 'No disruption'}</div>
</div>""", unsafe_allow_html=True)

ci4.markdown(f"""<div class="kpi-card">
    <div class="kpi-val c-yellow">{road_type_label}</div>
    <div class="kpi-lbl">Road Type</div>
    <div class="kpi-sub">{area} · Bangalore</div>
</div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
#  SECTION 04 — POLICY SIMULATION
# ═════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">04 — POLICY SIMULATION</div>', unsafe_allow_html=True)

if not simulate_btn and not st.session_state.last_results:
    st.info("👈 Select a policy in the sidebar and click **▶ Run Simulation**.")
else:
    if simulate_btn:
        with st.spinner(f"Simulating {POLICIES[selected_policy]['label']} …"):
            result = simulate_policy(features, selected_policy, model, scaler, road)
            st.session_state.last_results = {"result": result, "policy": selected_policy}
            st.session_state.history.append({
                "area": area, "road": road,
                "timestamp": time.strftime("%H:%M"),
                "result": result,
            })
            st.session_state.conclusions = {}

    saved       = st.session_state.last_results
    result      = saved["result"]
    policy_key  = saved["policy"]
    pc          = result["policy_color"]
    reduced     = result["reduction_pct"] > 0
    arrow       = "▼" if reduced else "▲"
    banner_cls  = "result-banner" if reduced else "result-banner warn"

    # ── Result banner ─────────────────────────────────────
    st.markdown(f"""
    <div class="{banner_cls}">
        <div class="result-title">
            {result['policy_icon']} {result['policy_label']}
            &nbsp;·&nbsp; {road}, {area}
        </div>
        <div class="result-meta">
            Congestion: <b>{result['baseline_score']:.1f}%</b> → <b>{result['modified_score']:.1f}%</b>
            &nbsp;|&nbsp; {arrow} <b>{abs(result['reduction_pct']):.1f}%</b> change
            &nbsp;|&nbsp; Speed ↑ <b>{result['speed_gain_pct']:.1f}%</b>
            &nbsp;|&nbsp; Effectiveness on {road_type_label}: <b>{result['effectiveness']:.0f}%</b>
        </div>
        <div style="font-size:0.7rem;font-family:'JetBrains Mono';color:#475569;margin-top:0.4rem;">
            {result['policy_description']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Before vs After KPI row ───────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Congestion Before", f"{result['baseline_score']:.1f}%")
    k2.metric("Congestion After",  f"{result['modified_score']:.1f}%",
              delta=f"{result['reduction_pct']:.1f}% reduction",
              delta_color="inverse")
    k3.metric("Speed Before",  f"{result['baseline_speed']:.1f} km/h")
    k4.metric("Speed After",   f"{result['modified_speed']:.1f} km/h",
              delta=f"+{result['speed_gain_pct']:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visualization tabs ────────────────────────────────
    st.markdown('<div class="sec-header">05 — VISUALISATIONS & AI ANALYSIS</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Before vs After",
        "📈 Volume Trend",
        "🔧 Feature Changes",
        "📋 Detailed Results",
    ])

    # Tab 1 — Before vs After
    with tab1:
        fig_ba = go.Figure()
        fig_ba.add_trace(go.Bar(
            name="Before (Current)", x=["Congestion Level %"],
            y=[result["baseline_score"]],
            marker_color="#f87171",
            text=[f"{result['baseline_score']:.1f}%"],
            textposition="outside", textfont=dict(color="#f0f2f8", size=13),
        ))
        fig_ba.add_trace(go.Bar(
            name=f"After — {result['policy_label']}", x=["Congestion Level %"],
            y=[result["modified_score"]],
            marker_color=pc,
            text=[f"{result['modified_score']:.1f}%"],
            textposition="outside", textfont=dict(color="#f0f2f8", size=13),
        ))
        fig_ba.update_layout(
            **_CL, barmode="group", height=380, bargap=0.35,
            title=dict(
                text=f"<b>Congestion — Current vs After {result['policy_label']}</b><br>"
                     f"<sup>{road}, {area}</sup>",
                font=dict(color="#f0f2f8", size=14, family="Syne")),
        )
        fig_ba.update_yaxes(title_text="Congestion Level (%)", range=[0, 115])
        st.plotly_chart(fig_ba, use_container_width=True, key="chart_ba")
        render_conclusion("before_after", result, area, road, loc_stats,
                          f"{cache_key}_{policy_key}_ba")

    # Tab 2 — Volume trend
    with tab2:
        st.plotly_chart(
            chart_trend(result, model, scaler, road),
            use_container_width=True, key=f"chart_trend_{policy_key}"
        )
        render_conclusion("trend", result, area, road, loc_stats,
                          f"{cache_key}_{policy_key}_trend")

    # Tab 3 — Feature changes
    with tab3:
        st.plotly_chart(
            chart_feature_impact(result),
            use_container_width=True, key=f"chart_feat_{policy_key}"
        )
        render_conclusion("before_after", result, area, road, loc_stats,
                          f"{cache_key}_{policy_key}_feat")

    # Tab 4 — Detailed results
    with tab4:
        st.dataframe(pd.DataFrame([{
            "Policy"          : f"{result['policy_icon']} {result['policy_label']}",
            "Location"        : f"{road}, {area}",
            "Road Type"       : road_type_label,
            "Baseline %"      : result["baseline_score"],
            "After Policy %"  : result["modified_score"],
            "Reduction %"     : result["reduction_pct"],
            "Speed Before"    : result["baseline_speed"],
            "Speed After"     : result["modified_speed"],
            "Speed Gain %"    : result["speed_gain_pct"],
            "TTI Improvement" : result["tti_improvement"],
            "Cap. Freed %"    : result["cap_freed_pct"],
            "Effectiveness"   : f"{result['effectiveness']:.0f}%",
        }]), use_container_width=True, hide_index=True)

        with st.expander("🔍 Feature Values Before / After"):
            feat_rows = []
            for feat in FEATURE_COLS:
                before  = result["inputs"].get(feat, FEATURE_DEFAULTS.get(feat, 0))
                after_v = result["modified_features"].get(feat, before)
                delta   = after_v - before
                feat_rows.append({
                    "Feature" : feat,
                    "Before"  : round(before, 3),
                    "After"   : round(after_v, 3),
                    "Δ Change": round(delta, 3),
                })
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════
#  SECTION 05 — WEATHER & HISTORICAL AI ANALYSIS
# ═════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">06 — WEATHER & HISTORICAL INSIGHTS</div>',
            unsafe_allow_html=True)

col_wa, col_ha = st.columns(2)
with col_wa:
    if weather_data:
        st.markdown("**Weather Impact Analysis**")
        render_conclusion("weather",
                          {"weather_data": weather_data},
                          area, road, loc_stats, f"{cache_key}_weather")
with col_ha:
    if not trend_df.empty:
        st.markdown("**Historical Trend Analysis**")
        render_conclusion("historical", {}, area, road, loc_stats,
                          f"{cache_key}_historical")

# ═════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════
st.markdown("""
<div style="border-top:1px solid #252a3a; margin-top:2rem; padding-top:0.9rem;
     text-align:center; font-family:'JetBrains Mono'; font-size:0.67rem; color:#454d62;">
    TrafficIQ Bangalore &nbsp;·&nbsp; Real Traffic Data 2022–2024 &nbsp;·&nbsp;
    Stacked Ensemble ML &nbsp;·&nbsp; Groq AI Analysis &nbsp;·&nbsp; 8 Areas · 16 Roads
</div>
""", unsafe_allow_html=True)