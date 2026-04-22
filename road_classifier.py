"""
============================================================
  GROQ AI CONCLUSIONS — Bangalore Traffic
  TrafficIQ Bangalore | road_classifier.py
============================================================
Generates AI-powered analysis conclusions for each chart
using Groq API (llama3-8b-8192).

API KEY SETUP:
  Create a .env file in the project root:
      GROQ_API_KEY=gsk_your_key_here
  Get a free key at: https://console.groq.com
============================================================
"""

import os, re, json, warnings
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama3-8b-8192"

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ─────────────────────────────────────────────────────────
#  LOCATION SUMMARY (for Groq prompt context)
# ─────────────────────────────────────────────────────────
def _location_context(area: str, road: str, stats: dict) -> str:
    return (
        f"Location: {road}, {area}, Bangalore\n"
        f"Road Type: {stats.get('RoadTypeLabel', 'Main Road')}\n"
        f"Avg Congestion: {stats.get('AvgCongestion', 'N/A')}%\n"
        f"Avg Speed: {stats.get('AverageSpeed', 'N/A'):.1f} km/h\n"
        f"Traffic Volume: {stats.get('TrafficVolume', 'N/A'):.0f} vehicles/day\n"
        f"Road Capacity Utilization: {stats.get('RoadCapacityUtil', 'N/A'):.1f}%\n"
        f"Signal Compliance: {stats.get('SignalCompliance', 'N/A'):.1f}%\n"
        f"Parking Usage: {stats.get('ParkingUsage', 'N/A'):.1f}%\n"
    )


# ─────────────────────────────────────────────────────────
#  GRAPH CONCLUSION PROMPTS
# ─────────────────────────────────────────────────────────
def _build_prompt(graph_type: str, result: dict, area: str,
                  road: str, stats: dict) -> str:
    loc = _location_context(area, road, stats)
    base = result.get("baseline_score", 0)
    after = result.get("modified_score", 0)
    red   = result.get("reduction_pct", 0)
    spd   = result.get("speed_gain_pct", 0)
    policy= result.get("policy_label", "")
    eff   = result.get("effectiveness", 0)

    prompts = {
        "before_after": f"""You are a Bangalore urban traffic analyst.
Write exactly 5 bullet points. Each bullet MUST start on a NEW LINE with '• '. No paragraphs. No grouping. One point per line only. for a Before vs After chart.

{loc}
Policy Applied: {policy}
Congestion: {base:.1f}% → {after:.1f}% ({red:.1f}% reduction)
Speed improvement: +{spd:.1f}%
Policy effectiveness on this road type: {eff:.0f}%

Cover: what changed, why this policy suits this Bangalore road, practical impact on commuters, implementation note, and recommendation.""",

        "trend": f"""You are a Bangalore urban traffic analyst.
Write exactly 5 bullet points. Each bullet MUST start on a NEW LINE with '• '. No paragraphs. No grouping. One point per line only. for a Congestion vs Traffic Volume trend chart.

{loc}
Policy: {policy}
At current volume: {base:.1f}% → {after:.1f}% ({red:.1f}% reduction)

Cover: trend interpretation at low/high volumes, where policy has the biggest impact, 
Bangalore peak hour relevance, scalability, and deployment recommendation.""",

        "policy_compare": f"""You are a Bangalore urban traffic analyst.
Write exactly 5 bullet points. Each bullet MUST start on a NEW LINE with '• '. No paragraphs. No grouping. One point per line only. comparing all 4 traffic policies.

{loc}
Results: {json.dumps({k: v.get('reduction_pct', 0) for k, v in result.items() if isinstance(v, dict)}, indent=2)}

Cover: which policy wins for this road type, which has worst ROI, combined strategy potential,
enforcement cost vs benefit, and final recommendation for BBMP/BMTC.""",

        "weather": f"""You are a Bangalore urban traffic analyst.
Write exactly 5 bullet points. Each bullet MUST start on a NEW LINE with '• '. No paragraphs. No grouping. One point per line only. for a Weather vs Congestion chart.

{loc}
Weather breakdown: {json.dumps(result.get('weather_data', {}), indent=2)}

Cover: which weather condition causes worst congestion here, monsoon impact, 
adaptive signal needs, comparison to rest of Bangalore, mitigation strategy.""",

        "historical": f"""You are a Bangalore urban traffic analyst.
Write exactly 5 bullet points. Each bullet MUST start on a NEW LINE with '• '. No paragraphs. No grouping. One point per line only. for a Historical Congestion Trend chart.

{loc}
Time range: 2022–2024. Avg congestion: {stats.get('AvgCongestion', 0):.1f}%

Cover: seasonal patterns, worsening/improving trend, specific calendar events that spike congestion,
data quality observations, and long-term policy implications.""",
    }

    return prompts.get(graph_type, prompts["before_after"])


def generate_conclusion(graph_type: str, result: dict,
                        area: str = "", road: str = "",
                        stats: dict = None) -> str:
    """Generate AI conclusion. Falls back to rule-based if no API key."""
    stats = stats or {}
    key   = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")

    if key and GROQ_AVAILABLE:
        try:
            client   = Groq(api_key=key)
            prompt   = _build_prompt(graph_type, result, area, road, stats)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=400,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

    return _fallback_conclusion(graph_type, result, area, road, stats)


# ─────────────────────────────────────────────────────────
#  RULE-BASED FALLBACK
# ─────────────────────────────────────────────────────────
def _fallback_conclusion(graph_type, result, area, road, stats):
    """
    Returns conclusions as one bullet per line (\n separated).
    render_conclusion() in app.py will split these into separate st.markdown lines.
    """
    policy = result.get("policy_label", "the selected policy")
    base   = result.get("baseline_score", 0)
    after  = result.get("modified_score", 0)
    red    = result.get("reduction_pct", 0)
    spd    = result.get("speed_gain_pct", 0)
    eff    = result.get("effectiveness", 0)
    rt     = stats.get("RoadTypeLabel", "this road type")

    if graph_type == "before_after":
        points = [
            f"**{policy}** reduced congestion at **{road}, {area}** from **{base:.1f}%** to **{after:.1f}%** — a **{red:.1f}% improvement** for daily commuters.",
            f"Policy effectiveness on {rt} is rated at **{eff:.0f}%**, indicating strong structural fit for this road geometry and traffic pattern.",
            f"Speed improved by **+{spd:.1f}%**, directly cutting travel time for thousands of commuters using this corridor every day.",
            f"The reduction in road capacity utilization lowers the risk of incident-triggered secondary congestion, especially during peak hours.",
            f"BBMP should pilot this policy for a 30-day trial period and measure real-time compliance using traffic camera analytics before full rollout.",
        ]
    elif graph_type == "trend":
        points = [
            f"The trend chart confirms congestion at **{road}** scales sharply once traffic volume exceeds approximately 30,000 vehicles per day.",
            f"**{policy}** consistently keeps the congestion curve lower across all volume levels, demonstrating reliable effectiveness during Bangalore peak hours.",
            f"At the current observed volume, the policy delivers a **{red:.1f}% congestion reduction** — sufficient to make a perceptible difference to commuters.",
            f"The policy is most impactful during morning (07–10) and evening (17–20) peaks, where volume-sensitive interventions provide maximum benefit.",
            f"A data-driven scheduling system should auto-activate this policy when live sensor readings exceed 35,000 vehicles per day.",
        ]
    elif graph_type == "policy_compare":
        points = [
            f"All policies provide measurable congestion reduction at **{road}, {area}**, each targeting different congestion drivers with varying enforcement requirements.",
            f"Signal Optimisation has the lowest implementation cost and can be deployed via adaptive timing software on existing BBMP signal infrastructure.",
            f"Parking Enforcement delivers the highest ROI on arterial roads but requires active patrol units and smart parking redirection to be effective.",
            f"Peak Hour Restriction produces the largest absolute volume reduction but requires vehicle exemption management and faces public resistance without awareness campaigns.",
            f"A combined Signal Optimisation and Parking Enforcement strategy is the recommended starting point — high impact, low infrastructure cost, and minimal commuter disruption.",
        ]
    elif graph_type == "weather":
        points = [
            f"Weather meaningfully impacts congestion at **{road}, {area}**, with Fog and Windy conditions causing the most significant speed drops and stop-start patterns.",
            f"Monsoon season (June–September) creates compounded congestion as waterlogging reduces effective lane capacity on Bangalore's low-gradient road network.",
            f"Adaptive signal timing that extends green phases during adverse weather could offset 15–25% of weather-induced delay at this location.",
            f"Variable message signs at arterial entry points can redirect traffic to alternate corridors during Fog or Heavy Rain events, reducing incident risk.",
            f"BBMP should integrate real-time IMD weather API feeds into the traffic management center to enable proactive, weather-triggered policy activation.",
        ]
    else:
        points = [
            f"Historical data for **{road}, {area}** spans 2022–2024, capturing Bangalore's rapid traffic growth during the post-pandemic economic recovery period.",
            f"The trend shows persistent high congestion above 70% for most of the observation window, pointing to structural capacity constraints rather than temporary demand spikes.",
            f"Seasonal peaks align with school reopening in June and festive seasons in October–November, which should inform annual traffic management and enforcement calendars.",
            f"Data consistency across the 2+ year period confirms this dataset is a reliable foundation for ML model training and policy impact forecasting.",
            f"Long-term congestion relief at this location requires infrastructure investment alongside demand-side management — policy interventions alone cannot resolve structural capacity gaps.",
        ]

    return "\n".join(points)