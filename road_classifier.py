"""
============================================================
  PART 2 — LLM ROAD CLASSIFIER (Groq API)
  Traffic Policy Simulator | road_classifier.py
============================================================
Sends extracted traffic features to Groq (groq.com) and
receives road classification + graph conclusions.

API KEY SETUP:
  Create a .env file in the project root with:
      GROQ_API_KEY=gsk_your_key_here
  Get a free key at: https://console.groq.com
============================================================
"""

import os, json, re, warnings
warnings.filterwarnings("ignore")

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # pip install python-dotenv

# API key is loaded from .env file or environment variable ONLY
# Never hardcode your key here — use the .env file instead
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GROQ_MODEL = "llama3-8b-8192"  # fast + free tier model

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False


# ─────────────────────────────────────────────────────────
#  ROAD PROFILE
# ─────────────────────────────────────────────────────────
class RoadProfile:
    def __init__(self, road_type, lanes, density_label, stopped_label,
                 speed_label, vehicle_mix, lane_discipline,
                 behavior_summary, congestion_level, source="LLM"):
        self.road_type        = road_type
        self.lanes            = lanes
        self.density_label    = density_label
        self.stopped_label    = stopped_label
        self.speed_label      = speed_label
        self.vehicle_mix      = vehicle_mix
        self.lane_discipline  = lane_discipline
        self.behavior_summary = behavior_summary
        self.congestion_level = congestion_level
        self.source           = source

    def to_dict(self):
        return {
            "Road Type"        : self.road_type,
            "Lanes"            : self.lanes,
            "Vehicle Density"  : self.density_label,
            "Stopped Vehicles" : self.stopped_label,
            "Speed"            : self.speed_label,
            "Traffic Mix"      : self.vehicle_mix,
            "Lane Discipline"  : self.lane_discipline,
            "Congestion Level" : self.congestion_level,
            "Behavior Summary" : self.behavior_summary,
            "Classified By"    : self.source,
        }

    def __repr__(self):
        return "\n".join([
            f"  Road Type       : {self.road_type}",
            f"  Lanes           : {self.lanes}",
            f"  Vehicle Density : {self.density_label}",
            f"  Stopped Vehicles: {self.stopped_label}",
            f"  Speed           : {self.speed_label}",
            f"  Traffic Mix     : {self.vehicle_mix}",
            f"  Lane Discipline : {self.lane_discipline}",
            f"  Congestion Level: {self.congestion_level}",
            f"  Summary         : {self.behavior_summary}",
        ])


# ─────────────────────────────────────────────────────────
#  LABEL HELPERS
# ─────────────────────────────────────────────────────────
def _density_label(d): return "Low" if d < 5 else "Moderate" if d < 12 else "High"
def _stopped_label(s): return "Low" if s < 2 else "Moderate" if s < 6 else "High"
def _speed_label(v):   return "Slow" if v < 15 else "Moderate" if v < 40 else "Fast"
def _mix_label(car_r, bike_r, bus_r):
    if bike_r > 0.35 or bus_r > 0.25: return "Heavy/Mixed"
    if car_r > 0.70:                   return "Light (Mostly Cars)"
    return "Mixed"
def _congestion_label(s): return "Low" if s<15 else "Moderate" if s<30 else "High" if s<50 else "Severe"


# ─────────────────────────────────────────────────────────
#  RULE-BASED FALLBACK
# ─────────────────────────────────────────────────────────
def _rule_based_classify(features: dict) -> RoadProfile:
    vc    = features.get("VehicleCount", 0)
    st    = features.get("Stopped",      0)
    spd   = features.get("EstSpeed",     30)
    lns   = int(features.get("Lanes",    2))
    sigs  = features.get("Signals",      0)
    peds  = features.get("Pedestrians",  0)
    bike_r = features.get("BikeRatio",   0.2)
    bus_r  = features.get("BusRatio",    0.1)
    car_r  = features.get("CarRatio",    0.5)
    den   = features.get("Density",      vc / max(lns, 1))
    score = features.get("CongestionScore", (0.5 * vc) + (2 * st))

    if sigs >= 1 and peds > 2:         road_type = "Signal Junction"
    elif lns >= 4 and spd > 40:        road_type = "Main Road / Highway"
    elif vc < 15 and lns <= 2:         road_type = "Residential Street"
    elif peds > 5 and bike_r > 0.3:    road_type = "Cross Road / Mixed Zone"
    else:                              road_type = "Main Road"

    sr = st / max(vc, 1)
    discipline = "Good" if sr < 0.05 else "Moderate" if sr < 0.15 else "Poor"
    summary = (
        f"Traffic is {'heavy' if vc > 30 else 'light'} with "
        f"{'frequent stops' if st > 5 else 'smooth flow'}. "
        f"Vehicle mix is {_mix_label(car_r, bike_r, bus_r).lower()}. "
        f"Speed {spd} km/h indicates {'congested' if spd < 20 else 'moderate'} conditions."
    )
    return RoadProfile(road_type, lns, _density_label(den), _stopped_label(st),
                       _speed_label(spd), _mix_label(car_r, bike_r, bus_r),
                       discipline, summary, _congestion_label(score),
                       source="Rule-Based (No API Key)")


# ─────────────────────────────────────────────────────────
#  GROQ ROAD CLASSIFIER
# ─────────────────────────────────────────────────────────
def _build_classify_prompt(features: dict) -> str:
    return f"""You are an expert traffic analyst AI for a city traffic management system.
Analyze these traffic features extracted from a road video and classify the road.

--- FEATURES ---
VehicleCount   : {features.get('VehicleCount','N/A')}
Stopped        : {features.get('Stopped','N/A')}
WrongParked    : {features.get('WrongParked','N/A')}
EstSpeed       : {features.get('EstSpeed','N/A')} km/h
Lanes          : {features.get('Lanes','N/A')}
Density        : {features.get('Density','N/A')} vehicles/lane
CarRatio       : {features.get('CarRatio','N/A')}
BikeRatio      : {features.get('BikeRatio','N/A')}
BusRatio       : {features.get('BusRatio','N/A')}
Pedestrians    : {features.get('Pedestrians','N/A')}
Signals        : {features.get('Signals','N/A')}
CongestionScore: {features.get('CongestionScore','N/A')}
----------------

Respond ONLY with valid JSON, no markdown:
{{
  "road_type": "Main Road | Signal Junction | Cross Road | Residential Street | Highway",
  "lane_discipline": "Good | Moderate | Poor",
  "congestion_level": "Low | Moderate | High | Severe",
  "density_label": "Low | Moderate | High",
  "stopped_label": "Low | Moderate | High",
  "speed_label": "Slow | Moderate | Fast",
  "vehicle_mix": "e.g. Mixed, Light (Mostly Cars), Heavy/Mixed",
  "behavior_summary": "2-3 sentence traffic summary with recommendations"
}}"""


def classify_with_groq(features: dict, api_key: str) -> RoadProfile:
    if not GROQ_SDK_AVAILABLE:
        raise ImportError("pip install groq")
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": _build_classify_prompt(features)}],
        temperature=0.2, max_tokens=500,
    )
    raw = re.sub(r"```(?:json)?", "", response.choices[0].message.content.strip()).strip("`").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}

    return RoadProfile(
        road_type        = data.get("road_type",        "Unknown"),
        lanes            = int(features.get("Lanes", 2)),
        density_label    = data.get("density_label",    _density_label(features.get("Density", 0))),
        stopped_label    = data.get("stopped_label",    _stopped_label(features.get("Stopped", 0))),
        speed_label      = data.get("speed_label",      _speed_label(features.get("EstSpeed", 30))),
        vehicle_mix      = data.get("vehicle_mix",      "Mixed"),
        lane_discipline  = data.get("lane_discipline",  "Moderate"),
        behavior_summary = data.get("behavior_summary", ""),
        congestion_level = data.get("congestion_level", "Moderate"),
        source           = f"Groq API ({GROQ_MODEL})",
    )


# ─────────────────────────────────────────────────────────
#  GRAPH CONCLUSION GENERATOR (Groq or fallback)
# ─────────────────────────────────────────────────────────
def generate_graph_conclusion(graph_type: str, result: dict, api_key: str = "") -> str:
    """
    Generate a 100-150 word pointwise conclusion for a graph.
    graph_type: "before_after" | "trend" | "feature_changes" | "scenario_compare"
    """
    prompts = {
        "before_after": f"""You are a traffic policy analyst. Write a 100-150 word pointwise conclusion 
(use bullet points starting with •) for a Before vs After bar chart.
Policy: {result['policy_label']}
Baseline Congestion: {result['baseline_score']} → After Policy: {result['modified_score']} ({result['reduction_pct']:.1f}% reduction)
Vehicle Count: {result['inputs']['VehicleCount']}, Stopped reduced to {result['modified_features'].get('Stopped','N/A')}
Write exactly 5 bullet points: what changed, mechanism, traffic impact, safety benefit, recommendation.""",

        "trend": f"""You are a traffic policy analyst. Write a 100-150 word pointwise conclusion
(use bullet points starting with •) for a Congestion vs Vehicle Count trend graph.
Policy: {result['policy_label']}
Current VC: {result['inputs']['VehicleCount']}, Congestion reduction: {result['reduction_pct']:.1f}%, Speed gain: {result['speed_improvement_pct']:.1f}%
Write exactly 5 bullet points: trend interpretation, where gap is largest, scalability, peak-hour impact, deployment recommendation.""",

        "feature_changes": f"""You are a traffic policy analyst. Write a 100-150 word pointwise conclusion
(use bullet points starting with •) for a Feature Changes grouped bar chart.
Policy: {result['policy_label']}
Original: VC={result['inputs']['VehicleCount']}, Stopped={result['inputs']['Stopped']}, WP={result['inputs'].get('WrongParked',0)}, Speed={result['inputs'].get('EstSpeed',25)}
After: {result['modified_features']}
Write exactly 5 bullet points: biggest feature change, mechanism, road safety effect, compliance requirement, overall feasibility.""",

        "scenario_compare": f"""You are a traffic policy analyst. Write a 100-150 word pointwise conclusion
(use bullet points starting with •) comparing traffic policy simulation scenarios.
Current Policy: {result['policy_label']}, Reduction: {result['reduction_pct']:.1f}%, Speed gain: {result['speed_improvement_pct']:.1f}%
Write exactly 5 bullet points: performance summary, best use case for this policy, tradeoffs vs alternatives, enforcement cost, final authority recommendation.""",
    }

    key = api_key or GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    prompt = prompts.get(graph_type, prompts["before_after"])

    if key and GROQ_SDK_AVAILABLE:
        try:
            client = Groq(api_key=key)
            resp   = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            pass  # fall through to rule-based

    return _fallback_conclusion(graph_type, result)


def _fallback_conclusion(graph_type: str, result: dict) -> str:
    r = result
    if graph_type == "before_after":
        return (
            f"• The **{r['policy_label']}** reduced the congestion score from **{r['baseline_score']}** "
            f"to **{r['modified_score']}**, achieving a **{r['reduction_pct']:.1f}% improvement** in traffic flow.\n"
            f"• Stopped vehicles in active lanes were directly reduced by the policy, freeing blocked road capacity and reducing bottleneck events.\n"
            f"• The lower congestion score translates to fewer rear-end incidents and smoother acceleration/deceleration cycles for all road users.\n"
            f"• Speed improved by **{r['speed_improvement_pct']:.1f}%**, indicating that vehicles now traverse the corridor more efficiently.\n"
            f"• Authorities should pilot this policy during peak hours to validate these simulation outcomes against live sensor data."
        )
    elif graph_type == "trend":
        return (
            f"• The trend graph confirms that under baseline conditions, congestion escalates sharply as vehicle count rises beyond 20 vehicles/frame.\n"
            f"• Under the **{r['policy_label']}**, congestion remains consistently lower across all vehicle density levels, demonstrating structural improvement.\n"
            f"• The benefit gap widens at higher vehicle counts, meaning the policy is **most effective during peak traffic hours** when it matters most.\n"
            f"• At the current input of **{int(r['inputs']['VehicleCount'])} vehicles**, the policy delivers a **{r['reduction_pct']:.1f}% congestion reduction**.\n"
            f"• This policy should be deployed primarily during high-volume periods (07:00–10:00 and 17:00–20:00) for maximum urban impact."
        )
    elif graph_type == "feature_changes":
        return (
            f"• The **{r['policy_label']}** directly modified vehicle count, stopped vehicles, and wrong-parked figures — the three primary congestion drivers.\n"
            f"• Reduction in stopped in-lane vehicles eliminates bottlenecks that cause cascading slowdowns across upstream road segments.\n"
            f"• Wrong parking clearance restores full lane width, improving throughput and reducing dangerous lane-switching manoeuvres.\n"
            f"• Speed increased by **{r['speed_improvement_pct']:.1f}%**, confirming better road space utilisation and reduced idle vehicle clustering.\n"
            f"• Combined feature improvements make this a high-impact, low-infrastructure-cost intervention suitable for immediate deployment."
        )
    else:
        return (
            f"• The **{r['policy_label']}** achieved **{r['reduction_pct']:.1f}%** congestion reduction, making it a statistically significant improvement.\n"
            f"• Comparing scenarios enables authorities to identify which policy delivers better outcomes for specific road types and time windows.\n"
            f"• Speed improvement of **{r['speed_improvement_pct']:.1f}%** adds measurable quality-of-life benefit beyond just reducing congestion scores.\n"
            f"• Each policy carries different enforcement costs: No-Parking requires active patrol; One-Way requires signage; Peak Restriction needs digital gates.\n"
            f"• Data-driven simulation allows evidence-based traffic management, reducing dependency on reactive, trial-and-error physical interventions."
        )


# ─────────────────────────────────────────────────────────
#  MAIN ENTRY
# ─────────────────────────────────────────────────────────
def classify_road(features: dict, api_key: str = None) -> RoadProfile:
    key = api_key or GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if key and GROQ_SDK_AVAILABLE:
        try:
            profile = classify_with_groq(features, key)
            return profile
        except Exception as e:
            print(f"[WARNING] Groq error: {e}. Using rule-based fallback.")
    return _rule_based_classify(features)


if __name__ == "__main__":
    sample = {
        "VehicleCount": 45, "Density": 11.25, "Stopped": 8, "WrongParked": 4,
        "EstSpeed": 18, "Lanes": 4, "CarRatio": 0.55, "BikeRatio": 0.25,
        "BusRatio": 0.12, "Pedestrians": 6, "Signals": 1, "CongestionScore": 38.5,
    }
    print(classify_road(sample))
