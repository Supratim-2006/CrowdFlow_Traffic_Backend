import os
import sys
import asyncio
import random
import traceback
import tempfile
import pickle
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

# Add subdirectories to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import CrowdFlow
from crowdflow_detector2 import CrowdFlowDetector
from download_model import MODEL_PATH as YOLO_MODEL_PATH

# Import Road Closure
import road_closure_app as rc
from huggingface_hub import hf_hub_download

# Import Traffic Disruption Model
import app as disruption_app

# Set environment variables for routing
TRAFFIC_ROUTING_DIR = os.path.join(BASE_DIR, "Traffic_routing")
GRAPHML_PATH = os.path.join(TRAFFIC_ROUTING_DIR, "bengaluru_crowdflow.graphml")
EVENTS_CSV_PATH = os.path.join(TRAFFIC_ROUTING_DIR, "data.csv")

if not os.path.isdir(TRAFFIC_ROUTING_DIR):
    os.makedirs(TRAFFIC_ROUTING_DIR, exist_ok=True)

root_events_csv = os.path.join(BASE_DIR, "data.csv")
if not os.path.exists(EVENTS_CSV_PATH) and os.path.exists(root_events_csv):
    EVENTS_CSV_PATH = root_events_csv

os.environ["EVENTS_CSV"] = EVENTS_CSV_PATH
os.environ["GRAPHML_PATH"] = GRAPHML_PATH

# Import Routing
import routing
from models import MainRoadRequest, BypassRequest, EventAwareRouteRequest

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CrowdFlow AI — Unified Backend API",
    description="Unified API combining Orchestrator, Object Detection, Road Closure, Disruption, and Routing.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO Model
detector = CrowdFlowDetector(model_name=YOLO_MODEL_PATH, device="cpu")

# Load Road Closure Model
try:
    rc_model_path = hf_hub_download(repo_id="SupratimKukri/road-closure-model", filename="road_closure_model.pkl")
    with open(rc_model_path, "rb") as f:
        RC_BUNDLE = pickle.load(f)
    RC_THRESHOLD = RC_BUNDLE.get("threshold", 0.5)
    print(f"Road Closure Model loaded | threshold = {RC_THRESHOLD:.3f}")
except Exception as e:
    RC_BUNDLE = None
    RC_THRESHOLD = 0.5
    print(f"Failed to load Road Closure Model: {e}")

# Load Traffic Disruption Model
try:
    disruption_model_path = hf_hub_download(repo_id="SupratimKukri/traffic-disruption-model", filename="traffic_disruption_model.pkl")
    with open(disruption_model_path, "rb") as f:
        DISRUPTION_MODEL = pickle.load(f)
    print("Traffic Disruption Model loaded successfully")
except Exception as e:
    DISRUPTION_MODEL = None
    print(f"Failed to load Traffic Disruption Model: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR CONFIG & UTILS (From orchestrator.py)
# ─────────────────────────────────────────────────────────────────────────────
EVENT_CAUSE_PLANNED_MAP = {
    "vehicle_breakdown":   "unplanned",
    "others":              "unplanned",
    "pot_holes":           "unplanned",
    "construction":        "planned",
    "water_logging":       "unplanned",
    "accident":            "unplanned",
    "tree_fall":           "unplanned",
    "road_conditions":     "unplanned",
    "congestion":          "unplanned",
    "public_event":        "planned",
    "procession":          "planned",
    "vip_movement":        "planned",
    "protest":             "unplanned",
    "debris":              "unplanned",
    "test_demo":           "unplanned",
    "fog_low_visibility":  "unplanned",
}

def _normalize_cause(raw: str) -> str:
    return raw.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_").strip("_")

def _resolve_event_cause(raw: str) -> tuple[str, str]:
    normalized = _normalize_cause(raw)
    if normalized in EVENT_CAUSE_PLANNED_MAP:
        return normalized, EVENT_CAUSE_PLANNED_MAP[normalized]
    return "others", "unplanned"

def _build_closure_payload(analysis: dict, latitude: float, longitude: float, zone: str, corridor: str, junction: Optional[str], event_cause: str, event_type: str) -> rc.PredictionRequest:
    emergency = analysis.get("emergency", {})
    em_level  = emergency.get("level", "LOW")
    veh_types = analysis.get("vehicle_types", {})
    priority_map = {"CRITICAL": "High", "HIGH": "High", "MEDIUM": "Low", "LOW": "Low"}
    dominant_veh = ""
    if veh_types:
        raw = max(veh_types, key=veh_types.get)
        dominant_veh = {"car": "Car", "truck": "Truck", "bus": "Bus", "motorbike": "Two-Wheeler", "van": "Van"}.get(raw, raw.title())

    return rc.PredictionRequest(
        start_datetime=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+0000"),
        latitude=latitude,
        longitude=longitude,
        event_cause=event_cause,
        priority=priority_map.get(em_level, "Low"),
        zone=zone,
        corridor=corridor,
        event_type=event_type,
        veh_type=dominant_veh or None,
        junction=junction
    )

def _build_disruption_payload(analysis: dict, closure_payload: rc.PredictionRequest) -> disruption_app.PredictionRequest:
    emergency = analysis.get("emergency", {})
    reasons   = emergency.get("reasons", [])
    desc      = "; ".join(reasons[:2]) if reasons else "Traffic incident detected"
    comment   = f"{analysis.get('vehicles', 0)} vehicle(s), {analysis.get('people', 0)} person(s) at scene."
    
    return disruption_app.PredictionRequest(
        start_datetime=closure_payload.start_datetime,
        latitude=closure_payload.latitude,
        longitude=closure_payload.longitude,
        event_cause=closure_payload.event_cause,
        priority=closure_payload.priority,
        zone=closure_payload.zone,
        corridor=closure_payload.corridor,
        event_type=closure_payload.event_type,
        veh_type=closure_payload.veh_type,
        junction=closure_payload.junction,
        description=desc,
        comment=comment,
        requires_road_closure=True,
        status="Open"
    )

def _congestion_pct(score: float) -> int:
    return min(100, max(0, int(round(score * 100))))

def _generate_heatmap_points(lat: float, lon: float, congestion_score: float, n: int = 50) -> list[list[float]]:
    spread = max(0.002, congestion_score * 0.025)
    return [[lat + random.gauss(0, spread), lon + random.gauss(0, spread), round(congestion_score * random.uniform(0.6, 1.0), 3)] for _ in range(n)]

def _generate_recommendations(em_level: str, road_closure_req: bool, severity_label: str, people: int, vehicles: int, congestion_score: float) -> list[str]:
    recs = []
    if em_level in ("CRITICAL", "HIGH"):
        recs.append("Deploy traffic police at scene immediately")
        recs.append("Dispatch emergency services (ambulance / fire)")
    if road_closure_req:
        recs.append("Implement road closure — use routing API for live diversion route")
    if people > 20:
        recs.append("Crowd control required — deploy marshals at perimeter")
    if vehicles > 10 or congestion_score > 0.7:
        recs.append("Increase public transport frequency on adjacent routes")
    if severity_label in ("90–240 mins (Major)", ">240 mins (Severe)"):
        recs.append("Inform public via variable message signs and radio")
        recs.append("Alert city traffic control centre")
    if not recs:
        recs.append("Monitor situation — no immediate action required")
    return recs

_CONGESTION_BASE_BRACKETS = [
    (0.25, "Light (2-3 personnel)",        2),
    (0.50, "Moderate (3-5 personnel)",     3),
    (0.70, "Heavy (5-8 personnel)",        5),
    (0.85, "Severe (8-12 personnel)",      8),
    (1.01, "Critical (12-20 personnel)",  12),
]
_EMERGENCY_LEVEL_BONUS = {"LOW": 0, "MEDIUM": 1, "HIGH": 3, "CRITICAL": 6}
POLICE_STATION_LOOKUP_THRESHOLD = 6
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

async def _find_nearest_police_station(client: httpx.AsyncClient, lat: float, lon: float, radius_m: int = 8000) -> Optional[dict]:
    query = f"""
    [out:json][timeout:10];
    node["amenity"="police"](around:{radius_m},{lat},{lon});
    out body;
    """
    try:
        resp = await client.post(OVERPASS_URL, data={"data": query}, timeout=15.0)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        return None
    if not elements: return None
    def _dist(el: dict) -> float:
        dlat = el["lat"] - lat
        dlon = el["lon"] - lon
        return (dlat * dlat) + (dlon * dlon)
    nearest = min(elements, key=_dist)
    tags = nearest.get("tags", {})
    dlat = nearest["lat"] - lat
    dlon = nearest["lon"] - lon
    approx_km = round(((dlat * 111.0) ** 2 + (dlon * 111.0 * 0.96) ** 2) ** 0.5, 2)
    return {
        "name": tags.get("name", "Unnamed Police Station"),
        "latitude": nearest["lat"], "longitude": nearest["lon"],
        "approx_distance_km": approx_km,
        "address": ", ".join(v for v in [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:suburb"), tags.get("addr:city")] if v) or None,
        "phone": tags.get("phone") or tags.get("contact:phone"),
        "source": "OpenStreetMap (Overpass API)",
    }

def _police_personnel_required(congestion_score: float, em_level: str, event_type: str, vehicles: int, people: int) -> dict:
    congestion_score = max(0.0, min(1.0, congestion_score))
    base = _CONGESTION_BASE_BRACKETS[-1]
    for upper, label, personnel in _CONGESTION_BASE_BRACKETS:
        if congestion_score < upper:
            base = (upper, label, personnel)
            break
    bracket_label, base_personnel = base[1], base[2]
    emergency_bonus = _EMERGENCY_LEVEL_BONUS.get(em_level.upper(), 0)
    planned_bonus = 0 if event_type == "planned" else 1
    scale_bonus = 0
    if vehicles > 10: scale_bonus += 2
    elif vehicles > 5: scale_bonus += 1
    if people > 20: scale_bonus += 2
    elif people > 10: scale_bonus += 1
    total_personnel = max(1, base_personnel + emergency_bonus + planned_bonus + scale_bonus)
    
    if total_personnel <= 3: final_bracket = "Light (2-3 personnel)"
    elif total_personnel <= 5: final_bracket = "Moderate (3-5 personnel)"
    elif total_personnel <= 8: final_bracket = "Heavy (5-8 personnel)"
    elif total_personnel <= 12: final_bracket = "Severe (8-12 personnel)"
    else: final_bracket = "Critical (12-20 personnel)"
    
    return {
        "personnel_required": total_personnel,
        "bracket": final_bracket,
        "breakdown": {"congestion_base": base_personnel, "emergency_bonus": emergency_bonus, "event_type_bonus": planned_bonus, "scale_bonus": scale_bonus},
    }

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "CrowdFlow AI Unified Backend"}

@app.post("/analyze", tags=["Analysis"])
async def analyze(
    file: UploadFile = File(...),
    road_block_reason: str = Form(...),
    latitude: float = Form(12.9716),
    longitude: float = Form(77.5946),
    zone: str = Form("East Zone 1"),
    corridor: str = Form("Non-corridor"),
    junction: str = Form(None),
):
    if not road_block_reason or not road_block_reason.strip():
        raise HTTPException(status_code=422, detail="road_block_reason is mandatory")

    event_cause, event_type = _resolve_event_cause(road_block_reason)

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    try:
        # Step 1: Object Detection
        try:
            report = detector.detect(image_path=image_path, save_path=None, show=False)
            analysis = report.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Object detection failed: {exc}")

        # Step 2: Build Downstream payloads
        closure_payload = _build_closure_payload(analysis, latitude, longitude, zone, corridor, junction, event_cause, event_type)
        disruption_payload = _build_disruption_payload(analysis, closure_payload)

        # Step 3: Road Closure Prediction
        road_closure_req, closure_conf, closure_risk = False, 0.0, "Unknown"
        if RC_BUNDLE:
            try:
                raw_df = rc.build_row(closure_payload)
                X = rc.run_preprocess(raw_df, RC_BUNDLE)
                prob = float(RC_BUNDLE["model"].predict_proba(X)[0][1])
                road_closure_req = bool(prob >= RC_THRESHOLD)
                closure_conf = round(float(prob), 3)
                closure_risk = "Low" if prob < 0.3 else ("Medium" if prob < 0.6 else "High")
            except Exception as e:
                print(f"Road closure prediction error: {e}")

        # Step 4: Disruption Prediction
        severity_label, severity_conf, eta_range = "Unknown", "N/A", "Unknown"
        if DISRUPTION_MODEL:
            try:
                feature_row = disruption_app.build_feature_row(disruption_payload)
                pred_class = int(DISRUPTION_MODEL.predict(feature_row)[0])
                proba = DISRUPTION_MODEL.predict_proba(feature_row)[0].tolist()
                
                severity_label = disruption_app.LABELS[pred_class]
                severity_conf = f"{round(max(proba) * 100, 1)}%"
                
                eta_map = {
                    "<30 mins (Quick)":    "5 – 15 mins",
                    "30–90 mins (Minor)":  "30 – 90 mins",
                    "90–240 mins (Major)": "90 – 240 mins",
                    ">240 mins (Severe)":  "240+ mins",
                }
                eta_range = eta_map.get(severity_label, "Unknown")
            except Exception as e:
                print(f"Disruption prediction error: {e}")

        # Step 5: Shared fields
        emergency        = analysis.get("emergency", {})
        em_level         = emergency.get("level", "LOW")
        scene_type       = emergency.get("scene_type", "unknown")
        congestion_score = float(analysis.get("congestion_score", 0.0))
        people           = int(analysis.get("people", 0))
        vehicles         = int(analysis.get("vehicles", 0))

        # Step 6: Derived outputs
        heatmap_points  = _generate_heatmap_points(latitude, longitude, congestion_score)
        recommendations = _generate_recommendations(em_level, road_closure_req, severity_label, people, vehicles, congestion_score)
        police_estimate = _police_personnel_required(congestion_score, em_level, event_type, vehicles, people)

        nearest_station = None
        if police_estimate["personnel_required"] >= POLICE_STATION_LOOKUP_THRESHOLD:
            async with httpx.AsyncClient() as client:
                nearest_station = await _find_nearest_police_station(client, latitude, longitude)

        return JSONResponse({
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "incident": {
                "latitude": latitude, "longitude": longitude, "zone": zone, "corridor": corridor, "junction": junction,
                "road_block_reason": road_block_reason, "event_cause": event_cause, "event_type": event_type,
            },
            "detected_objects": {
                "total": analysis.get("total_objects", 0), "vehicles": vehicles, "people": people,
                "road_blocks": analysis.get("road_blocks", 0), "illegal_parking": analysis.get("illegal_parking", 0),
                "vehicle_types": analysis.get("vehicle_types", {}),
            },
            "congestion": {
                "level": analysis.get("congestion_level", "Unknown"), "score": round(congestion_score, 3),
                "percentage": analysis.get("lane_occupancy_pct", _congestion_pct(congestion_score)), "emergency_level": em_level,
                "scene_type": scene_type, "emergency_reasons": emergency.get("reasons", []),
            },
            "predictions": {
                "road_closure_required": road_closure_req, "closure_confidence": closure_conf, "closure_risk_level": closure_risk,
                "disruption_severity": severity_label, "disruption_confidence": severity_conf, "expected_delay": eta_range,
            },
            "map": {"heatmap_points": heatmap_points},
            "dispatch": {
                "police_personnel_required": police_estimate["personnel_required"], "personnel_bracket": police_estimate["bracket"],
                "personnel_breakdown": police_estimate["breakdown"], "station_lookup_threshold": POLICE_STATION_LOOKUP_THRESHOLD,
                "nearest_police_station": nearest_station,
            },
            "recommendations": recommendations,
        })
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTING ENDPOINTS (Grouped under /routing)
# ─────────────────────────────────────────────────────────────────────────────
routing_router = APIRouter(prefix="/routing", tags=["Routing"])

@routing_router.post("/nearest-main-road")
def nearest_main_road_endpoint(data: MainRoadRequest):
    result = routing.route_to_nearest_main_road(routing.G, data.current_lat, data.current_lon)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"status": "success", "data": result}

@routing_router.post("/local-bypass")
def local_bypass_endpoint(data: BypassRequest):
    result = routing.get_immediate_local_bypass(routing.G, data.accident_lat, data.accident_lon)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"status": "success", "data": result}

@routing_router.post("/event-aware-route")
def event_aware_route_endpoint(data: EventAwareRouteRequest):
    result = routing.get_event_aware_route(
        routing.G, data.origin_lat, data.origin_lon, data.destination_lat, data.destination_lon,
        avoid_active_closures=data.avoid_active_closures,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"status": "success", "data": result}

app.include_router(routing_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
