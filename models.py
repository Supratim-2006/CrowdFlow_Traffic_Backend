"""
models.py — Pydantic request/response schemas for CrowdFlow AI Traffic Diversion API
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────

class MainRoadRequest(BaseModel):
    """Find the top-3 nearest main roads from the user's current location."""
    current_lat: float = Field(..., example=12.9716, description="Current latitude")
    current_lon: float = Field(..., example=77.5946, description="Current longitude")


class BypassRequest(BaseModel):
    """Find a local bypass route around a blocked accident road."""
    accident_lat: float = Field(..., example=13.0012, description="Accident latitude")
    accident_lon: float = Field(..., example=77.5873, description="Accident longitude")


class EventAwareRouteRequest(BaseModel):
    """
    Route from origin to destination weighted by live event severity.
    Active road-closure edges are removed before routing by default.
    """
    origin_lat:              float = Field(..., example=12.9716)
    origin_lon:              float = Field(..., example=77.5946)
    destination_lat:         float = Field(..., example=13.0827)
    destination_lon:         float = Field(..., example=77.5877)
    avoid_active_closures:   bool  = Field(
        True,
        description="Remove edges with active road closures before routing"
    )


# ── Legacy model (kept for backward compatibility) ────────────────────────────

class RouteRequest(BaseModel):
    """
    Original three-point route request model.
    Retained for any consumers still using the old schema.
    Use EventAwareRouteRequest for new integrations.
    """
    accident_lat:     float
    accident_lon:     float
    source_lat:       float
    source_lon:       float
    destination_lat:  float
    destination_lon:  float


# ── Response sub-models (optional — for OpenAPI docs) ────────────────────────

class DirectionStep(BaseModel):
    instruction:      str
    street:           str
    distance_meters:  float
    event_alert:      Optional[str] = "none"
    event_weight:     Optional[float] = 0.0


class IncidentEdge(BaseModel):
    from_node:        int
    to_node:          int
    dominant_cause:   str
    event_weight:     float