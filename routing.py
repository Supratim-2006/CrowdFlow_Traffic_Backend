"""
routing.py — CrowdFlow AI Traffic Diversion API
================================================
Loads the Bengaluru road graph, enriches edges with live event weights
from the Astram CSV, and exposes three routing functions:

  route_to_nearest_main_road   — top-3 nearest main roads from current position
  get_immediate_local_bypass   — bypass a blocked road after an accident
  get_event_aware_route        — Dijkstra weighted by real-time event severity
"""

import os
import gdown
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Config ───────────────────────────────────────────────────────────────────
GRAPHML_PATH = os.environ.get("GRAPHML_PATH", "bengaluru_crowdflow.graphml")
EVENTS_CSV   = os.environ.get("EVENTS_CSV",
    "data.csv"
)
SNAP_DIST_M  = float(os.environ.get("SNAP_DIST_M", "50"))   # metres

# Severity weights — used to inflate travel_time on affected edges
CAUSE_WEIGHT = {
    "accident":          3.0,
    "road_conditions":   2.5,
    "construction":      2.0,
    "water_logging":     2.0,
    "tree_fall":         1.8,
    "pot_holes":         1.5,
    "congestion":        1.5,
    "public_event":      1.2,
    "procession":        1.2,
    "vip_movement":      1.0,
    "protest":           1.0,
    "vehicle_breakdown": 0.8,
    "debris":            0.7,
    "others":            0.5,
}
PRIORITY_WEIGHT   = {"high": 1.5, "low": 0.5}
CLOSURE_WEIGHT    = 4.0   # flat penalty added when road is physically closed
ACTIVE_MULTIPLIER = 1.3   # extra penalty for still-active events

MAIN_ROAD_TYPES = {"motorway", "trunk", "primary", "secondary", "tertiary"}


# ── 1. Download graph if needed ───────────────────────────────────────────────
graph_dir = os.path.dirname(GRAPHML_PATH) or "."
if graph_dir and not os.path.isdir(graph_dir):
    os.makedirs(graph_dir, exist_ok=True)

if not os.path.exists(GRAPHML_PATH):
    print(f"Downloading {GRAPHML_PATH} from Google Drive …")
    gdown.download(
        "https://drive.google.com/uc?id=1i73fKSwvJi-DPvCUAQViRZb0mOgv42LS",
        GRAPHML_PATH,
        quiet=False
    )

# ── 2. Load raw OSMnx graph ───────────────────────────────────────────────────
print(f"Loading graph from {GRAPHML_PATH} …")
raw_G: ox.graph = ox.load_graphml(GRAPHML_PATH)

# ── 3. Load & preprocess events ───────────────────────────────────────────────
def _load_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["latitude", "longitude"])
    # Keep Bengaluru bounding box
    df = df[
        df["latitude"].between(12.7, 13.35)
        & df["longitude"].between(77.25, 77.85)
    ].copy()
    df["event_cause"] = (
        df["event_cause"].str.strip().str.lower().fillna("others")
    )
    df["priority"] = df["priority"].str.lower().fillna("low")
    df["status"]   = df["status"].str.lower().fillna("closed")
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.lower() == "true"
    )
    return df.reset_index(drop=True)


def _snap_events_to_edges(graph, df: pd.DataFrame, snap_dist: float) -> dict:
    """Returns {(u,v): [row_indices]} mapping events to nearest edges.
    Uses plain (u,v) keys since graph is a DiGraph (no parallel edges)."""
    edges = list(graph.edges(data=True))          # (u, v, data) — no keys arg
    mid_lats, mid_lons = [], []
    for u, v, d in edges:
        n1, n2 = graph.nodes[u], graph.nodes[v]
        mid_lats.append((n1["y"] + n2["y"]) / 2)
        mid_lons.append((n1["x"] + n2["x"]) / 2)

    lat0  = float(np.mean(mid_lats))
    m_lat = 111_320.0
    m_lon = 111_320.0 * np.cos(np.radians(lat0))

    tree = cKDTree(np.column_stack([
        np.array(mid_lats) * m_lat,
        np.array(mid_lons) * m_lon,
    ]))
    event_xy = np.column_stack([
        df["latitude"].values  * m_lat,
        df["longitude"].values * m_lon,
    ])
    dists, idxs = tree.query(event_xy, k=1, workers=-1)

    mapping: dict = {}
    for evt_i, (d, ei) in enumerate(zip(dists, idxs)):
        if d <= snap_dist:
            key = (edges[ei][0], edges[ei][1])  # (u, v) — DiGraph has no parallel edges
            mapping.setdefault(key, []).append(evt_i)
    return mapping


def _edge_event_weight(rows: pd.DataFrame) -> float:
    """Aggregate a group of events on one edge into a single weight scalar."""
    w = 0.0
    for _, r in rows.iterrows():
        cause = str(r.get("event_cause", "others"))
        ew    = CAUSE_WEIGHT.get(cause, 0.5)
        ew   *= PRIORITY_WEIGHT.get(str(r.get("priority", "low")), 0.5)
        if r.get("requires_road_closure"):
            ew += CLOSURE_WEIGHT
        if str(r.get("status", "closed")) == "active":
            ew *= ACTIVE_MULTIPLIER
        w += ew
    return round(w, 4)


# Build events dataframe and edge→event mapping
_events_df    = _load_events(EVENTS_CSV) if os.path.exists(EVENTS_CSV) else pd.DataFrame()
_edge_evt_map: dict = {}   # populated when graph is built below


# ── 4. Build enriched DiGraph ─────────────────────────────────────────────────
def _build_graph(raw_graph, events_df: pd.DataFrame, snap_dist: float) -> nx.DiGraph:
    """
    Convert the OSMnx MultiDiGraph to a plain DiGraph, keeping the shortest
    parallel edge and adding event-aware travel-time weights.
    """
    G = nx.DiGraph()

    # Copy nodes
    for node, data in raw_graph.nodes(data=True):
        G.add_node(node, **data)

    # Collapse parallel edges — keep shortest length
    for u, v, data in raw_graph.edges(data=True):
        length   = float(data.get("length", 1.0))
        speed    = float(data.get("speed_kph", 30.0))
        tt       = float(data.get("travel_time", length / (speed * 1000 / 3600)))
        name     = data.get("name", "")
        highway  = data.get("highway", "residential")

        if G.has_edge(u, v):
            if length < G[u][v]["length"]:
                G[u][v].update(
                    length=length, travel_time=tt,
                    name=name, highway=highway
                )
        else:
            G.add_edge(u, v,
                length=length,
                travel_time=tt,
                name=name,
                highway=highway,
                event_weight=0.0,
                travel_time_weighted=tt,
                event_count=0,
                active_closure=False,
                dominant_cause="none",
            )

    # Snap events onto edges and enrich
    if not events_df.empty:
        mapping = _snap_events_to_edges(G, events_df, snap_dist)
        for (u, v), row_idxs in mapping.items():
            if not G.has_edge(u, v):
                continue
            rows = events_df.iloc[row_idxs]
            ew   = _edge_event_weight(rows)
            base = G[u][v]["travel_time"]
            has_active_closure = bool(
                ((rows["requires_road_closure"]) & (rows["status"] == "active")).any()
            )
            cause_counts = rows["event_cause"].value_counts()
            G[u][v].update(
                event_weight=ew,
                travel_time_weighted=round(base * (1 + ew), 4),
                event_count=len(rows),
                active_closure=has_active_closure,
                dominant_cause=cause_counts.index[0] if len(cause_counts) else "none",
            )
        snapped = sum(len(v) for v in mapping.values())
        print(f"  Snapped {snapped:,}/{len(events_df):,} events onto graph edges")

    print(f"  Graph ready: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


G = _build_graph(raw_G, _events_df, SNAP_DIST_M)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _street_name(val) -> str:
    if isinstance(val, list):
        return val[0] if val else ""
    return val or ""


def _highway_type(val) -> str:
    if isinstance(val, list):
        return val[0] if val else "residential"
    return val or "residential"


def _is_main_road(highway) -> bool:
    hw = _highway_type(highway)
    return hw in MAIN_ROAD_TYPES


def build_directions(path: list, graph: nx.DiGraph) -> list[dict]:
    """Convert a node-path into turn-by-turn instruction steps."""
    directions = []
    current_street = None
    street_distance = 0.0

    for i in range(len(path) - 1):
        data        = graph.get_edge_data(path[i], path[i + 1]) or {}
        street_name = _street_name(data.get("name", "")) or "Local Road"
        seg_len     = float(data.get("length", 0))

        if street_name != current_street:
            if current_street is not None:
                directions.append({
                    "instruction":      f"Continue onto {street_name}",
                    "street":           street_name,
                    "distance_meters":  round(seg_len, 2),
                    "event_alert":      data.get("dominant_cause", "none"),
                    "event_weight":     data.get("event_weight", 0.0),
                })
            else:
                directions.append({
                    "instruction":      f"Head along {street_name}",
                    "street":           street_name,
                    "distance_meters":  round(seg_len, 2),
                    "event_alert":      data.get("dominant_cause", "none"),
                    "event_weight":     data.get("event_weight", 0.0),
                })
            current_street  = street_name
            street_distance = seg_len
        else:
            street_distance                        += seg_len
            directions[-1]["distance_meters"]       = round(street_distance, 2)
            # Surface the worst event on this merged segment
            if data.get("event_weight", 0) > directions[-1].get("event_weight", 0):
                directions[-1]["event_alert"]  = data.get("dominant_cause", "none")
                directions[-1]["event_weight"] = data.get("event_weight", 0.0)

    return directions


def get_current_road_name(lon: float, lat: float) -> str:
    """Return the name of the road nearest to (lon, lat)."""
    u, v, _ = ox.distance.nearest_edges(raw_G, lon, lat)
    data = G.get_edge_data(u, v) or {}
    return _street_name(data.get("name", ""))


def build_main_road_node_map(graph: nx.DiGraph,
                              exclude_road_name: str | None = None) -> dict:
    """{ node_id: road_name } for every node that sits on a main road."""
    node_map = {}
    for u, v, data in graph.edges(data=True):
        if not _is_main_road(data.get("highway", "")):
            continue
        road_name = _street_name(data.get("name", "")) or "Unnamed Main Road"
        if exclude_road_name and road_name == exclude_road_name:
            continue
        node_map.setdefault(u, road_name)
        node_map.setdefault(v, road_name)
    return node_map


def _path_stats(path: list, graph: nx.DiGraph) -> tuple[float, float, int]:
    """Returns (total_length_m, total_weighted_tt_s, active_closures)."""
    total_len, total_tt, closures = 0.0, 0.0, 0
    for i in range(len(path) - 1):
        d = graph.get_edge_data(path[i], path[i + 1]) or {}
        total_len += float(d.get("length", 0))
        total_tt  += float(d.get("travel_time_weighted", d.get("travel_time", 0)))
        if d.get("active_closure"):
            closures += 1
    return round(total_len, 2), round(total_tt, 2), closures


def _is_valid_main_road_path(path: list) -> bool:
    """Ignore snap points that already sit on a main-road node (zero-length routes)."""
    return len(path) >= 2


# ── Endpoint 1: Nearest main road (top 3) ────────────────────────────────────
def route_to_nearest_main_road(graph: nx.DiGraph,
                                current_lat: float,
                                current_lon: float) -> dict:
    """
    From the user's current position find the top-3 nearest main roads
    (excluding the road they are currently on).  Uses travel_time_weighted
    so heavier-incident roads are ranked lower.
    """
    try:
        current_road  = get_current_road_name(current_lon, current_lat)
        main_node_map = build_main_road_node_map(graph, exclude_road_name=current_road)

        if not main_node_map:
            return {"error": "No other main roads found nearby."}

        start_node = ox.distance.nearest_nodes(raw_G, current_lon, current_lat)

        distances, paths = nx.single_source_dijkstra(
            graph,
            source=start_node,
            weight="travel_time_weighted",
            cutoff=600,          # 10-minute travel-time horizon
        )

        seen_roads: dict = {}
        for node, tt in sorted(distances.items(), key=lambda x: x[1]):
            if node not in main_node_map:
                continue
            path = paths[node]
            if not _is_valid_main_road_path(path):
                continue
            road_name = main_node_map[node]
            if road_name in seen_roads:
                continue
            seen_roads[road_name] = {"travel_time": tt, "path": path}
            if len(seen_roads) == 3:
                break

        if not seen_roads:
            return {"error": "No main road reachable within 10 minutes."}

        results = []
        for road_name, info in seen_roads.items():
            path = info["path"]
            length_m, tt_s, closures = _path_stats(path, graph)
            directions = build_directions(path, graph)
            directions.append({
                "instruction":     f"Arrive at {road_name}",
                "street":          road_name,
                "distance_meters": 0,
                "event_alert":     "none",
                "event_weight":    0.0,
            })
            results.append({
                "target_main_road":        road_name,
                "distance_meters":         length_m,
                "estimated_travel_time_s": round(tt_s, 1),
                "active_closures_on_path": closures,
                "directions":              directions,
                "nodes":                   path,
            })

        results.sort(key=lambda x: x["estimated_travel_time_s"])

        return {
            "currently_on": current_road or "Unknown Road",
            "nearest_roads": results,
        }

    except Exception as e:
        return {"error": str(e)}


# ── Endpoint 2: Accident bypass ───────────────────────────────────────────────
def get_immediate_local_bypass(graph: nx.DiGraph,
                                accident_lat: float,
                                accident_lon: float) -> dict:
    """
    Given an accident location, block the entire accident road, then
    return the nearest reachable main road with event-aware directions.
    """
    try:
        blocked_road = get_current_road_name(accident_lon, accident_lat)

        # Remove all edges belonging to the blocked road
        local_graph = graph.copy()
        to_remove = [
            (u, v)
            for u, v, d in local_graph.edges(data=True)
            if _street_name(d.get("name", "")) == blocked_road and blocked_road
        ]
        local_graph.remove_edges_from(to_remove)

        main_node_map = build_main_road_node_map(
            local_graph, exclude_road_name=blocked_road
        )
        if not main_node_map:
            return {"error": "No accessible main road found after blocking the accident road."}

        start_node = ox.distance.nearest_nodes(raw_G, accident_lon, accident_lat)
        distances, paths = nx.single_source_dijkstra(
            local_graph,
            source=start_node,
            weight="travel_time_weighted",
            cutoff=600,
        )

        best = None
        for node, tt in sorted(distances.items(), key=lambda x: x[1]):
            if node not in main_node_map:
                continue
            path = paths[node]
            if not _is_valid_main_road_path(path):
                continue
            best = (node, tt, path)
            break

        if not best:
            return {"error": "No main road reachable within 10 minutes after blocking accident road."}

        target_node, tt, path = best
        target_road = main_node_map[target_node]

        length_m, tt_s, closures = _path_stats(path, local_graph)
        directions = build_directions(path, local_graph)
        directions.append({
            "instruction":     f"Arrive at {target_road}",
            "street":          target_road,
            "distance_meters": 0,
            "event_alert":     "none",
            "event_weight":    0.0,
        })

        return {
            "blocked_road":            blocked_road or "Unknown Road",
            "edges_removed":           len(to_remove),
            "target_main_road":        target_road,
            "distance_meters":         length_m,
            "estimated_travel_time_s": round(tt_s, 1),
            "active_closures_on_path": closures,
            "directions":              directions,
            "nodes":                   path,
        }

    except nx.NetworkXNoPath:
        return {"error": "No route found — accident road may have caused a dead end."}
    except Exception as e:
        return {"error": str(e)}


# ── Endpoint 3: Event-aware A→B route ────────────────────────────────────────
def get_event_aware_route(graph: nx.DiGraph,
                           origin_lat: float,
                           origin_lon: float,
                           dest_lat: float,
                           dest_lon: float,
                           avoid_active_closures: bool = True) -> dict:
    """
    Dijkstra from origin → destination weighted by travel_time_weighted.
    Optionally removes edges with active road closures before routing.
    Returns the event-aware route alongside a free-flow baseline for comparison.
    """
    try:
        origin_node = ox.distance.nearest_nodes(raw_G, origin_lon, origin_lat)
        dest_node   = ox.distance.nearest_nodes(raw_G, dest_lon,   dest_lat)

        routing_graph = graph
        closures_removed = 0
        if avoid_active_closures:
            routing_graph = graph.copy()
            to_remove = [
                (u, v)
                for u, v, d in routing_graph.edges(data=True)
                if d.get("active_closure")
            ]
            routing_graph.remove_edges_from(to_remove)
            closures_removed = len(to_remove)

        # Event-aware path
        try:
            tt_weighted, path_weighted = nx.single_source_dijkstra(
                routing_graph, origin_node, target=dest_node,
                weight="travel_time_weighted"
            )
        except nx.NetworkXNoPath:
            # Fall back to graph with closures re-enabled
            tt_weighted, path_weighted = nx.single_source_dijkstra(
                graph, origin_node, target=dest_node,
                weight="travel_time_weighted"
            )
            closures_removed = 0

        # Free-flow baseline (travel_time, no event penalty)
        try:
            tt_free, path_free = nx.single_source_dijkstra(
                graph, origin_node, target=dest_node,
                weight="travel_time"
            )
        except nx.NetworkXNoPath:
            tt_free, path_free = None, []

        length_m, tt_s, closures_on = _path_stats(path_weighted, routing_graph)
        directions = build_directions(path_weighted, routing_graph)
        directions.append({
            "instruction":     "You have arrived at your destination",
            "street":          "",
            "distance_meters": 0,
            "event_alert":     "none",
            "event_weight":    0.0,
        })

        # Incident summary along the route
        incident_edges = [
            {
                "from_node":     path_weighted[i],
                "to_node":       path_weighted[i + 1],
                "dominant_cause": routing_graph.get_edge_data(
                    path_weighted[i], path_weighted[i + 1], {}
                ).get("dominant_cause", "none"),
                "event_weight":   routing_graph.get_edge_data(
                    path_weighted[i], path_weighted[i + 1], {}
                ).get("event_weight", 0.0),
            }
            for i in range(len(path_weighted) - 1)
            if (routing_graph.get_edge_data(
                path_weighted[i], path_weighted[i + 1], {}
            ).get("event_weight", 0.0) or 0) > 0
        ]

        origin_road = get_current_road_name(origin_lon, origin_lat)
        dest_road   = get_current_road_name(dest_lon,   dest_lat)

        return {
            "origin_road":             origin_road or "Unknown",
            "destination_road":        dest_road   or "Unknown",
            "distance_meters":         length_m,
            "estimated_travel_time_s": round(tt_s, 1),
            "free_flow_travel_time_s": round(tt_free, 1) if tt_free else None,
            "delay_due_to_events_s":   round(tt_s - tt_free, 1) if tt_free else None,
            "active_closures_avoided": closures_removed,
            "active_closures_on_path": closures_on,
            "incident_edges":          incident_edges,
            "directions":              directions,
            "nodes":                   path_weighted,
        }

    except Exception as e:
        return {"error": str(e)}