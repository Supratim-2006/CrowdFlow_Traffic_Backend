"""
CrowdFlow AI — Road Congestion & Emergency Detector
=====================================================
Detects from a road/traffic image:
  1. Number of vehicles  (cars, bikes, buses, trucks)
  2. Number of people    (pedestrians, crowd)
  3. Number of road blocks (barricades, cones, barriers)
  4. Number of illegal parking instances
  5. Emergency level + reason
  6. Scene type: accident / crowding / celebration / normal
  7. *** Lane occupancy % — pixel-accurate road fill via YOLOv8-seg (NEW) ***

Congestion is now measured by LANE OCCUPANCY RATIO, not vehicle count:
  - Runs YOLOv8n-seg (swap-in, same API as YOLOv8n) to get instance masks
  - Road mask: semantic road pixels from the seg model's road/sidewalk classes,
    OR a perspective-trapezoid fallback when seg is unavailable
  - Vehicle footprint: intersection of each vehicle's pixel mask with the road mask
  - Occupancy = total vehicle-on-road pixels / total road pixels
  - A 4-truck scene with 60% occupancy beats a 10-car highway scene at 20%

Emergency Detection Logic:
  - CRITICAL: Vehicles with abnormal overlap (crash geometry), many people
               surrounding stopped vehicles, or heavy road blockage
  - HIGH:     Multiple stopped vehicles blocking lane, dense crowd on road
  - MODERATE: Unusual pedestrian counts near vehicles, partial blockage
  - LOW:      Minor irregularities but no immediate danger
  - NONE:     Normal traffic flow

Model choice:
  - yolov8n-seg.pt  (default) — fast, gives masks, ~6 MB
  - yolov8x-seg.pt            — accurate, slower, ~137 MB
  Pass --model yolov8n-seg.pt (or set MODEL_PATH in download_model.py)

Usage:
    python crowdflow_detector.py --image road.jpg
    python crowdflow_detector.py --image road.jpg --model yolov8x-seg.pt --device cuda --output result.jpg --show
    python crowdflow_detector.py --webcam
"""

import cv2
import numpy as np
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

VEHICLE_IDS = {
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}
PERSON_ID = 0
CONE_IDS  = {9: "traffic_light", 11: "stop_sign"}

# ── Seg-model road/sidewalk class IDs (COCO-Stuff / ADE20K not used here) ────
# YOLOv8-seg uses the same 80-class COCO set as YOLOv8-det — there is no
# dedicated "road" class in COCO.  We therefore derive the drivable mask from
# the COMPLEMENT of non-road detections: we build a binary road mask using a
# perspective trapezoid, then ERASE pixels that belong to detected buildings,
# sky, vegetation etc. (handled in _RoadMaskEstimator below).
# If you later swap in a Cityscapes/ADE20K seg model, set these to its IDs:
ROAD_SEG_CLASS_IDS: set = set()   # populated automatically when seg available

# ── Emergency-vehicle COCO proxies ───────────────────────────────────────────
# COCO has no "ambulance" class. We detect large trucks/buses in white/light
# tones at the scene edges as a proxy — handled in EmergencyAnalyser via
# the EMERGENCY_VEHICLE_LABELS list which maps detected labels to the concept.
# Additionally: "truck" at a road scene + uniformed people = strong signal.
EMERGENCY_VEHICLE_LABELS = {"truck", "bus"}   # large vehicles = possible fire/ambulance
EMERGENCY_PERSON_LABELS  = {"person"}         # all persons count; ratio does the work

CONF_VEHICLE = 0.22   # NOTE: kept low intentionally to catch damaged/occluded
CONF_PERSON  = 0.22   # vehicles at real accident scenes. This is a recall/precision
CONF_BLOCK   = 0.18   # trade-off — see EmergencyAnalyser corroboration gate below,
                       # which compensates by requiring more evidence before
                       # escalating to ACCIDENT/CRITICAL, instead of starving
                       # detection itself.

PARKING_ZONE_RATIO = 0.94   # raised: car bottom must be very close to frame bottom
PARKING_EDGE_RATIO = 0.07   # tightened: only flag cars extremely close to frame edge
                            # (≤7% from edge). At 12%, ANY curbside car on a typical
                            # street photo gets flagged — that is normal parking, not
                            # illegal. This is still a FRAMING proxy without lane-marking
                            # awareness; keep this informational, not punitive in scoring.

# ── Emergency tuning knobs ───────────────────────────────────────────────────
CRASH_IOU_THRESHOLD    = 0.15   # raised from 0.08. NOTE: bbox overlap alone cannot
                                 # distinguish a collision from bumper-to-bumper
                                 # parked/queued cars — they produce identical box
                                 # geometry. This threshold change only reduces
                                 # noise; the real fix is the corroboration gate
                                 # below (DENSE_SCENE_VEHICLE_THRESHOLD).

# Minimum IoU-distance (centre-to-centre / avg-box-size) for proximity crash
CRASH_PROXIMITY_RATIO  = 1.0    # tightened from 1.4 — was "very close" was wide
                                 # enough to flag almost every car in a curb-side row

CROWD_VEHICLE_RATIO    = 0.40
INCIDENT_PPV_RATIO     = 2.0    # lowered: 2 people per vehicle is already unusual
CROWDING_PERSON_MIN    = 12     # lowered from 15

# NEW: large-vehicle + high PPV = likely emergency-responder scene
RESPONDER_PPV_THRESHOLD = 1.5   # ≥ 1.5 people per large vehicle = responders on scene

# Debris heuristic — DISABLED from driving classification (see analyse()).
# As written this only measures "small/far already-classified objects", not
# validated debris, so it produced false signals on any wide street scene with
# distant cars/pedestrians. Kept here for reference / informational reporting only.
DEBRIS_AREA_FRACTION   = 0.003
DEBRIS_MIN_COUNT       = 4

# ── NEW: density + corroboration gate ────────────────────────────────────────
# A street scene with many vehicles (parking lot, busy curb, traffic queue) will
# always produce some touching/close bounding-box pairs — that is expected, not
# anomalous. Raw box-overlap is therefore only trusted as a crash signal on its
# own when the scene is sparse. Above this vehicle count, overlap must be
# corroborated by an independent signal (people clustered tightly around the
# SAME vehicles, or road barriers present) before it can escalate to
# ACCIDENT / HIGH / CRITICAL.
DENSE_SCENE_VEHICLE_THRESHOLD = 8
CORROBORATION_PEOPLE_NEAR_MIN = 2
CORROBORATION_ROAD_BLOCKS_MIN = 1

# ── Colours (BGR) ────────────────────────────
COLORS = {
    "vehicle":   (255, 165,   0),   # orange
    "person":    ( 50, 200,  50),   # green
    "roadblock": (  0, 100, 255),   # blue-red
    "parking":   (  0,   0, 255),   # red
    "crashed":   (  0,   0, 200),   # deep red  ← NEW
}

# Emergency badge colours (BGR)
EMERGENCY_COLORS = {
    "CRITICAL": (  0,   0, 220),
    "HIGH":     (  0,  80, 255),
    "MODERATE": (  0, 165, 255),
    "LOW":      (  0, 215, 255),
    "NONE":     ( 50, 180,  50),
}

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Detection:
    category:   str
    label:      str
    confidence: float
    bbox:       List[int]
    is_illegal_parking: bool = False
    is_crashed:         bool = False   # NEW


@dataclass
class EmergencyAssessment:
    level:                str
    scene_type:           str
    reasons:              List[str]
    crash_pairs:          int
    crash_proximity:      int
    people_near_vehicles: int
    emergency_vehicles:   int
    debris_count:         int

    def to_dict(self):
        return {
            "level":                    self.level,
            "scene_type":               self.scene_type,
            "reasons":                  self.reasons,
            "crash_pairs_iou":          self.crash_pairs,
            "crash_pairs_proximity":    self.crash_proximity,
            "people_near_vehicles":     self.people_near_vehicles,
            "emergency_vehicles_proxy": self.emergency_vehicles,
            "debris_objects":           self.debris_count,
        }


@dataclass
class AnalysisReport:
    image_path:       str
    image_size:       Tuple[int, int]
    inference_ms:     float
    total_objects:    int
    vehicles:         int = 0
    people:           int = 0
    road_blocks:      int = 0
    illegal_parking:  int = 0
    vehicle_types:    Dict[str, int] = field(default_factory=dict)
    congestion_level: str = "Unknown"
    lane_occupancy:   float = 0.0   # NEW: 0.0–1.0, fraction of road pixels covered by vehicles
    emergency:        Optional[EmergencyAssessment] = None
    detections:       List[Detection] = field(default_factory=list)

    def congestion_score(self) -> float:
        """
        Occupancy-primary composite score, 0–10.

        Primary signal (6 pts): lane_occupancy — the fraction of actual road
        pixels covered by vehicle masks/footprints. This is perspective-correct
        and scene-size-agnostic: 4 trucks filling a narrow lane scores identically
        to what it looks like, regardless of whether 10 other cars are parked
        on the shoulder outside the drivable zone.

        Secondary signals (4 pts total):
          - people density on road  (1.5 pts)
          - road barriers present   (1.5 pts)
          - illegal parking nudge   (0.5 pts) — low weight, noisy proxy signal
            (no lane-marking awareness; kept informational not determinative)

        Score bands → _grade():
          < 2.5  Clear      occupancy < ~40%
          < 4.5  Light      occupancy ~40–60%
          < 6.5  Moderate   occupancy ~60–80%
          < 8.5  Congested  occupancy ~80–95%
          ≥ 8.5  Gridlock   lanes physically blocked
        """
        occ_score     = self.lane_occupancy * 6.0
        people_score  = min(self.people,          30) / 30 * 1.5
        block_score   = min(self.road_blocks,      5) /  5 * 1.5
        parking_score = min(self.illegal_parking,  5) /  5 * 0.5   # nudge only
        return round(occ_score + people_score + block_score + parking_score, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["congestion_score"]   = self.congestion_score()
        d["lane_occupancy_pct"] = round(self.lane_occupancy * 100, 1)
        d.pop("detections")
        if self.emergency:
            d["emergency"] = self.emergency.to_dict()
        return d


# ─────────────────────────────────────────────
# EMERGENCY ANALYSER  (NEW MODULE)
# ─────────────────────────────────────────────

class EmergencyAnalyser:
    """
    Analyses spatial relationships between detections to determine
    whether the scene constitutes an emergency.

    Heuristics used (no extra model needed — pure geometry):

    1. CRASH DETECTION
       Compute pairwise IoU between all vehicle bounding boxes.
       IoU > CRASH_IOU_THRESHOLD  →  vehicles are overlapping / touching,
       which in a top-down or street-level view strongly suggests a collision.

    2. PEOPLE CLUSTERING NEAR VEHICLES
       Count pedestrians whose centre falls within an expanded (×1.5)
       version of any vehicle box. A crowd around a stopped vehicle is
       a classic post-accident signature.

    3. ROAD BLOCKAGE RATIO
       Total horizontal span of vehicle boxes vs image width.
       High ratio + road blocks = blocked road → emergency.

    4. SCENE TYPE CLASSIFICATION
       accident    : crash pairs > 0  OR  road_blocks ≥ 2 AND vehicles stopped
       crowding    : people > CROWDING_PERSON_MIN with high PPV ratio
       celebration : many people, low road_block, low crash, spread spatially
       normal      : everything else
    """

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _box_centre(bbox: List[int]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def _expand_box(bbox: List[int], scale: float,
                    img_w: int, img_h: int) -> List[int]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw = (x2 - x1) / 2 * scale
        hh = (y2 - y1) / 2 * scale
        return [
            int(max(0,     cx - hw)),
            int(max(0,     cy - hh)),
            int(min(img_w, cx + hw)),
            int(min(img_h, cy + hh)),
        ]

    def analyse(self, report: AnalysisReport) -> EmergencyAssessment:
        img_w, img_h = report.image_size
        img_area     = img_w * img_h
        detections   = report.detections

        vehicle_dets = [d for d in detections
                        if d.category in ("vehicle", "parking", "crashed")]
        person_dets  = [d for d in detections if d.category == "person"]

        reasons: List[str] = []

        # ── SIGNAL 1: IoU overlap crash pairs ────────────────────────────────
        # Two vehicle boxes overlap in 2D → they are touching/on top of each other
        crash_pairs     = 0
        crashed_indices = set()
        vboxes = [d.bbox for d in vehicle_dets]

        for i in range(len(vboxes)):
            for j in range(i + 1, len(vboxes)):
                if self._iou(vboxes[i], vboxes[j]) > CRASH_IOU_THRESHOLD:
                    crash_pairs += 1
                    crashed_indices.add(i)
                    crashed_indices.add(j)

        for idx in crashed_indices:
            vehicle_dets[idx].is_crashed = True

        if crash_pairs > 0:
            reasons.append(
                f"{crash_pairs} vehicle box overlap(s) detected — possible collision"
            )

        # ── SIGNAL 2: Proximity crash (centre-distance) ───────────────────────
        # Vehicles very close but not overlapping — works for post-crash positions
        crash_proximity = 0
        for i in range(len(vboxes)):
            for j in range(i + 1, len(vboxes)):
                cx_i = (vboxes[i][0] + vboxes[i][2]) / 2
                cy_i = (vboxes[i][1] + vboxes[i][3]) / 2
                cx_j = (vboxes[j][0] + vboxes[j][2]) / 2
                cy_j = (vboxes[j][1] + vboxes[j][3]) / 2
                dist = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
                avg_w = ((vboxes[i][2] - vboxes[i][0]) +
                         (vboxes[j][2] - vboxes[j][0])) / 2
                if avg_w > 0 and (dist / avg_w) < CRASH_PROXIMITY_RATIO:
                    crash_proximity += 1
                    crashed_indices.add(i)
                    crashed_indices.add(j)

        # Mark proximity-flagged vehicles too
        for idx in crashed_indices:
            vehicle_dets[idx].is_crashed = True

        if crash_proximity > 0 and crash_pairs == 0:
            reasons.append(
                f"{crash_proximity} vehicle(s) in abnormally close proximity"
            )

        # ── SIGNAL 3: People clustering near vehicles ─────────────────────────
        people_near = 0
        for p in person_dets:
            pcx, pcy = self._box_centre(p.bbox)
            for v in vehicle_dets:
                exp = self._expand_box(v.bbox, 1.8, img_w, img_h)
                if exp[0] <= pcx <= exp[2] and exp[1] <= pcy <= exp[3]:
                    people_near += 1
                    break

        if people_near >= 3 and report.vehicles >= 1:
            reasons.append(
                f"{people_near} people clustered around vehicle(s)"
            )

        # ── GATE: density + corroboration ──────────────────────────────────
        # Bounding-box overlap/proximity between vehicles is geometrically
        # IDENTICAL whether the vehicles collided or are simply parked/queued
        # close together. In a dense scene (lots of vehicles — busy curb,
        # parking lot, traffic queue) some touching pairs are the expected
        # default, not an anomaly. We only let raw overlap drive an
        # ACCIDENT/CRITICAL call on its own when the scene is sparse; in a
        # dense scene we require an independent corroborating signal first
        # (a tight cluster of people around the SAME vehicles, or barriers
        # present) before treating overlap as evidence of a collision.
        dense_scene = report.vehicles >= DENSE_SCENE_VEHICLE_THRESHOLD
        crash_corroborated = (
            people_near >= CORROBORATION_PEOPLE_NEAR_MIN
            or report.road_blocks >= CORROBORATION_ROAD_BLOCKS_MIN
        )
        crash_trusted = (not dense_scene) or crash_corroborated

        if dense_scene and (crash_pairs > 0 or crash_proximity > 0) and not crash_corroborated:
            reasons.append(
                f"{report.vehicles} vehicles is a dense scene (parking/queueing "
                f"expected) — box overlap alone not treated as collision evidence "
                f"without people/barrier corroboration"
            )

        # ── SIGNAL 4: Road blockage ratio ────────────────────────────────────
        # FIXED: previously summed every vehicle box's width independently,
        # which double/triple-counts overlapping cars and trivially exceeds
        # 100% on any photo with several vehicles (e.g. cars on both curbs +
        # passing traffic), regardless of whether anything is actually
        # blocked. We instead take the union of horizontal x-spans, which
        # measures how much of the frame's width is actually covered.
        if vboxes:
            intervals = sorted((b[0], b[2]) for b in vboxes)
            merged_span = 0
            cur_s, cur_e = intervals[0]
            for s, e in intervals[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    merged_span += (cur_e - cur_s)
                    cur_s, cur_e = s, e
            merged_span += (cur_e - cur_s)
            blockage = min(merged_span / max(img_w, 1), 1.0)
        else:
            blockage = 0.0

        # NOTE: horizontal union span on ANY normal street photo with parked cars
        # on both curbs approaches 80-100% trivially — the cars span the full frame
        # width even with completely clear lanes. This signal is only meaningful
        # when barriers are ALSO present (confirming deliberate blockage), or when
        # blockage exceeds a very high threshold AND the scene is otherwise abnormal.
        if blockage > 0.65 and report.road_blocks >= 1:
            reasons.append(
                f"Road blockage ~{blockage:.0%} with {report.road_blocks} barrier(s)"
            )
        elif blockage > 0.92:   # raised from 0.80 — curb parking hits 84% on a clear road
            reasons.append(f"Road blockage ~{blockage:.0%} — lane(s) obstructed")

        # ── SIGNAL 5: Emergency-vehicle proxy ────────────────────────────────
        # Large vehicles (truck/bus label) at a congested scene with many people
        # proxies for ambulance/fire truck — COCO has no ambulance class
        large_vehicles = sum(
            1 for d in vehicle_dets
            if d.label in EMERGENCY_VEHICLE_LABELS
        )
        ppv = report.people / max(report.vehicles, 1)

        # Strong emergency-responder signature:
        # ≥1 large vehicle + ≥3 people per vehicle + blockage
        em_vehicle_signal = (
            large_vehicles >= 1
            and ppv >= RESPONDER_PPV_THRESHOLD
            and blockage > 0.40
        )
        if em_vehicle_signal:
            reasons.append(
                f"{large_vehicles} large vehicle(s) + {report.people} people on blocked road "
                f"— possible emergency responders on scene"
            )

        # ── SIGNAL 6: Debris field heuristic ─────────────────────────────────
        # Count all YOLO detections (any class) with very small bounding boxes
        # on the lower half of the image — scattered debris on road surface
        debris_count = sum(
            1 for d in detections
            if (d.bbox[3] > img_h * 0.4)                          # lower half
            and ((d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1])    # tiny box
                 < img_area * DEBRIS_AREA_FRACTION)
        )
        if debris_count >= DEBRIS_MIN_COUNT:
            reasons.append(
                f"{debris_count} small objects on road surface — possible debris/scatter"
            )

        # ── SIGNAL 7: PPV ratio ───────────────────────────────────────────────
        if ppv >= INCIDENT_PPV_RATIO and report.people >= 4:
            reasons.append(
                f"People-to-vehicle ratio {ppv:.1f}:1 — crowd gathering around vehicles"
            )

        # ── SIGNAL 8: Illegal parking under congestion ────────────────────────
        # Raised score threshold to 5.5 (was 4) — with the recalibrated scoring,
        # 5.5 represents a genuinely busy scene; at 4 this fired on any street
        # with a handful of parked cars on both sides.
        if report.illegal_parking >= 3 and report.congestion_score() >= 5.5:
            reasons.append(
                f"{report.illegal_parking} illegally parked vehicle(s) worsening scene"
            )

        # ── Scene type ────────────────────────────────────────────────────────
        scene_type = self._classify_scene(
            crash_pairs, crash_proximity, report,
            people_near, blockage, ppv,
            em_vehicle_signal, debris_count, crash_trusted
        )

        # ── Emergency level ───────────────────────────────────────────────────
        level = self._emergency_level(
            crash_pairs, crash_proximity, people_near, blockage,
            report, scene_type, em_vehicle_signal, debris_count, crash_trusted
        )

        return EmergencyAssessment(
            level=level,
            scene_type=scene_type,
            reasons=reasons if reasons else ["No emergency indicators detected"],
            crash_pairs=crash_pairs,
            crash_proximity=crash_proximity,
            people_near_vehicles=people_near,
            emergency_vehicles=large_vehicles,
            debris_count=debris_count,
        )

    # ── scene classifier ─────────────────────────────────────────────────────

    @staticmethod
    def _classify_scene(crash_pairs: int, crash_proximity: int,
                        report: AnalysisReport, people_near: int,
                        blockage: float, ppv: float,
                        em_vehicle_signal: bool, debris_count: int,
                        crash_trusted: bool) -> str:

        # ACCIDENT: crash geometry signal, but only when trusted (sparse scene,
        # or corroborated by people/barriers in a dense scene) — see crash_trusted
        if crash_pairs > 0 and crash_trusted:
            return "accident"

        # ACCIDENT: emergency-vehicle proxy + people clustered + road blocked
        if em_vehicle_signal and people_near >= 3:
            return "accident"

        # ACCIDENT: heavy blockage with barriers and many responder-like people
        if report.road_blocks >= 2 and blockage > 0.45 and people_near >= 4:
            return "accident"

        # ACCIDENT: proximity flagged (trusted) + high blockage (post-crash lane block)
        if crash_proximity >= 1 and blockage > 0.60 and crash_trusted:
            return "accident"

        # CROWDING: dense pedestrian presence, high PPV
        if report.people >= CROWDING_PERSON_MIN and ppv >= 2.0:
            return "crowding"

        # CELEBRATION: many people spread out, low blockage, no crash signals
        if (report.people >= CROWDING_PERSON_MIN
                and blockage < 0.35
                and report.road_blocks < 2
                and crash_pairs == 0
                and not em_vehicle_signal):
            return "celebration"

        return "normal"

    # ── level grader ─────────────────────────────────────────────────────────

    @staticmethod
    def _emergency_level(crash_pairs: int, crash_proximity: int,
                         people_near: int, blockage: float,
                         report: AnalysisReport, scene_type: str,
                         em_vehicle_signal: bool, debris_count: int,
                         crash_trusted: bool) -> str:

        # ── CRITICAL ─────────────────────────────────────────────────────────
        # Confirmed crash geometry + bystanders — only trusted (see crash_trusted)
        if crash_pairs >= 2 and crash_trusted:
            return "CRITICAL"
        if crash_pairs >= 1 and people_near >= 3:
            return "CRITICAL"

        # Emergency vehicles on scene + people clustered → active response
        if em_vehicle_signal and people_near >= 4:
            return "CRITICAL"

        # Full road block with barriers
        if blockage > 0.80 and report.road_blocks >= 2:
            return "CRITICAL"

        # ── HIGH ─────────────────────────────────────────────────────────────
        if crash_pairs == 1 and crash_trusted:
            return "HIGH"

        # Accident scene type with any supporting signal
        if scene_type == "accident":
            return "HIGH"

        # Emergency responder proxy without full CRITICAL signals
        if em_vehicle_signal and people_near >= 2:
            return "HIGH"

        # Close proximity crash
        if crash_proximity >= 1 and people_near >= 3:
            return "HIGH"

        # Multiple close-proximity vehicles, trusted (replaces old debris-based
        # rule — debris_count as implemented just measured small/far objects,
        # not validated debris, so it's no longer used to drive severity)
        if crash_proximity >= 2 and crash_trusted:
            return "HIGH"

        # Dense crowd blocking road
        if scene_type == "crowding" and blockage > 0.45:
            return "HIGH"

        if report.people >= 25 and report.vehicles >= 4:
            return "HIGH"

        # ── MODERATE ─────────────────────────────────────────────────────────
        # NOTE: blockage alone (vehicles spanning much of the frame width) is
        # the normal appearance of any street with traffic/curb parking on
        # both sides — it only means something paired with barriers present.
        if people_near >= 4 or report.road_blocks >= 3:
            return "MODERATE"
        if blockage > 0.55 and report.road_blocks >= 1:
            return "MODERATE"
        if scene_type == "crowding":
            return "MODERATE"
        if crash_proximity >= 1 and crash_trusted:
            return "MODERATE"
        if em_vehicle_signal:
            return "MODERATE"

        # ── LOW ──────────────────────────────────────────────────────────────
        # Raised illegal_parking threshold to 4 — at 2 this fired on almost any
        # street photo with cars on both curbs (which is the detector's normal FP
        # rate for curbside parking misclassified as "illegal").
        if report.illegal_parking >= 4 or people_near >= 2:
            return "LOW"

        return "NONE"


# ─────────────────────────────────────────────
# ROAD MASK + OCCUPANCY ESTIMATOR  (NEW)
# ─────────────────────────────────────────────

class _RoadMaskEstimator:
    """
    Computes a binary road mask and vehicle lane-occupancy ratio from
    YOLOv8-seg instance masks.

    Two-stage pipeline
    ──────────────────
    STAGE 1 — Road mask
      YOLOv8-seg returns per-instance masks for the same 80 COCO classes as
      YOLOv8-det.  COCO has no "road" class, so we invert: we start with a
      perspective-trapezoid prior (the lower-centre region that is almost
      always road in a forward-facing camera) and subtract pixels that are
      demonstrably NOT road — buildings, vegetation, sky, persons, and
      vehicles parked outside the lane.

      Trapezoid geometry (normalised, tunable):
        bottom: full image width, at y = img_h
        top:    40% of width, centred, at y = img_h * TOP_Y_FRAC
      This captures the vanishing-point perspective of a typical dashcam /
      CCTV street view without needing a separate segmentation model.

    STAGE 2 — Vehicle footprint on road
      For each detected vehicle we have either:
        (a) a pixel-accurate instance mask from YOLOv8-seg  ← preferred
        (b) a bounding-box rectangle weighted by vertical position ← fallback
      We intersect the vehicle mask/footprint with the road mask to get only
      the road pixels actually covered by that vehicle.

    STAGE 3 — Occupancy ratio
      occupancy = Σ vehicle_road_pixels / road_mask_pixels
      Clipped to [0, 1].  Values:
        < 0.20  → free-flowing
        0.20–0.45 → light
        0.45–0.65 → moderate
        0.65–0.85 → congested
        > 0.85  → gridlock / blocked
    """

    # Trapezoid parameters (fraction of image dimensions)
    TOP_Y_FRAC    = 0.38   # road starts here (top of trapezoid, near horizon)
    TOP_W_FRAC    = 0.38   # road width at horizon (fraction of img_w)
    BOTTOM_W_FRAC = 1.00   # road width at camera (full frame)

    def build_road_mask(self, img_h: int, img_w: int,
                        seg_results=None) -> np.ndarray:
        """
        Returns a uint8 binary mask (255 = road, 0 = non-road),
        shape (img_h, img_w).

        seg_results: ultralytics Results object from a -seg model (may be None).
        """
        mask = self._trapezoid_prior(img_h, img_w)

        if seg_results is not None:
            mask = self._subtract_non_road(mask, seg_results, img_h, img_w)

        return mask

    def vehicle_occupancy(self, road_mask: np.ndarray,
                          vehicle_dets: List,
                          seg_results=None,
                          img_h: int = 0,
                          img_w: int = 0) -> float:
        """
        Returns occupancy ratio in [0, 1].

        vehicle_dets: list of Detection objects with category vehicle/parking/crashed.
        seg_results:  ultralytics Results with .masks (may be None).
        """
        road_px = int(road_mask.sum() / 255)
        if road_px == 0:
            return 0.0

        # Build a combined vehicle footprint mask (same shape as road_mask)
        vehicle_mask = np.zeros_like(road_mask)

        if seg_results is not None and seg_results.masks is not None:
            vehicle_mask = self._masks_from_seg(
                vehicle_mask, seg_results, img_h, img_w)
        else:
            vehicle_mask = self._masks_from_bboxes(
                vehicle_mask, vehicle_dets, img_h, img_w)

        # Intersection: road pixels covered by vehicles
        on_road = cv2.bitwise_and(road_mask, vehicle_mask)
        covered = int(on_road.sum() / 255)

        return min(covered / road_px, 1.0)

    # ── private helpers ──────────────────────────────────────────────────────

    def _trapezoid_prior(self, img_h: int, img_w: int) -> np.ndarray:
        """Fill a trapezoidal road-prior mask."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        top_y  = int(img_h * self.TOP_Y_FRAC)
        half_top  = int(img_w * self.TOP_W_FRAC  / 2)
        half_bot  = int(img_w * self.BOTTOM_W_FRAC / 2)
        cx = img_w // 2
        pts = np.array([
            [cx - half_top, top_y],
            [cx + half_top, top_y],
            [cx + half_bot, img_h],
            [cx - half_bot, img_h],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    # COCO class IDs that are definitely NOT drivable road surface.
    # We erase their seg masks from the road prior to avoid counting
    # buildings, trees, sky, sidewalk-people etc. as road area.
    _NON_ROAD_COCO_IDS = {
        # Outdoor / structural
        0,   # person  (standing on road ≠ road itself)
        # vegetation / nature handled via colour heuristic below
        # (COCO has no "tree" or "sky" class in the detection set)
    }

    def _subtract_non_road(self, mask: np.ndarray, seg_results,
                           img_h: int, img_w: int) -> np.ndarray:
        """
        Erase seg-model masks for non-road classes from the road prior.
        Also erases the upper SKY BAND (top 30% of frame) which is never road.
        """
        # Always erase the sky band — nothing above the horizon is road
        sky_cut = int(img_h * 0.30)
        mask[:sky_cut, :] = 0

        if seg_results.masks is None:
            return mask

        cls_ids = seg_results.boxes.cls.cpu().numpy().astype(int)
        masks_data = seg_results.masks.data.cpu().numpy()  # (N, H, W) float32

        for cls_id, m in zip(cls_ids, masks_data):
            if cls_id in self._NON_ROAD_COCO_IDS:
                # Resize mask to image dimensions
                m_resized = cv2.resize(
                    m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                erase = (m_resized > 0.5).astype(np.uint8) * 255
                # Subtract: road_mask AND NOT erase
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(erase))

        return mask

    def _masks_from_seg(self, vehicle_mask: np.ndarray,
                        seg_results, img_h: int, img_w: int) -> np.ndarray:
        """Paint vehicle instance masks onto vehicle_mask."""
        cls_ids    = seg_results.boxes.cls.cpu().numpy().astype(int)
        masks_data = seg_results.masks.data.cpu().numpy()

        for cls_id, m in zip(cls_ids, masks_data):
            if cls_id in VEHICLE_IDS:
                m_resized = cv2.resize(
                    m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                vehicle_mask = cv2.bitwise_or(
                    vehicle_mask,
                    (m_resized > 0.5).astype(np.uint8) * 255
                )
        return vehicle_mask

    def _masks_from_bboxes(self, vehicle_mask: np.ndarray,
                           vehicle_dets: List,
                           img_h: int, img_w: int) -> np.ndarray:
        """
        Fallback: paint perspective-weighted bbox rectangles.
        Vehicles near the top of the frame (far away, small) are shrunk
        toward their bottom edge to avoid over-counting road coverage.
        """
        for d in vehicle_dets:
            x1, y1, x2, y2 = d.bbox
            # Normalised vertical centre: 0 = top, 1 = bottom
            v_centre = ((y1 + y2) / 2) / max(img_h, 1)
            # Perspective weight: far vehicles contribute less footprint
            weight = 0.25 + 0.75 * v_centre
            # Shrink height toward the bottom (ground contact line)
            new_h = max(int((y2 - y1) * weight), 1)
            new_y1 = y2 - new_h
            cv2.rectangle(vehicle_mask, (x1, new_y1), (x2, y2), 255, -1)
        return vehicle_mask


# ─────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────

class CrowdFlowDetector:

    def __init__(self, model_name: str = "yolov8n-seg.pt", device: str = "cpu"):
        self.model_name   = model_name
        self.device       = device
        self.model        = None
        self.is_seg_model = "-seg" in model_name   # True when masks available
        self._load_model()
        self._analyser    = EmergencyAnalyser()
        self._road_mask   = _RoadMaskEstimator()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            print(f"[CrowdFlow] Loading {self.model_name} …")
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            mode = "SEG" if self.is_seg_model else "DET"
            print(f"[CrowdFlow] Model ready [{mode}] on {self.device.upper()}")
            if not self.is_seg_model:
                print("[CrowdFlow] ℹ️  Detection-only model — using bbox fallback "
                      "for occupancy. For pixel-accurate occupancy, use "
                      "yolov8n-seg.pt or yolov8x-seg.pt")
        except Exception as e:
            print(f"[CrowdFlow] Could not load model: {e}")
            print("[CrowdFlow]    Run:  pip install ultralytics")

    # ── core detection ──────────────────────────────────────────────────────

    def detect(self, image_path: str, save_path: str = None,
               show: bool = False) -> AnalysisReport:

        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = frame.shape[:2]
        report = AnalysisReport(
            image_path   = str(img_path),
            image_size   = (w, h),
            inference_ms = 0,
            total_objects= 0,
        )

        if self.model is None:
            print("[CrowdFlow] No model loaded — returning empty report.")
            return report

        # ── run inference ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = self.model(
            frame,
            verbose=False,
            conf=0.22,
            iou=0.35,
            max_det=500,
            agnostic_nms=True,
            retina_masks=True,   # full-res masks when seg model is used
        )[0]
        report.inference_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── parse detections ─────────────────────────────────────────────────
        boxes  = results.boxes.xyxy.cpu().numpy()
        clsids = results.boxes.cls.cpu().numpy().astype(int)
        confs  = results.boxes.conf.cpu().numpy()
        names  = results.names

        for box, cls_id, conf in zip(boxes, clsids, confs):
            x1, y1, x2, y2 = map(int, box)
            det = self._classify(cls_id, names.get(cls_id, "unknown"),
                                 float(conf), [x1, y1, x2, y2], w, h)
            if det is None:
                continue
            report.detections.append(det)
            self._tally(report, det)

        # ── road mask + lane occupancy (NEW) ─────────────────────────────────
        seg_res = results if self.is_seg_model else None

        road_mask = self._road_mask.build_road_mask(
            img_h=h, img_w=w, seg_results=seg_res)

        vehicle_dets = [d for d in report.detections
                        if d.category in ("vehicle", "parking", "crashed")]

        report.lane_occupancy = self._road_mask.vehicle_occupancy(
            road_mask    = road_mask,
            vehicle_dets = vehicle_dets,
            seg_results  = seg_res,
            img_h        = h,
            img_w        = w,
        )

        # ── emergency analysis (BEFORE drawing so we mark crashed boxes) ─────
        report.emergency = self._analyser.analyse(report)

        # ── draw all boxes + road/occupancy overlays ─────────────────────────
        annotated = frame.copy()
        self._draw_road_mask(annotated, road_mask)   # semi-transparent road tint

        all_boxes = results.boxes.xyxy.cpu().numpy()
        all_cls   = results.boxes.cls.cpu().numpy().astype(int)
        all_conf  = results.boxes.conf.cpu().numpy()
        for box, cls_id, conf in zip(all_boxes, all_cls, all_conf):
            x1, y1, x2, y2 = map(int, box)
            det = self._classify(cls_id, names.get(cls_id, "unknown"),
                                 float(conf), [x1, y1, x2, y2], w, h)
            if det:
                self._draw_box(annotated, det, x1, y1, x2, y2)

        # ── post-processing ──────────────────────────────────────────────────
        report.total_objects    = (report.vehicles + report.people +
                                   report.road_blocks + report.illegal_parking)
        report.congestion_level = self._grade(report.congestion_score())
        self._draw_overlay(annotated, report)

        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"[CrowdFlow] Saved annotated image → {save_path}")

        if show:
            cv2.imshow("CrowdFlow AI Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return report

    # ── classification logic ─────────────────────────────────────────────────

    def _classify(self, cls_id: int, name: str, conf: float,
                  bbox: List[int], img_w: int, img_h: int):
        x1, y1, x2, y2 = bbox

        # 1. Vehicle
        if cls_id in VEHICLE_IDS and conf >= CONF_VEHICLE:
            is_parked = self._is_illegal_park(x1, y1, x2, y2, img_w, img_h)
            return Detection(
                category="vehicle" if not is_parked else "parking",
                label=VEHICLE_IDS[cls_id],
                confidence=conf, bbox=bbox,
                is_illegal_parking=is_parked
            )

        # 2. Person
        if cls_id == PERSON_ID and conf >= CONF_PERSON:
            return Detection("person", "person", conf, bbox)

        # 3. Known road control objects
        if cls_id in CONE_IDS and conf >= CONF_BLOCK:
            return Detection("roadblock", name.replace("_", " "), conf, bbox)

        # 4. Shape heuristic for cones / barriers
        bw, bh   = x2 - x1, y2 - y1
        aspect   = bw / max(bh, 1)
        area     = bw * bh
        centre_x = (x1 + x2) / 2 / img_w
        centre_y = (y1 + y2) / 2 / img_h
        if (0.25 < centre_x < 0.75 and centre_y > 0.3
                and area < (img_w * img_h * 0.01)
                and 0.3 < aspect < 3.0
                and conf >= CONF_BLOCK):
            return Detection("roadblock", "barrier/cone", conf, bbox)

        return None

    def _is_illegal_park(self, x1, y1, x2, y2, img_w, img_h) -> bool:
        bottom_y   = y2 / img_h
        left_edge  = x1 / img_w
        right_edge = x2 / img_w
        on_shoulder = bottom_y > PARKING_ZONE_RATIO
        on_edge     = (left_edge < PARKING_EDGE_RATIO or
                       right_edge > (1 - PARKING_EDGE_RATIO))
        return on_shoulder and on_edge

    # ── counting ─────────────────────────────────────────────────────────────

    def _tally(self, report: AnalysisReport, det: Detection):
        if det.category == "vehicle":
            report.vehicles += 1
            report.vehicle_types[det.label] = (
                report.vehicle_types.get(det.label, 0) + 1)
        elif det.category == "person":
            report.people += 1
        elif det.category == "roadblock":
            report.road_blocks += 1
        elif det.category == "parking":
            report.illegal_parking += 1
            report.vehicles += 1
            report.vehicle_types[det.label] = (
                report.vehicle_types.get(det.label, 0) + 1)

    # ── drawing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_road_mask(img: np.ndarray, road_mask: np.ndarray,
                        color: Tuple = (60, 200, 60), alpha: float = 0.18):
        """
        Tints the estimated road area with a faint green overlay so the user
        can see exactly which pixels are counted as 'drivable zone'.
        """
        tint = img.copy()
        tint[road_mask == 255] = (
            np.clip(
                tint[road_mask == 255].astype(np.int32) * (1 - alpha) +
                np.array(color, dtype=np.int32) * alpha,
                0, 255
            ).astype(np.uint8)
        )
        cv2.addWeighted(tint, 1.0, img, 0.0, 0, img)

    def _draw_box(self, img, det: Detection, x1, y1, x2, y2):
        if det.is_crashed:
            color     = COLORS["crashed"]
            thickness = 3
            tag = f"CRASH {det.label} {det.confidence:.0%}"
        elif det.is_illegal_parking:
            color     = COLORS["parking"]
            thickness = 3
            tag = f"ILLEGAL PARK {det.confidence:.0%}"
        else:
            color     = COLORS.get(det.category, (200, 200, 200))
            thickness = 2
            tag = f"{det.label} {det.confidence:.0%}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(tag, LABEL_FONT, 0.48, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, tag, (x1 + 2, y1 - 4),
                    LABEL_FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_overlay(self, img, report: AnalysisReport):
        """Top-left summary panel."""
        em        = report.emergency
        panel_h   = 290 if em else 200
        overlay   = img.copy()
        cv2.rectangle(overlay, (0, 0), (420, panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        em_color  = EMERGENCY_COLORS.get(em.level if em else "NONE", (180,180,180))
        scene_str = em.scene_type.upper() if em else "UNKNOWN"
        em_str    = em.level if em else "N/A"
        debris    = em.debris_count if em else 0
        em_veh    = em.emergency_vehicles if em else 0

        occ_pct   = round(report.lane_occupancy * 100, 1)
        # Occupancy bar colour: green → yellow → red
        if occ_pct < 40:
            occ_color = (50, 210, 50)
        elif occ_pct < 65:
            occ_color = (0, 200, 255)
        else:
            occ_color = (0, 80, 255)

        lines = [
            ("CrowdFlow AI",
             (255, 220, 60), 0.65),
            (f"Vehicles      : {report.vehicles:>4}",
             (255, 165,  0), 0.52),
            (f"People        : {report.people:>4}",
             ( 50, 200, 50), 0.52),
            (f"Road blocks   : {report.road_blocks:>4}",
             (100, 180, 255), 0.52),
            (f"Illegal park  : {report.illegal_parking:>4}",
             (  0,  80, 255), 0.52),
            (f"Lane occupancy: {occ_pct:>5.1f}%",          # NEW
             occ_color, 0.52),
            (f"Congestion    : {report.congestion_level} ({report.congestion_score()}/10)",
             (200, 200, 200), 0.48),
            (f"Scene type    : {scene_str}",
             em_color, 0.52),
            (f"EMERGENCY     : {em_str}",
             em_color, 0.62),
            (f"Debris objs   : {debris:>4}  |  Lg.veh: {em_veh}",
             (180, 180, 180), 0.42),
            (f"Inference     : {report.inference_ms} ms",
             (130, 130, 130), 0.40),
        ]

        # Bottom banner for HIGH / CRITICAL
        if em and em.level in ("HIGH", "CRITICAL"):
            banner_color = EMERGENCY_COLORS[em.level]
            cv2.rectangle(img,
                          (0, img.shape[0] - 44),
                          (img.shape[1], img.shape[0]),
                          banner_color, -1)
            banner_txt = (f"EMERGENCY {em.level}: {em.scene_type.upper()}"
                          f"  |  {em.reasons[0][:72]}")
            cv2.putText(img, banner_txt,
                        (10, img.shape[0] - 14),
                        LABEL_FONT, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        y = 24
        for text, color, scale in lines:
            cv2.putText(img, text, (10, y),
                        LABEL_FONT, scale, color, 1, cv2.LINE_AA)
            y += int(scale * 38 + 6)

    @staticmethod
    def _grade(score: float) -> str:
        # Thresholds recalibrated to match the new congestion_score() formula.
        # With vehicle ceiling at 50: a 15-car normal street ≈ 1.5 vehicle pts
        # → total score ~2-3 depending on parking/people → "Light" or "Moderate".
        # "Congested" now requires genuinely heavy traffic; "Critical" needs
        # gridlock-level counts or road blocks.
        if score < 2.5: return "Clear"
        if score < 4.5: return "Light"
        if score < 6.5: return "Moderate"
        if score < 8.5: return "Congested"
        return "Critical"


# ─────────────────────────────────────────────
# WEBCAM MODE
# ─────────────────────────────────────────────

def run_webcam(detector: CrowdFlowDetector):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CrowdFlow] No webcam found.")
        return
    print("[CrowdFlow] Webcam live — press Q to quit.")
    analyser   = EmergencyAnalyser()
    road_est   = _RoadMaskEstimator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        results = detector.model(frame, verbose=False, conf=0.22,
                                 iou=0.35, agnostic_nms=True,
                                 retina_masks=True)[0]
        ms = round((time.perf_counter() - t0) * 1000, 1)

        report = AnalysisReport("webcam", (w, h), ms, 0)
        boxes  = results.boxes.xyxy.cpu().numpy()
        clsids = results.boxes.cls.cpu().numpy().astype(int)
        confs  = results.boxes.conf.cpu().numpy()
        names  = results.names

        for box, cls_id, conf in zip(boxes, clsids, confs):
            x1, y1, x2, y2 = map(int, box)
            det = detector._classify(cls_id, names.get(cls_id, "?"),
                                     float(conf), [x1, y1, x2, y2], w, h)
            if det:
                report.detections.append(det)
                detector._tally(report, det)

        seg_res = results if detector.is_seg_model else None
        road_mask = road_est.build_road_mask(img_h=h, img_w=w, seg_results=seg_res)
        vehicle_dets = [d for d in report.detections
                        if d.category in ("vehicle", "parking", "crashed")]
        report.lane_occupancy = road_est.vehicle_occupancy(
            road_mask, vehicle_dets, seg_res, h, w)

        report.total_objects    = report.vehicles + report.people + report.road_blocks
        report.congestion_level = detector._grade(report.congestion_score())
        report.emergency        = analyser.analyse(report)

        detector._draw_road_mask(frame, road_mask)
        for box, cls_id, conf in zip(boxes, clsids, confs):
            x1, y1, x2, y2 = map(int, box)
            det = detector._classify(cls_id, names.get(cls_id, "?"),
                                     float(conf), [x1, y1, x2, y2], w, h)
            if det:
                detector._draw_box(frame, det, x1, y1, x2, y2)
        detector._draw_overlay(frame, report)

        cv2.imshow("CrowdFlow AI — Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CrowdFlow AI — Road Congestion & Emergency Detector"
    )
    parser.add_argument("--image",  type=str)
    parser.add_argument("--output", type=str, default="crowdflow_result.jpg")
    parser.add_argument("--model",  type=str, default="yolov8n-seg.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--json",   action="store_true")
    parser.add_argument("--show",   action="store_true")
    args = parser.parse_args()

    det = CrowdFlowDetector(model_name=args.model, device=args.device)

    if args.webcam:
        run_webcam(det)
        return

    if not args.image:
        parser.print_help()
        return

    report = det.detect(
        image_path = args.image,
        save_path  = args.output,
        show       = args.show,
    )

    em = report.emergency

    print("\n" + "═" * 56)
    print("  CrowdFlow AI — Detection Report")
    print("═" * 56)
    print(f"  Image       : {report.image_path}")
    print(f"  Size        : {report.image_size[0]}×{report.image_size[1]} px")
    print(f"  Inference   : {report.inference_ms} ms")
    print("─" * 56)
    print(f"   Vehicles         : {report.vehicles}")
    if report.vehicle_types:
        for vtype, cnt in sorted(report.vehicle_types.items(), key=lambda x: -x[1]):
            print(f"       ├─ {vtype:<16}: {cnt}")
    print(f"   People           : {report.people}")
    print(f"   Road blocks      : {report.road_blocks}")
    print(f"   Illegal parking  : {report.illegal_parking}")
    print("─" * 56)
    print(f"  Congestion score : {report.congestion_score()} / 10")
    print(f"  Congestion level : {report.congestion_level}")
    print(f"  Lane occupancy   : {round(report.lane_occupancy * 100, 1)}%")
    print("─" * 56)

    if em:
        level_icon = {
            "NONE":     "",
            "LOW":      "",
            "MODERATE": "",
            "HIGH":     "",
            "CRITICAL": "",
        }.get(em.level, "?")
        print(f"  {level_icon} EMERGENCY LEVEL    : {em.level}")
        print(f"   Scene type        : {em.scene_type.upper()}")
        print(f"   Crash pairs (IoU) : {em.crash_pairs}")
        print(f"   Crash proximity   : {em.crash_proximity}")
        print(f"   People near veh.  : {em.people_near_vehicles}")
        print(f"   Large veh. (proxy): {em.emergency_vehicles}")
        print(f"   Debris objects    : {em.debris_count}")
        print("   Reasons:")
        for r in em.reasons:
            print(f"       • {r}")

    print("═" * 56)
    print(f"  Annotated image → {args.output}\n")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()