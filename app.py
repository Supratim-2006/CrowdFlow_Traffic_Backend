"""
Traffic Disruption Prediction API
----------------------------------
Users supply ~20 raw fields; the API derives all 40+ model features internally.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (must match training code exactly)
# ─────────────────────────────────────────────────────────────────────────────
BANGALORE_LAT, BANGALORE_LON = 12.9716, 77.5946

BINS   = [0, 30, 90, 240, float("inf")]
LABELS = {0: "<30 mins (Quick)", 1: "30–90 mins (Minor)",
          2: "90–240 mins (Major)", 3: ">240 mins (Severe)"}

CATEGORICAL_COLS = [
    "event_type", "event_cause", "priority", "zone", "junction",
    "veh_type", "corridor", "direction", "police_station",
    "reason_breakdown", "cargo_material", "status",
]
NUMERICAL_COLS = [
    # Spatial
    "latitude", "longitude", "dist_from_centre",
    "has_end_coords", "end_lat_delta", "end_lon_delta", "incident_spread",
    # Temporal
    "is_weekend", "is_rush_hour", "is_night", "is_lunch_hour",
    "start_day_of_week", "start_month", "start_quarter", "start_week",
    "mins_from_midnight",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Text
    "desc_length", "comment_length", "desc_word_count", "comment_word_count",
    "has_comment",
    # Keyword flags
    "flag_heavy", "flag_blocked", "flag_bmtc", "flag_accident",
    "flag_infra", "flag_fire", "flag_tree", "keyword_severity_score",
    # Operational
    "triage_lag_mins", "requires_road_closure_bin",
    "zone_median_resolution", "junction_median_resolution", "cause_severity",
]

# Zone / junction historical medians learned from your training data.
# Replace these values with real medians computed during training for best accuracy.
ZONE_MEDIANS: dict[str, float] = {
    "Central Zone 1": 57.08,
    "Central Zone 2": 51.4,
    "East Zone 1": 56.02,
    "East Zone 2": 86.2,
    "North Zone 1": 80.65,
    "North Zone 2": 61.22,
    "South Zone 1": 113.63,
    "South Zone 2": 81.6,
    "West Zone 1": 53.74,
    "West Zone 2": 52.84
}

JUNCTION_MEDIANS: dict[str, float] = {
    "17th Mn 1st Crs Aishwarya Stores Jn": 55.78,
    "27th Cross Jayanagar(Ganapathi Temple)": 119.25,
    "28thMainJayanagarJunc": 133311.1,
    "29thMainRdBTM LayoutJunc": 35896.27,
    "5thMainHSR": 30846.67,
    "5thMainRPC Layout-Vijayanagar": 30.15,
    "A S CharStreet-MysoreRdJunc": 5203.56,
    "ASC Junction": 78.13,
    "AdugodiJunc": 16.72,
    "AgaraJunction": 315.37,
    "AnandRaoJunction": 44.15,
    "AnepalyaJunc": 16234.7,
    "AnilKumbleCircle": 76.63,
    "ArakereGateJunc": 74019.14,
    "Arbindo Circle": 52.62,
    "Arts&CraftsCircle": 9.2,
    "AshirwadamCircle": 40239.07,
    "Ashoknagar Junction(ShoolayCircle)": 53.18,
    "AttiguppeCircleJunction": 13.02,
    "AyyappaTempleJunc": 87.56,
    "BDA Junctio-Koramangala": 64.2,
    "BEL Circle": 75.57,
    "BEML GateJunc(SuranjandasRd)": 2.65,
    "BHEL Gate": 31.15,
    "BM ShriJunc(CMH-100FtRd)Junc": 51.23,
    "BMTCJunction-K H Road": 56.38,
    "BTM16thMain-ORR Junc": 33.11,
    "BagalakunteCrossJunction": 67342.95,
    "BagalurCrossJunc": 74.82,
    "Bamboobazar(Shivajinagar)": 118162.02,
    "BanashankariBusStandJunc": 23439.62,
    "BangaloreBodyBuildersJunc": 27.14,
    "Basappa Circle Junction": 60.69,
    "BasavamantapaJunc-Dr RajkumarRd": 5812.45,
    "BasaweshwaraCircle": 69.36,
    "Batrayanapura(Amrutahalli)Junction": 89.09,
    "BegumMahalJunc": 827.15,
    "Bellandur Junction, Outer ring road": 23.43,
    "BhadrappaLayout": 29961.39,
    "BhashyamCircle": 45.58,
    "BhashyamCircle-SadashivNagar": 26.47,
    "BigBazaar(Whitefield)Junc": 13014.92,
    "BilekahalliJunc": 426.61,
    "BinnyMillJunction": 5915.67,
    "BloodBankCircle": 19838.51,
    "Bommanahalli": 31.21,
    "BucheryJunction": 29293.75,
    "CID-CarltonHouseJunc": 23657.59,
    "CMP GateJunc": 5282.48,
    "CashPharmacyJunction": 50.36,
    "ChandrikaJunction": 44.99,
    "ChaudrayaCircle/UdayaTVCircle(CantonmentJunc)": 140.88,
    "Chokasandra (Tumkur road)": 2739.9,
    "CholurpalyaJunction(MagadiRd)": 37.65,
    "CoffeeBoardJunc": 37822.97,
    "CommandoHospitalJunc": 33.23,
    "D'SouzaCircle": 142.19,
    "DairyCircle": 47.46,
    "Delmia-Jayanagar": 102211.02,
    "DevangaHostelJunction": 14.66,
    "Devasandra(k r puram)": 61.51,
    "DevegowdaPetrolBunkJunc": 92.29,
    "Deverabeesanahalli-ORR Junc": 25.07,
    "DhobiGhatJunc": 61.66,
    "DoddaballapuraCrossJunc": 4305.02,
    "DomlurWaterTank": 64.64,
    "Dr RajkumarRd-10thCrossRdJunc": 119.56,
    "Dr TCM RoyanRd near AmbedkarStatue": 90.11,
    "ElectronicCityGate-2Junc": 52.79,
    "EliteJunc": 41.77,
    "FTI Junction(KanteeravaStudio)": 104.5,
    "Fire Force Junction": 18.63,
    "GangammagudiJunc": 37.56,
    "Geeta Circle": 70.25,
    "GokuldasImagesJunc": 34.59,
    "GopalGowdaJUnc": 64.04,
    "GorappanapalyaJunction": 11.78,
    "GoruguntepalyaJunc": 38.3,
    "GowdanapalyaJunction": 19.04,
    "HSR14thMainJunc": 60.44,
    "HainsJunc": 70.59,
    "HalliThindi": 98.25,
    "HebbalFlyoverJunc": 283.88,
    "HennurRoad-ORR Junc": 72.92,
    "HesaraghattaJunction": 30.43,
    "HoodiJunction": 67.44,
    "HopefarmJunction": 1343.12,
    "HoramavuJunction": 109.08,
    "HudsonCircle": 14.24,
    "HulimaveRd-BanneraghattaRdJunc": 51442.15,
    "HunsemarammanahalliJunction": 16.33,
    "ISRO Junction-Airport rd": 4300.17,
    "IbblurJunction'": 1971.34,
    "IndianExpressJunction": 77.91,
    "ItmaduJunction": 56.04,
    "J D MaraJunc": 20.98,
    "JP Nagar 15th cross junction": 25.9,
    "JP Nagar 9th cross-24th main jn": 114.13,
    "JaiMuniRaoCircle": 100.31,
    "Jaipuria-Adugodi": 26033.58,
    "JakkurCrossJunction": 25.25,
    "JalahalliCross(SM Circle)": 40.02,
    "JayadevaHospitalJunc": 46.77,
    "Jayanagar 4th main,36th cross": 152.83,
    "JohnsonMarket": 5570.1,
    "K H Road-SiddaiahRdJunc(BMTC-BigBazaar)": 53.86,
    "K R Circle": 61.14,
    "K R MarketJunction": 52.68,
    "KCG HospitalJunction": 27375.6,
    "KIMCO Junction": 65.45,
    "KR Road-14thCross Junc": 123.51,
    "KadirenahalliJunction": 106.08,
    "Kadubeesanahalli-ORR Junc": 78.02,
    "KamakyaJunction": 44.53,
    "Kammanahalli-RingRoadJunction": 55.73,
    "KamrajRdJunction": 108.49,
    "KarnatakaBhavanJunction": 35.87,
    "KatriguppeJunction": 58.63,
    "KhodaysCircle(DV UrsCircle)": 42.0,
    "KogilluCrossJunc": 38.39,
    "KonanakunteJunction(KanakapurarRd)": 34.98,
    "KoramangalaWaterTankJunc": 8134.35,
    "KrishnaFlourMill": 15.54,
    "Krupanidhi College": 88.09,
    "Kudlu Gate Junc": 84.77,
    "KundanahalliGateJunc": 34.21,
    "KuvempuCircle": 1673.71,
    "LRDE Junction": 40.51,
    "LalbaghMainGateJunc": 44.92,
    "LeprosyhospitalJunc": 48.67,
    "LinkRoadMalleswaramJunc": 30.87,
    "LowerAgaram(IndiaGarage)Junc": 21.93,
    "MC Circle": 25.03,
    "MICO Bande": 47.35,
    "MS RamaiahJunc(TollGate)": 47.91,
    "MahalaxmiLayoutEntranceJunc": 2499.52,
    "MaharajaJunction": 33.48,
    "Malleswaram18thCrossRd-SampigeRdJunc": 51.78,
    "ManipalCentreJunc": 38.22,
    "MaratahalliBridgeJunc": 99.97,
    "MarenahalliRd-18thMainRdJunc": 20.33,
    "MargosaRd-18thCrossJunc": 58.05,
    "Marigowda-TavarkereRdJunc": 80.83,
    "MayohallJunction": 40.27,
    "MeenakshiMallJunc": 51452.84,
    "MekhriCircle": 64.49,
    "MillCornerRd-SampigeRdJunc": 99.01,
    "Minerva Circle": 9.07,
    "MinskSquare(CTO Junction)": 245.92,
    "MintoJunction": 76.29,
    "ModiBridgeJunction": 56508.94,
    "ModiHospital": 81.25,
    "MotherTeressaCircle": 30.67,
    "MysoreRd--MadhuPetrolBunkJunction": 30.16,
    "MysoreRd-RingRdJunc(Nayandanahallii)": 84.16,
    "N R SquareJunc": 33.82,
    "NCERT Junction": 146.17,
    "NGV RearGateJunc": 82805.94,
    "NIMHANS Junction": 74.3,
    "NTTF JunctionPeenya": 81.32,
    "NagaTheaterJunc-Ulsoor": 2292.3,
    "NaganathapuraJunction": 54.73,
    "Nagarbhavi": 347.22,
    "Nagavara-ORR Junction": 3254.18,
    "NandiCross(RaniCross)": 80.8,
    "NavarangBarJunc-Dr RajkumarRd": 155558.62,
    "OldMadrasRd-BMTC DepotJunc": 19.61,
    "OldMadrasRd-DoubleRdJunc": 10.71,
    "OldMadrasRd-Indranagar100ftRdJunc": 26.9,
    "OldMadrasRd-NGEF Junc": 37.85,
    "OldPoliceStation-Ashoknagar": 64.1,
    "OperaHouseJunc": 445.13,
    "PES-DevegowdaCircle": 51.14,
    "PeenyaPoliceStation": 30.71,
    "PlatformRdJunction": 44.54,
    "PoliceCornerJunc": 31.25,
    "PoliceTimmaiahCircle(GPO)": 72.16,
    "PotteryCircle": 72.35,
    "PrasannaJunction": 44.39,
    "PriyadarshiniHotel,Jayamahal,RT Nagar": 4163.11,
    "QueensStatueCircle": 4697.87,
    "RRR(Okalipuram)Junction": 49.17,
    "RajeshwariJunc": 66.57,
    "RamaiahCircle-UlsoorPoliceStation": 49.78,
    "RamamurthyNagarJunction": 49.64,
    "Richmond circle jn": 48.62,
    "Ring road-Near Kengunte Junction": 101.18,
    "RingRoad-UllalJunction": 64.4,
    "RoyanCircle-Chamrajpete": 11.69,
    "SRS Peenya Junc": 18.98,
    "SadahalliGateJunc(AirportRd)": 116.48,
    "SadashivnagarJunc": 27.49,
    "SafinaPlazaJunc": 64.51,
    "SagarTheatreJunc": 40.43,
    "SandeepUnnikrishnan-Yelhanka": 36.75,
    "SantheCircle": 69.22,
    "SarjapurRd-St JohnsRdJunc": 61541.13,
    "SatteliteBusStandJunc": 32.07,
    "ShankarMuttCircle": 38.99,
    "ShantalaJunction": 81.24,
    "ShivajiTalkiesJunc": 47.98,
    "Shivajinagar(BRV)Junction": 66.87,
    "ShivanahalliJunctionWOC": 178.83,
    "ShivandaCircle": 45.69,
    "Shivashankara circle": 91.55,
    "SiddalingaiahCircle": 33.68,
    "SiddapuraJunction": 125.77,
    "SilkBoardJunc": 44.66,
    "SindhiColonyJunction": 35646.79,
    "SirsiCircle": 35.14,
    "SonyworldJunction": 41.87,
    "South end circle": 18.19,
    "Srigandakaval": 103.61,
    "Srinivagilu(Ejipura)Junc": 16.96,
    "StateBankofMysoreJunc": 50.23,
    "SubbannaJunction": 91777.02,
    "SubedarChatramRd near SheshadripuramPS": 48.96,
    "Sumanhalli": 43.2,
    "SwagathMainRd-EastEndRdJunc": 5314.92,
    "TC Palya-OM RoadJunc": 154.2,
    "TVS CrossJunction": 74.81,
    "TataInstituteCircle": 49.72,
    "TownhallJunction": 43.43,
    "Trilight Circle,Race course Road": 63.35,
    "TrinityCircle": 98.99,
    "TumkurRdMarappanapalyaJunc": 31.92,
    "TyagiHengalvarayaJunc(DickensonRd)": 55.22,
    "UCO Bank(Forum)": 117.2,
    "UlsoorGateJunc": 31.28,
    "UniversityJunc(Janabharti)": 18.85,
    "UrvashiJunction": 51.4,
    "UttarahalliJunction": 8324.55,
    "VeerannapalyaJunction(BEL,HO)": 5001.93,
    "VeerasandraGateJunction": 89.56,
    "VijayanagarBusStandJunction": 31.68,
    "WebbsCircle": 149.87,
    "WilsonGarden10thCrossJunc": 91.13,
    "WilsonGarden12thCrossJunc": 30.09,
    "WiproJunc-Koramangala": 215.32,
    "YediyurMeternityHospital": 53.01,
    "YelhankaBypass": 67.37,
    "YelhankaCircle": 50.72,
    "Yemalur cross junc": 218.98,
    "YeshwanthpuraCircle": 27.86,
    "toll gate mysore road": 75.18
}
GLOBAL_MEDIAN = 64.526588325

CAUSE_SEVERITY_MAP = {
    "vehicle_breakdown": 1, "tree_fall": 2, "others": 2,
    "accident": 3, "road_work": 3, "water_logging": 3,
    "unknown": 2,
}

KEYWORD_GROUPS = {
    "flag_heavy":    ["heavy", "truck", "container", "tipper", "bus", "lorry", "tanker"],
    "flag_blocked":  ["blocked", "jam", "blocking", "stopping", "closed", "obstruct"],
    "flag_accident": ["accident", "crash", "collision", "hit", "skid"],
    "flag_infra":    ["drain", "sewer", "pothole", "road work", "dig", "pipe", "cement"],
    "flag_fire":     ["fire", "smoke", "burn"],
    "flag_tree":     ["tree", "branch", "fell", "fallen"],
}


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMA  (what the user actually sends)
# ─────────────────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    # --- Required core fields ---
    start_datetime: str = Field(
        ..., example="2026-06-19 08:30:00+0530",
        description="When the incident started. Format: YYYY-MM-DD HH:MM:SS±HHMM"
    )
    latitude: float  = Field(..., ge=-90,  le=90)
    longitude: float = Field(..., ge=-180, le=180)

    # --- Optional but recommended ---
    created_date:    Optional[str]   = Field(None, example="2026-06-19 08:10:00+0530")
    endlatitude:     Optional[float] = Field(None, ge=-90,  le=90)
    endlongitude:    Optional[float] = Field(None, ge=-180, le=180)
    description:     Optional[str]   = Field(None, example="Heavy truck breakdown causing blockage")
    comment:         Optional[str]   = Field(None, example="Tow vehicle requested")
    requires_road_closure: Optional[bool] = Field(False)

    # --- Categorical context fields ---
    event_type:        Optional[str] = Field(None, example="Breakdown")
    event_cause:       Optional[str] = Field(None, example="vehicle_breakdown")
    priority:          Optional[str] = Field(None, example="High")
    zone:              Optional[str] = Field(None, example="East")
    junction:          Optional[str] = Field(None, example="Silk Board")
    veh_type:          Optional[str] = Field(None, example="Truck")
    corridor:          Optional[str] = Field(None, example="ORR")
    direction:         Optional[str] = Field(None, example="North")
    police_station:    Optional[str] = Field(None, example="HSR")
    reason_breakdown:  Optional[str] = Field(None, example="Engine Failure")
    cargo_material:    Optional[str] = Field(None, example="Container")
    status:            Optional[str] = Field(None, example="Open")

    @field_validator("start_datetime", "created_date", mode="before")
    @classmethod
    def _parse_dt(cls, v):
        if v is None:
            return v
        # Accept multiple common formats
        for fmt in (
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                datetime.strptime(v, fmt)
                return v
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse datetime '{v}'. "
            "Expected format: YYYY-MM-DD HH:MM:SS+HHMM"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors training exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _kw_flag(text: str, words: list[str]) -> int:
    return int(any(w in text for w in words))


def _parse_flexible(dt_str: str) -> pd.Timestamp:
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return pd.to_datetime(dt_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(dt_str, errors="coerce")


def build_feature_row(req: PredictionRequest) -> pd.DataFrame:
    """Convert the minimal user request into the full feature vector."""

    start = _parse_flexible(req.start_datetime)

    # ── Temporal ────────────────────────────────────────────────────────────
    hour = start.hour
    dow  = start.dayofweek
    month = start.month

    feats: dict = {
        "start_hour":        hour,
        "start_day_of_week": dow,
        "start_month":       month,
        "start_quarter":     start.quarter,
        "start_week":        start.isocalendar()[1],

        # Cyclical
        "hour_sin":  np.sin(hour  * (2 * np.pi / 24)),
        "hour_cos":  np.cos(hour  * (2 * np.pi / 24)),
        "dow_sin":   np.sin(dow   * (2 * np.pi / 7)),
        "dow_cos":   np.cos(dow   * (2 * np.pi / 7)),
        "month_sin": np.sin((month - 1) * (2 * np.pi / 12)),
        "month_cos": np.cos((month - 1) * (2 * np.pi / 12)),

        "is_weekend":    int(dow in (5, 6)),
        "is_rush_hour":  int((8 <= hour <= 11) or (17 <= hour <= 20)),
        "is_night":      int(hour < 6 or hour >= 22),
        "is_lunch_hour": int(12 <= hour <= 14),
        "mins_from_midnight": hour * 60 + start.minute,
    }

    # ── Spatial ─────────────────────────────────────────────────────────────
    lat, lon = req.latitude, req.longitude
    feats["latitude"]  = lat
    feats["longitude"] = lon
    feats["dist_from_centre"] = np.sqrt(
        (lat - BANGALORE_LAT) ** 2 + (lon - BANGALORE_LON) ** 2
    )

    end_lat = req.endlatitude  if req.endlatitude  is not None else lat
    end_lon = req.endlongitude if req.endlongitude is not None else lon
    feats["has_end_coords"] = int(
        req.endlatitude is not None and req.endlongitude is not None
    )
    feats["end_lat_delta"]   = abs(end_lat - lat)
    feats["end_lon_delta"]   = abs(end_lon - lon)
    feats["incident_spread"] = np.sqrt(
        feats["end_lat_delta"] ** 2 + feats["end_lon_delta"] ** 2
    )

    # ── Text / keyword ───────────────────────────────────────────────────────
    desc    = (req.description or "").lower()
    comment = (req.comment     or "").lower()

    feats["desc_length"]        = len(desc)
    feats["comment_length"]     = len(comment)
    feats["desc_word_count"]    = len(desc.split())
    feats["comment_word_count"] = len(comment.split())
    feats["has_comment"]        = int(len(comment) > 0)

    for flag_name, words in KEYWORD_GROUPS.items():
        feats[flag_name] = _kw_flag(desc, words)
    feats["flag_bmtc"] = int("bmtc" in desc)
    feats["keyword_severity_score"] = sum(
        feats[f] for f in ("flag_heavy", "flag_blocked", "flag_accident",
                           "flag_fire", "flag_tree", "flag_infra")
    )

    # ── Operational ──────────────────────────────────────────────────────────
    if req.created_date:
        created = _parse_flexible(req.created_date)
        lag = (start - created).total_seconds() / 60.0
        feats["triage_lag_mins"] = max(0.0, lag)
    else:
        feats["triage_lag_mins"] = 0.0

    feats["requires_road_closure_bin"] = int(req.requires_road_closure or False)

    # Historical medians (look up from training-time tables)
    zone = (req.zone or "Unknown").strip()
    junc = (req.junction or "Unknown").strip()
    feats["zone_median_resolution"]     = ZONE_MEDIANS.get(zone,     GLOBAL_MEDIAN)
    feats["junction_median_resolution"] = JUNCTION_MEDIANS.get(junc, GLOBAL_MEDIAN)

    cause_key = (req.event_cause or "unknown").lower().strip()
    feats["cause_severity"] = CAUSE_SEVERITY_MAP.get(cause_key, 2)

    # ── Categorical columns (fill Unknown for missing) ───────────────────────
    cat_values = {
        "event_type":       req.event_type       or "Unknown",
        "event_cause":      req.event_cause      or "Unknown",
        "priority":         req.priority         or "Unknown",
        "zone":             zone,
        "junction":         junc,
        "veh_type":         req.veh_type         or "Unknown",
        "corridor":         req.corridor         or "Unknown",
        "direction":        req.direction        or "Unknown",
        "police_station":   req.police_station   or "Unknown",
        "reason_breakdown": req.reason_breakdown or "Unknown",
        "cargo_material":   req.cargo_material   or "Unknown",
        "status":           req.status           or "Unknown",
    }
    feats.update(cat_values)

    # Build DataFrame in the column order the model expects
    row = pd.DataFrame([feats])[CATEGORICAL_COLS + NUMERICAL_COLS]
    for col in CATEGORICAL_COLS:
        row[col] = row[col].astype(str).replace(["nan", "None", ""], "Unknown")

    return row


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Disruption Prediction API",
    description=(
        "Predict how long a traffic disruption will last.\n\n"
        "Send ~20 raw incident fields and get back a category:\n"
        "- **0** – Quick      (<30 mins)\n"
        "- **1** – Minor      (30–90 mins)\n"
        "- **2** – Major      (90–240 mins)\n"
        "- **3** – Severe     (>240 mins)"
    ),
    version="1.0.0",
)

# Load model once at startup
try:
    model_file = hf_hub_download(
        repo_id="SupratimKukri/traffic-disruption-model",
        filename="traffic_disruption_model.pkl"
    )

    with open(model_file, "rb") as f:
        MODEL = pickle.load(f)

    print("Model loaded successfully")

except Exception as e:
    MODEL = None
    print(f"Failed to load model: {e}")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", tags=["Prediction"])
def predict(req: PredictionRequest):
    """
    Predict the disruption severity category for a traffic incident.

    Supply at minimum `start_datetime`, `latitude`, and `longitude`.
    All other fields are optional but improve accuracy.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Train the model first and place "
                   "'traffic_disruption_model.pkl' in the working directory.",
        )

    try:
        feature_row = build_feature_row(req)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature engineering error: {exc}")

    try:
        pred_class = int(MODEL.predict(feature_row)[0])
        proba      = MODEL.predict_proba(feature_row)[0].tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")

    class_proba = {LABELS[i]: round(p, 4) for i, p in enumerate(proba)}
    confidence  = round(max(proba) * 100, 1)

    return JSONResponse({
        "prediction": {
            "class_id":    pred_class,
            "label":       LABELS[pred_class],
            "confidence":  f"{confidence}%",
        },
        "class_probabilities": class_proba,
        "input_summary": {
            "start_datetime": req.start_datetime,
            "location":       {"lat": req.latitude, "lon": req.longitude},
            "event_cause":    req.event_cause,
            "zone":           req.zone,
        },
    })


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests: list[PredictionRequest]):
    """
    Predict for multiple incidents in one call (max 100).
    Returns a list of predictions in the same order.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max 100 records per batch.")

    results = []
    for i, req in enumerate(requests):
        try:
            feature_row = build_feature_row(req)
            pred_class  = int(MODEL.predict(feature_row)[0])
            proba       = MODEL.predict_proba(feature_row)[0].tolist()
            results.append({
                "index":      i,
                "class_id":   pred_class,
                "label":      LABELS[pred_class],
                "confidence": f"{round(max(proba) * 100, 1)}%",
                "class_probabilities": {LABELS[j]: round(p, 4) for j, p in enumerate(proba)},
            })
        except Exception as exc:
            results.append({"index": i, "error": str(exc)})

    return JSONResponse({"predictions": results, "count": len(results)})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860
    )
