"""
gps_generator.py
----------------
Generates synthetic GPS probe data that follows the REAL road geometry
of Connaught Place, Delhi's inner/middle/outer circle corridor.

Key fixes vs the broken version:
  1. Center is Connaught Place, Delhi (28.6315, 77.2167) — NOT Karnataka.
  2. Points are sampled ALONG road segments, not randomly over a bounding box.
  3. Each probe point carries a `speed` attribute; heatmap weight =
     (1 - speed/max_speed) so slow/congested vehicles appear BRIGHT.
  4. Three intersection anchors placed at real CP junction coordinates.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Real-world anchor coordinates — Connaught Place, New Delhi
# ---------------------------------------------------------------------------
# These are the three arterial intersections used in the simulation corridor.
# Verified against OSM / Google Maps.
INTERSECTIONS = {
    "INT_1_Barakhamba": (28.6328, 77.2195),   # Barakhamba Rd × KG Marg
    "INT_2_Connaught":  (28.6315, 77.2167),   # CP inner-circle centroid
    "INT_3_Patel":      (28.6287, 77.2140),   # Patel Chowk junction
}

# Road segments as (start_lat, start_lon, end_lat, end_lon)
# These trace the main arterial corridor through CP
CP_ROAD_SEGMENTS: List[Tuple[float, float, float, float]] = [
    # Outer Ring (north arc)
    (28.6355, 77.2130,  28.6355, 77.2200),
    (28.6355, 77.2200,  28.6335, 77.2225),
    # Kasturba Gandhi Marg (east approach)
    (28.6328, 77.2240,  28.6328, 77.2195),
    (28.6328, 77.2195,  28.6315, 77.2167),
    # Sansad Marg (south approach)
    (28.6270, 77.2155,  28.6287, 77.2140),
    (28.6287, 77.2140,  28.6315, 77.2167),
    # Baba Kharak Singh Marg (west)
    (28.6315, 77.2100,  28.6315, 77.2140),
    (28.6315, 77.2140,  28.6315, 77.2167),
    # Inner circle loop
    (28.6315, 77.2167,  28.6328, 77.2195),
    (28.6328, 77.2195,  28.6335, 77.2225),
    (28.6335, 77.2225,  28.6320, 77.2240),
    (28.6320, 77.2240,  28.6300, 77.2225),
    (28.6300, 77.2225,  28.6287, 77.2200),
    (28.6287, 77.2200,  28.6287, 77.2140),
    # Janpath
    (28.6350, 77.2167,  28.6315, 77.2167),
    # Tolstoy Marg
    (28.6315, 77.2167,  28.6290, 77.2167),
]

MAX_SPEED_KMPH = 50.0
RNG = np.random.default_rng(42)


def _sample_along_segment(
    lat1: float, lon1: float, lat2: float, lon2: float,
    n: int, congestion_factor: float
) -> pd.DataFrame:
    """
    Place `n` probe points along a road segment.
    congestion_factor in [0,1]: 1 = fully jammed, 0 = free-flow.
    """
    t = RNG.uniform(0, 1, n)
    lats = lat1 + t * (lat2 - lat1) + RNG.normal(0, 0.00003, n)
    lons = lon1 + t * (lon2 - lon1) + RNG.normal(0, 0.00003, n)

    # Speed: slow when congested
    base_speed = MAX_SPEED_KMPH * (1.0 - congestion_factor)
    speeds = np.clip(
        RNG.normal(base_speed, base_speed * 0.25, n), 1, MAX_SPEED_KMPH
    )

    # Heatmap weight = inverse of normalised speed (congested = bright)
    weights = 1.0 - (speeds / MAX_SPEED_KMPH)
    weights = np.clip(weights, 0.05, 1.0)

    return pd.DataFrame({"lat": lats, "lon": lons,
                         "speed_kmph": speeds, "weight": weights})


def generate_gps_probes(
    n_vehicles: int = 600,
    congestion_level: float = 0.65,   # 0=free, 1=gridlock
    accident_at_int2: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Public API — returns a DataFrame of synthetic GPS probe points
    that follow real CP road geometry.

    Parameters
    ----------
    n_vehicles       : total number of probe records
    congestion_level : baseline congestion (0–1)
    accident_at_int2 : if True, create a hotspot near INT_2 (gridlock)
    seed             : optional RNG seed for reproducibility
    """
    if seed is not None:
        global RNG
        RNG = np.random.default_rng(seed)

    frames = []
    n_segs = len(CP_ROAD_SEGMENTS)
    per_seg = max(1, n_vehicles // n_segs)

    for seg in CP_ROAD_SEGMENTS:
        lat1, lon1, lat2, lon2 = seg
        # Vary congestion slightly per segment
        cf = np.clip(congestion_level + RNG.uniform(-0.15, 0.15), 0, 1)

        # Extra congestion near accident intersection
        if accident_at_int2:
            int2_lat, int2_lon = INTERSECTIONS["INT_2_Connaught"]
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            dist = np.sqrt((mid_lat - int2_lat)**2 + (mid_lon - int2_lon)**2)
            if dist < 0.003:       # within ~300 m
                cf = min(cf + 0.4, 1.0)

        frames.append(_sample_along_segment(lat1, lon1, lat2, lon2, per_seg, cf))

    df = pd.concat(frames, ignore_index=True)
    return df


def get_intersection_coords() -> dict:
    """Return the three intersection lat/lon anchors."""
    return INTERSECTIONS.copy()


def select_probe_vehicles(vehicle_ids: List[str], probe_ratio: float = 0.15) -> List[str]:
    """
    Select a subset of vehicles as GPS probes.
    
    Parameters
    ----------
    vehicle_ids  : list of all vehicle IDs in simulation
    probe_ratio  : fraction of vehicles to select as probes (default 15%)
    
    Returns
    -------
    list of selected probe vehicle IDs
    """
    n_probes = max(1, int(len(vehicle_ids) * probe_ratio))
    selected = RNG.choice(vehicle_ids, size=min(n_probes, len(vehicle_ids)), replace=False)
    return list(selected)


def collect_gps_frame(step: int, probe_vids: List[str], 
                      congestion_level: float = 0.5) -> List[dict]:
    """
    Collect synthetic GPS data for probe vehicles at a given simulation step.
    
    Parameters
    ----------
    step                : simulation step number (s)
    probe_vids          : list of probe vehicle IDs
    congestion_level    : current congestion level (0–1)
    
    Returns
    -------
    list of dicts with keys: {vehicle_id, step, lat, lon, speed_kmph, weight}
    """
    records = []
    for vid in probe_vids:
        # Generate synthetic GPS point along one of the CP road segments
        seg = CP_ROAD_SEGMENTS[step % len(CP_ROAD_SEGMENTS)]
        lat1, lon1, lat2, lon2 = seg
        
        t = RNG.uniform(0, 1)
        lat = lat1 + t * (lat2 - lat1) + RNG.normal(0, 0.00003)
        lon = lon1 + t * (lon2 - lon1) + RNG.normal(0, 0.00003)
        
        # Speed varies with congestion
        base_speed = MAX_SPEED_KMPH * (1.0 - congestion_level)
        speed = np.clip(RNG.normal(base_speed, base_speed * 0.25), 1, MAX_SPEED_KMPH)
        
        # Weight for heatmap: inverse of normalized speed
        weight = 1.0 - (speed / MAX_SPEED_KMPH)
        weight = np.clip(weight, 0.05, 1.0)
        
        records.append({
            "vehicle_id": vid,
            "step":       step,
            "lat":        lat,
            "lon":        lon,
            "speed_kmph": speed,
            "weight":     weight,
        })
    
    return records


def build_dataframe(gps_records: List[dict]) -> pd.DataFrame:
    """
    Convert list of GPS records to a DataFrame.
    
    Parameters
    ----------
    gps_records : list of dicts with GPS probe data
    
    Returns
    -------
    pandas DataFrame with columns: vehicle_id, step, lat, lon, speed_kmph, weight
    """
    if not gps_records:
        return pd.DataFrame(columns=[
            "vehicle_id", "step", "lat", "lon", "speed_kmph", "weight"
        ])
    
    return pd.DataFrame(gps_records)
