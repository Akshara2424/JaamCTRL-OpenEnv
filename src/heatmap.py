"""
src/heatmap.py
--------------
Folium heatmap builder for Jaam Ctrl.

Exports required by app.py:
  heatmap_to_html(gps_df, title, zoom)          -> str (HTML)
  combined_heatmap_to_html(mode_dfs, zoom)      -> str (HTML)
  per_junction_density(gps_df)                  -> dict[str, float]
  flow_balance_score(gps_df)                    -> float
  delay_reduction_pct(gps_df_a, gps_df_b)      -> float
  JUNCTION_NAMES                                -> dict[str, str]

GPS DataFrame schema (produced by run_simulation.py):
  lat       float   WGS-84 latitude
  lon       float   WGS-84 longitude
  speed_kmph float  vehicle speed
  weight    float   1 - speed/max_speed  (high = congested)
  junction  str     nearest junction id ("J0" / "J1" / "J2")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap

# ---------------------------------------------------------------------------
# Real-world anchor coordinates — Connaught Place, New Delhi
# Verified against OpenStreetMap / Google Maps
# ---------------------------------------------------------------------------
JUNCTION_COORDS: dict[str, tuple[float, float]] = {
    "J0": (28.6315, 77.2167),   # Tolstoy Marg / Janpath core
    "J1": (28.6328, 77.2195),   # Barakhamba Rd × KG Marg
    "J2": (28.6287, 77.2140),   # Patel Chowk
}

JUNCTION_NAMES: dict[str, str] = {
    "J0": "Tolstoy Marg",
    "J1": "CC Inner Ring (KG Marg)",
    "J2": "Patel Chowk",
}

# Corridor centre for map initialisation
_CP_CENTER = [28.6310, 77.2168]

# Max speed used for weight normalisation
_MAX_SPEED = 50.0

# Radius of influence for density assignment (degrees, ≈ 300 m)
_JUNCTION_RADIUS = 0.003

# ---------------------------------------------------------------------------
# Colour gradients per mode
# ---------------------------------------------------------------------------
_GRADIENTS = {
    "fixed": {
        0.2: "#1a0033",
        0.5: "#7C4DFF",
        0.8: "#FF2FD6",
        1.0: "#FF4444",
    },
    "adaptive": {
        0.2: "#001a33",
        0.5: "#0077FF",
        0.8: "#00E5FF",
        1.0: "#00F5D4",
    },
    "rl": {
        0.2: "#1a0033",
        0.5: "#FF2FD6",
        0.8: "#FF9900",
        1.0: "#FFFF00",
    },
    "default": {
        0.2: "#001a33",
        0.5: "#7C4DFF",
        0.8: "#00E5FF",
        1.0: "#FF2FD6",
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _base_map(zoom: int = 15) -> folium.Map:
    """Return a dark CartoDB map centred on Connaught Place."""
    return folium.Map(
        location=_CP_CENTER,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )


def _add_junction_markers(m: folium.Map) -> None:
    """Draw labelled circle markers at each junction's real coordinates."""
    for jid, (lat, lon) in JUNCTION_COORDS.items():
        name = JUNCTION_NAMES[jid]
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color="#00E5FF",
            fill=True,
            fill_color="#7C4DFF",
            fill_opacity=0.85,
            weight=2,
            tooltip=folium.Tooltip(f"{jid} | {name}", sticky=True),
            popup=folium.Popup(
                f"<b style='color:#00E5FF'>{jid}</b><br>{name}",
                max_width=200,
            ),
        ).add_to(m)


def _heat_layer(
    gps_df: pd.DataFrame,
    name: str,
    gradient: dict,
) -> HeatMap:
    """Build a HeatMap layer from a GPS DataFrame."""
    if gps_df.empty:
        data = []
    else:
        data = gps_df[["lat", "lon", "weight"]].dropna().values.tolist()
    return HeatMap(
        data,
        name=name,
        min_opacity=0.35,
        max_zoom=18,
        radius=18,
        blur=22,
        gradient=gradient,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def heatmap_to_html(
    gps_df: pd.DataFrame,
    title: str = "Traffic Heatmap",
    zoom: int = 15,
    mode: str | None = None,
) -> str:
    """
    Build a single-mode heatmap and return it as an HTML string.

    Parameters
    ----------
    gps_df : DataFrame with columns lat, lon, weight (and optionally speed_kmph)
    title  : shown in the layer control
    zoom   : initial map zoom level
    mode   : gradient mode ("fixed", "adaptive", "rl") — auto-inferred from title if None
    """
    # Auto-detect mode from title if not provided
    if mode is None:
        title_lower = title.lower()
        if "fixed" in title_lower:
            mode = "fixed"
        elif "adaptive" in title_lower:
            mode = "adaptive"
        elif "rl" in title_lower or "ppo" in title_lower:
            mode = "rl"
        else:
            mode = "default"
    
    m = _base_map(zoom)
    gradient = _GRADIENTS.get(mode, _GRADIENTS["default"])
    _heat_layer(gps_df, title, gradient).add_to(m)
    _add_junction_markers(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m._repr_html_()


def combined_heatmap_to_html(
    mode_dfs: dict[str, pd.DataFrame],
    zoom: int = 15,
) -> str:
    """
    Build a multi-layer heatmap (one layer per simulation mode).

    Parameters
    ----------
    mode_dfs : {"fixed": df, "adaptive": df, "rl": df}  (any subset is fine)
    zoom     : initial map zoom level
    """
    m = _base_map(zoom)

    layer_labels = {
        "fixed":    "Fixed-Time (baseline)",
        "adaptive": "Rule-Based Adaptive",
        "rl":       "PPO RL Agent",
    }

    for mode_key, gps_df in mode_dfs.items():
        if gps_df is None or gps_df.empty:
            continue
        gradient = _GRADIENTS.get(mode_key, _GRADIENTS["default"])
        label    = layer_labels.get(mode_key, mode_key.capitalize())
        _heat_layer(gps_df, label, gradient).add_to(m)

    _add_junction_markers(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m._repr_html_()


def per_junction_density(gps_df: pd.DataFrame) -> dict[str, float]:
    """
    Return average congestion weight for points near each junction.

    Returns
    -------
    {"J0": float, "J1": float, "J2": float}
    A value of 0.0 means no probe points near that junction.
    """
    result: dict[str, float] = {jid: 0.0 for jid in JUNCTION_COORDS}
    if gps_df.empty:
        return result

    # If the DataFrame already has a junction column use it directly
    if "junction" in gps_df.columns:
        for jid in JUNCTION_COORDS:
            sub = gps_df[gps_df["junction"] == jid]
            if not sub.empty:
                result[jid] = float(sub["weight"].mean())
        return result

    # Otherwise assign by proximity
    lats = gps_df["lat"].values
    lons = gps_df["lon"].values
    wts  = gps_df["weight"].values

    for jid, (jlat, jlon) in JUNCTION_COORDS.items():
        dist = np.sqrt((lats - jlat) ** 2 + (lons - jlon) ** 2)
        mask = dist < _JUNCTION_RADIUS
        if mask.any():
            result[jid] = float(wts[mask].mean())

    return result


def flow_balance_score(gps_df: pd.DataFrame) -> float:
    """
    Measure how evenly congestion is distributed across junctions.

    Returns std(density) / (mean(density) + 1e-6).
    Lower = more even = better.
    """
    dens = per_junction_density(gps_df)
    vals = np.array(list(dens.values()), dtype=float)
    if vals.mean() < 1e-9:
        return 0.0
    return float(vals.std() / (vals.mean() + 1e-6))


def delay_reduction_pct(
    gps_df_a: pd.DataFrame,
    gps_df_b: pd.DataFrame,
) -> float:
    """
    Estimate congestion reduction from mode A to mode B.

    Uses mean(weight) as a proxy for delay.
    Returns percentage reduction (positive = B is better).
    """
    if gps_df_a.empty or gps_df_b.empty:
        return 0.0
    w_a = gps_df_a["weight"].mean()
    w_b = gps_df_b["weight"].mean()
    if w_a < 1e-9:
        return 0.0
    return float((w_a - w_b) / w_a * 100.0)
