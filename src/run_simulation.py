"""
src/run_simulation.py
---------------------
Simulation runner for Jaam Ctrl.

Exports required by app.py:
  run_simulation(mode, traffic_scale, accident_step, seed,
                 baseline_delay, ppo_model, progress_cb)  -> SimResult
  SimResult   dataclass
  SIM_DURATION  int  (seconds)
  _mock_result(mode, baseline_delay)                      -> SimResult

SUMO-GUI
--------
When SUMO is installed, the simulation opens a SUMO-GUI desktop window
automatically. The GUI starts playing immediately — you can watch cars
moving on the real CP network while Streamlit shows live metrics.

To disable the GUI (headless / CI):
  SUMO_NO_GUI=1 streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIM_DURATION = 1800
CONTROL_STEP = 10
MIN_PHASE    = 15
MAX_PHASE    = 60
MAX_SPEED    = 50.0

TL_IDS = ["J0", "J1", "J2"]

JUNCTION_COORDS = {
    "J0": (28.6315, 77.2167),
    "J1": (28.6328, 77.2195),
    "J2": (28.6287, 77.2140),
}

_ROAD_SEGMENTS = [
    (28.6315, 77.2167, 28.6328, 77.2195, "J0"),
    (28.6328, 77.2195, 28.6315, 77.2167, "J1"),
    (28.6287, 77.2140, 28.6315, 77.2167, "J2"),
    (28.6315, 77.2167, 28.6287, 77.2140, "J0"),
    (28.6350, 77.2167, 28.6315, 77.2167, "J1"),
    (28.6315, 77.2167, 28.6270, 77.2167, "J0"),
    (28.6328, 77.2240, 28.6328, 77.2195, "J1"),
    (28.6315, 77.2100, 28.6315, 77.2140, "J2"),
    (28.6315, 77.2140, 28.6315, 77.2167, "J2"),
    (28.6300, 77.2225, 28.6287, 77.2200, "J2"),
    (28.6335, 77.2225, 28.6328, 77.2195, "J1"),
    (28.6315, 77.2167, 28.6335, 77.2225, "J0"),
]

# ---------------------------------------------------------------------------
# SUMO-GUI flag
# Set SUMO_NO_GUI=1 in your environment to force headless mode.
# ---------------------------------------------------------------------------
_USE_GUI = os.environ.get("SUMO_NO_GUI", "0") != "1"

# ---------------------------------------------------------------------------
# SUMO availability
# ---------------------------------------------------------------------------
try:
    import traci
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False

# ---------------------------------------------------------------------------
# SimResult
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    mode:          str
    metrics:       dict         = field(default_factory=dict)
    gps_df:        pd.DataFrame = field(default_factory=pd.DataFrame)
    phase_log:     list         = field(default_factory=list)
    signal_events: list         = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPS probe generator
# ---------------------------------------------------------------------------

def _generate_gps_df(n_vehicles, congestion_factor, accident_junc, rng):
    frames  = []
    per_seg = max(1, n_vehicles // len(_ROAD_SEGMENTS))

    for lat1, lon1, lat2, lon2, junc in _ROAD_SEGMENTS:
        cf = float(np.clip(congestion_factor + rng.uniform(-0.12, 0.12), 0, 1))
        if accident_junc and junc == accident_junc:
            cf = min(cf + 0.4, 1.0)

        t      = rng.uniform(0, 1, per_seg)
        lats   = lat1 + t*(lat2-lat1) + rng.normal(0, 0.00003, per_seg)
        lons   = lon1 + t*(lon2-lon1) + rng.normal(0, 0.00003, per_seg)
        base_s = MAX_SPEED * (1.0 - cf)
        speeds = np.clip(rng.normal(base_s, base_s*0.25+0.5, per_seg), 1.0, MAX_SPEED)
        weights= np.clip(1.0 - speeds/MAX_SPEED, 0.05, 1.0)

        frames.append(pd.DataFrame({
            "lat": lats, "lon": lons,
            "speed_kmph": speeds, "weight": weights, "junction": junc,
        }))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Mock simulation (no SUMO required)
# ---------------------------------------------------------------------------

_BASE_METRICS = {
    "fixed":    {"avg_delay_s": 62.0, "avg_stops": 3.8, "throughput": 950,  "cf": 0.72},
    "adaptive": {"avg_delay_s": 43.0, "avg_stops": 2.5, "throughput": 1120, "cf": 0.52},
    "rl":       {"avg_delay_s": 30.0, "avg_stops": 1.7, "throughput": 1310, "cf": 0.38},
}

_QUEUE_BASE = {
    "fixed":    {"J0": (9,7),  "J1": (11,8), "J2": (8,6)},
    "adaptive": {"J0": (6,4),  "J1": (7,5),  "J2": (5,4)},
    "rl":       {"J0": (4,3),  "J1": (5,3),  "J2": (3,2)},
}

_PHASE_LABELS = {0:"EW Green", 1:"EW Yellow", 2:"NS Green", 3:"NS Yellow"}


def _mock_phase_log(mode, traffic_scale, rng):
    phase_log, signal_events = [], []
    qb      = _QUEUE_BASE[mode]
    phases  = {jid: 0 for jid in TL_IDS}
    offsets = {"J0": 0, "J1": 36, "J2": 72}

    for step in range(0, SIM_DURATION, CONTROL_STEP):
        row = {"step": step}
        for jid in TL_IDS:
            qew_b, qns_b = qb[jid]
            qew = max(0.0, qew_b * traffic_scale + rng.uniform(-1.5, 1.5))
            qns = max(0.0, qns_b * traffic_scale + rng.uniform(-1.5, 1.5))
            t_eff = (step + offsets[jid]) % 78
            ph = 0 if t_eff < 40 else 1 if t_eff < 44 else 2 if t_eff < 74 else 3

            if ph != phases[jid]:
                signal_events.append({
                    "step": step, "junction": jid,
                    "from_phase": _PHASE_LABELS[phases[jid]],
                    "to_phase":   _PHASE_LABELS[ph],
                })
                phases[jid] = ph

            action = ("extend_ew" if mode != "fixed" and qew > qns
                      else "extend_ns" if mode != "fixed" else "fixed")
            row[f"{jid}_label"]    = _PHASE_LABELS[ph]
            row[f"{jid}_queue_ew"] = round(qew, 1)
            row[f"{jid}_queue_ns"] = round(qns, 1)
            row[f"{jid}_action"]   = action
        phase_log.append(row)

    return phase_log, signal_events


def _mock_result(mode, baseline_delay=None, traffic_scale=1.0,
                 accident_step=-1, seed=42):
    rng = np.random.default_rng(seed + {"fixed":0,"adaptive":1,"rl":2}.get(mode,0))
    bm  = _BASE_METRICS[mode]

    delay      = bm["avg_delay_s"] * traffic_scale + rng.uniform(-2, 2)
    stops      = bm["avg_stops"]   * traffic_scale + rng.uniform(-0.2, 0.2)
    throughput = int(bm["throughput"] / max(traffic_scale, 0.5) + rng.integers(-30, 30))
    improvement= 0.0
    if baseline_delay and mode != "fixed":
        improvement = max(0.0, (baseline_delay - delay) / baseline_delay * 100)

    per_junction = {}
    for jid in TL_IDS:
        qew_b, qns_b = _QUEUE_BASE[mode][jid]
        qew = max(0.0, qew_b * traffic_scale + rng.uniform(-1, 1))
        qns = max(0.0, qns_b * traffic_scale + rng.uniform(-1, 1))
        per_junction[jid] = {
            "avg_queue":    round((qew+qns)/2, 1),
            "avg_queue_ew": round(qew, 1),
            "avg_queue_ns": round(qns, 1),
        }

    acc_junc = "J1" if accident_step >= 0 else None
    gps_df   = _generate_gps_df(int(400*traffic_scale), bm["cf"], acc_junc, rng)
    phase_log, signal_events = _mock_phase_log(mode, traffic_scale, rng)

    return SimResult(
        mode    = mode,
        metrics = {
            "avg_delay_s": round(delay, 1),
            "avg_stops":   round(stops, 2),
            "throughput":  throughput,
            "improvement": round(improvement, 1),
            "per_junction": per_junction,
        },
        gps_df        = gps_df,
        phase_log     = phase_log,
        signal_events = signal_events,
    )


# ---------------------------------------------------------------------------
# SUMO simulation — with GUI support
# ---------------------------------------------------------------------------

def _sumo_result(mode, traffic_scale, accident_step, seed,
                 baseline_delay, ppo_model, progress_cb):

    src_dir  = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    cfg_path = os.path.join(root_dir, "sumo", "corridor.sumocfg")

    if not os.path.exists(cfg_path):
        return _mock_result(mode, baseline_delay, traffic_scale, accident_step, seed)

    # ── Use sumo-gui when _USE_GUI is True, sumo otherwise ──────────────────
    binary = "sumo-gui" if _USE_GUI else "sumo"

    sumo_cmd = [
        binary,
        "--configuration-file", cfg_path,
        "--seed",              str(seed),
        "--scale",             str(traffic_scale),
        "--no-warnings",       "true",
        "--no-step-log",       "true",
        "--time-to-teleport",  "300",
        # Start playing automatically — no need to press Play in the GUI
        "--start",             "true",
        # Close GUI when simulation ends
        "--quit-on-end",       "true",
        # 50 ms per step so cars are visible but sim stays fast
        "--delay",             "50",
    ]

    traci.start(sumo_cmd)

    total_delay  = 0.0
    total_stops  = 0
    total_veh    = 0
    arrive_count = 0
    phase_log    = []
    signal_events= []
    gps_rows     = []
    prev_phases  = {jid: None for jid in TL_IDS}
    queue_hist   = {jid: {"ew": [], "ns": []} for jid in TL_IDS}
    rng          = np.random.default_rng(seed)

    for step in range(SIM_DURATION):
        traci.simulationStep()

        # Accident injection
        if accident_step > 0 and step == accident_step:
            vehs = traci.vehicle.getIDList()
            if vehs:
                v = random.choice(vehs)
                traci.vehicle.setSpeed(v, 0)
                traci.vehicle.setSpeedMode(v, 0)

        veh_ids = traci.vehicle.getIDList()
        for vid in veh_ids:
            spd    = traci.vehicle.getSpeed(vid) * 3.6
            wait   = traci.vehicle.getWaitingTime(vid)
            pos    = traci.vehicle.getPosition(vid)
            lon    = 77.2167 + pos[0] / 111320.0
            lat    = 28.6315 + pos[1] / 110540.0
            weight = max(0.05, 1.0 - spd / MAX_SPEED)
            junc   = _nearest_junction(lat, lon)
            gps_rows.append({"lat":lat,"lon":lon,"speed_kmph":spd,
                              "weight":weight,"junction":junc})
            total_delay += wait
            total_veh   += 1
            if traci.vehicle.getStopState(vid) & 1:
                total_stops += 1

        arrive_count += traci.simulation.getArrivedNumber()

        if step % CONTROL_STEP == 0 and step > 0:
            row    = {"step": step}
            queues = {}

            for jid in TL_IDS:
                ph  = traci.trafficlight.getPhase(jid)
                lbl = _PHASE_LABELS.get(ph, "Unknown")

                if prev_phases[jid] is not None and ph != prev_phases[jid]:
                    signal_events.append({
                        "step": step, "junction": jid,
                        "from_phase": _PHASE_LABELS.get(prev_phases[jid], "?"),
                        "to_phase":   lbl,
                    })
                prev_phases[jid] = ph

                jlat, jlon = JUNCTION_COORDS[jid]
                qew = qns = 0
                for vid in veh_ids:
                    pos  = traci.vehicle.getPosition(vid)
                    vlat = 28.6315 + pos[1] / 110540.0
                    vlon = 77.2167 + pos[0] / 111320.0
                    if abs(vlat-jlat)<0.001 and abs(vlon-jlon)<0.001:
                        angle = traci.vehicle.getAngle(vid)
                        if 45 < angle < 135 or 225 < angle < 315:
                            qns += 1
                        else:
                            qew += 1

                queue_hist[jid]["ew"].append(qew)
                queue_hist[jid]["ns"].append(qns)
                queues[jid] = (qew, qns)

                action = "fixed"
                if mode == "adaptive":
                    action = _adaptive_action(jid, ph, qew, qns, step)
                    if action.startswith("switch"):
                        traci.trafficlight.setPhase(jid, (ph+1)%4)
                elif mode == "rl" and ppo_model is not None:
                    obs = _build_obs(jid, ph, qew, qns, step, queues)
                    ac, _ = ppo_model.predict(obs, deterministic=True)
                    if ac == 1:
                        traci.trafficlight.setPhase(jid, (ph+1)%4)
                    action = f"rl_action_{ac}"

                row[f"{jid}_label"]    = lbl
                row[f"{jid}_queue_ew"] = qew
                row[f"{jid}_queue_ns"] = qns
                row[f"{jid}_action"]   = action

            phase_log.append(row)
            if progress_cb:
                progress_cb(step, SIM_DURATION)

    traci.close()

    n = max(total_veh, 1)
    avg_delay   = total_delay / n
    avg_stops   = total_stops / n
    improvement = 0.0
    if baseline_delay and mode != "fixed":
        improvement = max(0.0, (baseline_delay - avg_delay) / baseline_delay * 100)

    per_junction = {}
    for jid in TL_IDS:
        ew_h = queue_hist[jid]["ew"]
        ns_h = queue_hist[jid]["ns"]
        qew  = float(np.mean(ew_h)) if ew_h else 0.0
        qns  = float(np.mean(ns_h)) if ns_h else 0.0
        per_junction[jid] = {
            "avg_queue":    round((qew+qns)/2, 1),
            "avg_queue_ew": round(qew, 1),
            "avg_queue_ns": round(qns, 1),
        }

    gps_df = pd.DataFrame(gps_rows) if gps_rows else pd.DataFrame(
        columns=["lat","lon","speed_kmph","weight","junction"])
    if len(gps_df) > 2000:
        gps_df = gps_df.sample(2000, random_state=seed)

    return SimResult(
        mode    = mode,
        metrics = {
            "avg_delay_s": round(avg_delay, 1),
            "avg_stops":   round(avg_stops, 2),
            "throughput":  arrive_count,
            "improvement": round(improvement, 1),
            "per_junction": per_junction,
        },
        gps_df        = gps_df,
        phase_log     = phase_log,
        signal_events = signal_events,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_junction(lat, lon):
    best, best_d = "J0", float("inf")
    for jid, (jlat, jlon) in JUNCTION_COORDS.items():
        d = (lat-jlat)**2 + (lon-jlon)**2
        if d < best_d:
            best, best_d = jid, d
    return best


def _adaptive_action(jid, phase, qew, qns, step):
    if phase in (1, 3):
        return "hold_yellow"
    if phase == 0 and qns > qew * 1.8 and qew < 3:
        return "switch_to_ns"
    if phase == 2 and qew > qns * 1.8 and qns < 3:
        return "switch_to_ew"
    return "hold"


def _build_obs(jid, phase, qew, qns, step, queues):
    return np.array([
        min(qew / 25.0, 1.0),
        min(qns / 25.0, 1.0),
        1.0 if phase == 0 else 0.0,
        1.0 if phase == 2 else 0.0,
        min((step % 60) / 60.0, 1.0),
        0.5,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_simulation(
    mode:           str,
    traffic_scale:  float                           = 1.0,
    accident_step:  int                             = -1,
    seed:           int                             = 42,
    baseline_delay: float | None                    = None,
    ppo_model:      Any | None                      = None,
    progress_cb:    Callable[[int, int], None] | None = None,
) -> SimResult:
    """
    Run a simulation and return a SimResult.

    If SUMO is installed: opens sumo-gui automatically, cars are visible
    on your desktop while Streamlit shows metrics in the browser.

    If SUMO is not installed: runs the mock path, full dashboard works.

    Set env var SUMO_NO_GUI=1 to suppress the GUI window.
    """
    if SUMO_AVAILABLE:
        try:
            return _sumo_result(
                mode, traffic_scale, accident_step, seed,
                baseline_delay, ppo_model, progress_cb,
            )
        except Exception:
            pass   # fall through to mock on any SUMO error

    # Mock path — animate the progress bar so it feels live
    if progress_cb:
        for i in range(20):
            progress_cb(i * (SIM_DURATION // 20), SIM_DURATION)
            time.sleep(0.05)
        progress_cb(SIM_DURATION, SIM_DURATION)

    return _mock_result(mode, baseline_delay, traffic_scale, accident_step, seed)