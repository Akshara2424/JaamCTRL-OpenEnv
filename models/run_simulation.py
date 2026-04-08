"""
run_simulation.py  -  Jaam Ctrl
=================================
Core simulation runner.  Supports three modes:

  "fixed"    SUMO built-in fixed-time program  (baseline)
  "adaptive" Rule-based queue-aware controller  (signal_controller.py)
  "rl"       Trained PPO agent                  (rl_agent.py)
             Falls back to "adaptive" if no model is loaded.

Returns a SimResult with:
  metrics       – avg_delay, avg_stops, throughput, improvement, per_junction
  gps_df        – GPS probe DataFrame  (for heatmap)
  phase_log     – per-step phase state for all 3 junctions  (for timeline chart)
  signal_events – list of every phase switch with timestamp and junction
  raw_delays    – per-vehicle accumulated delays
  raw_stops     – per-vehicle stop counts
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── src/ on path ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── TraCI ──────────────────────────────────────────────────────────────────────
try:
    import traci
    TRACI_OK = True
except ImportError:
    TRACI_OK = False

from gps_generator import build_dataframe, collect_gps_frame, select_probe_vehicles
from signal_controller import FixedTimeController, RuleBasedController

# ── Paths & constants ──────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(_HERE)
SUMO_CFG     = os.path.join(_ROOT, "sumo", "config.sumocfg")
SIM_DURATION = 1800          # seconds per run

TL_IDS = ["J0", "J1", "J2"]

JUNCTION_EDGES: dict[str, dict[str, list[str]]] = {
    "J0": {"ew": ["W0J0", "J1J0"], "ns": ["N0J0", "S0J0"]},
    "J1": {"ew": ["J0J1", "J2J1"], "ns": ["N1J1", "S1J1"]},
    "J2": {"ew": ["J1J2"],          "ns": ["N2J2", "S2J2"]},
}

# RL observation constants (must match rl_agent.py)
RL_MAX_QUEUE      = 25.0
RL_MAX_THROUGHPUT = 10.0
RL_MAX_PHASE_DUR  = 60.0
RL_CONTROL_STEP   = 10
RL_MIN_PHASE_DUR  = 15
RL_YELLOW_DUR     = 5

PHASE_EW_GREEN  = 0
PHASE_EW_YELLOW = 1
PHASE_NS_GREEN  = 2
PHASE_NS_YELLOW = 3

PHASE_LABELS = {
    PHASE_EW_GREEN:  "EW Green",
    PHASE_EW_YELLOW: "EW Yellow",
    PHASE_NS_GREEN:  "NS Green",
    PHASE_NS_YELLOW: "NS Yellow",
}


# ══════════════════════════════════════════════════════════════════════════════
# SimResult
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    mode:          str
    metrics:       dict                   = field(default_factory=dict)
    gps_df:        pd.DataFrame           = field(default_factory=pd.DataFrame)
    phase_log:     list[dict]             = field(default_factory=list)
    signal_events: list[dict]             = field(default_factory=list)
    raw_delays:    list[float]            = field(default_factory=list)
    raw_stops:     list[int]              = field(default_factory=list)
    controller_log: list[dict]            = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    mode:            str                        = "fixed",
    traffic_scale:   float                      = 1.0,
    accident_step:   int                        = -1,
    seed:            int                        = 42,
    baseline_delay:  Optional[float]            = None,
    ppo_model                                   = None,
    progress_cb:     Optional[Callable]         = None,
) -> SimResult:
    """
    Run one full Jaam Ctrl simulation episode.

    Parameters
    ----------
    mode           : "fixed" | "adaptive" | "rl"
    traffic_scale  : Vehicle flow multiplier (applied via TraCI scale command).
    accident_step  : Simulation second to inject a blocking vehicle (-1 = none).
    seed           : SUMO random seed for reproducibility.
    baseline_delay : avg_delay_s from a prior "fixed" run (for % improvement).
    ppo_model      : Loaded SB3 PPO model.  Required when mode="rl".
                     If None and mode="rl", falls back silently to "adaptive".
    progress_cb    : Optional callable(step: int, total: int) for progress bar.

    Returns
    -------
    SimResult
    """
    # ── SUMO not available → return mock data ─────────────────────────────────
    if not TRACI_OK:
        return _mock_result(mode, baseline_delay)

    # ── Resolve mode / controller ──────────────────────────────────────────────
    actual_mode = mode
    if mode == "rl" and ppo_model is None:
        actual_mode = "adaptive"   # silent fallback

    if actual_mode == "adaptive":
        controller = RuleBasedController()
        controller.reset()
    elif actual_mode == "rl":
        controller = None          # PPO drives actions
    else:
        controller = FixedTimeController()

    # ── RL per-junction phase state tracker ───────────────────────────────────
    rl_phase_timer = {tl: 0   for tl in TL_IDS}
    rl_cur_phase   = {tl: PHASE_EW_GREEN for tl in TL_IDS}

    # ── Launch SUMO ───────────────────────────────────────────────────────────
    sumo_cmd = [
        "sumo",
        "-c", SUMO_CFG,
        "--seed", str(seed),
        "--no-warnings",
        "--no-step-log",
        "--quit-on-end",
    ]
    traci.start(sumo_cmd)

    # Apply traffic scale
    if traffic_scale != 1.0:
        try:
            traci.simulation.setScale(traffic_scale)
        except Exception:
            pass

    # ── Accumulators ──────────────────────────────────────────────────────────
    gps_records:        list[dict]         = []
    phase_log:          list[dict]         = []
    signal_events:      list[dict]         = []
    controller_log:     list[dict]         = []
    delays_per_vehicle: dict[str, float]   = {}
    stops_per_vehicle:  dict[str, int]     = {}
    arrived_count:      int                = 0
    probe_vids:         set                = set()

    # Track previous phase to detect switches
    prev_phase = {tl: -1 for tl in TL_IDS}

    # ── Main loop ─────────────────────────────────────────────────────────────
    for step in range(SIM_DURATION):

        # Update probe vehicle set every 60 s
        if step % 60 == 0:
            all_ids    = traci.vehicle.getIDList()
            probe_vids = select_probe_vehicles(list(all_ids))

        # Collect GPS probes
        gps_records.extend(collect_gps_frame(step, probe_vids))

        # ── Signal control ────────────────────────────────────────────────────
        if actual_mode == "rl" and step % RL_CONTROL_STEP == 0:
            obs    = _build_rl_obs(rl_phase_timer, rl_cur_phase)
            action, _ = ppo_model.predict(obs, deterministic=True)
            ctrl_state = _apply_rl_action(
                int(action), rl_phase_timer, rl_cur_phase, step
            )
        elif controller is not None:
            ctrl_state = controller.step(step)
        else:
            ctrl_state = {tl: {"phase": _safe_phase(tl),
                               "queue_ew": 0, "queue_ns": 0,
                               "action": "fixed"}
                          for tl in TL_IDS}

        # ── Phase timeline snapshot (every 5 s) ───────────────────────────────
        if step % 5 == 0:
            snap: dict = {"step": step}
            for tl in TL_IDS:
                ph = _safe_phase(tl)
                snap[f"{tl}_phase"]    = ph
                snap[f"{tl}_label"]    = PHASE_LABELS.get(ph, "?")
                snap[f"{tl}_queue_ew"] = ctrl_state.get(tl, {}).get("queue_ew", 0)
                snap[f"{tl}_queue_ns"] = ctrl_state.get(tl, {}).get("queue_ns", 0)
                snap[f"{tl}_action"]   = ctrl_state.get(tl, {}).get("action", "")
            phase_log.append(snap)

        # ── Detect phase switches → signal_events ─────────────────────────────
        for tl in TL_IDS:
            ph = _safe_phase(tl)
            if ph != prev_phase[tl] and prev_phase[tl] != -1:
                signal_events.append({
                    "step":      step,
                    "junction":  tl,
                    "from_phase": PHASE_LABELS.get(prev_phase[tl], "?"),
                    "to_phase":   PHASE_LABELS.get(ph, "?"),
                    "mode":      actual_mode,
                })
            prev_phase[tl] = ph

        # ── Per-vehicle metrics ───────────────────────────────────────────────
        for vid in traci.vehicle.getIDList():
            delays_per_vehicle[vid] = _accumulated_delay(vid)
            stops_per_vehicle[vid]  = stops_per_vehicle.get(vid, 0) + _is_stopped(vid)

        arrived_count += traci.simulation.getArrivedNumber()

        # ── Accident injection ────────────────────────────────────────────────
        if step == accident_step:
            _inject_accident()

        # ── Advance ──────────────────────────────────────────────────────────
        traci.simulationStep()
        if progress_cb:
            progress_cb(step + 1, SIM_DURATION)

    traci.close()

    # ── Build result ──────────────────────────────────────────────────────────
    all_delays = list(delays_per_vehicle.values())
    all_stops  = list(stops_per_vehicle.values())
    metrics    = _build_metrics(
        all_delays, all_stops, arrived_count,
        actual_mode, baseline_delay, phase_log
    )

    return SimResult(
        mode           = actual_mode,
        metrics        = metrics,
        gps_df         = build_dataframe(gps_records),
        phase_log      = phase_log,
        signal_events  = signal_events,
        raw_delays     = all_delays,
        raw_stops      = all_stops,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RL helpers (18-dim obs matching rl_agent.py CorridorEnv._get_obs)
# ══════════════════════════════════════════════════════════════════════════════

def _build_rl_obs(
    phase_timer: dict[str, int],
    cur_phase:   dict[str, int],
) -> np.ndarray:
    """
    Build 18-dim observation vector for PPO inference.
    Layout: [q_ew, q_ns, ph_ew, ph_ns, t_norm, thru] × 3 junctions
    Must exactly match CorridorEnv._get_obs() in rl_agent.py.
    """
    obs: list[float] = []
    for tl in TL_IDS:
        edges  = JUNCTION_EDGES[tl]
        q_ew   = min(_sum_queue(edges["ew"])  / RL_MAX_QUEUE,      1.0)
        q_ns   = min(_sum_queue(edges["ns"])  / RL_MAX_QUEUE,      1.0)
        phase  = cur_phase[tl]
        ph_ew  = 1.0 if phase == PHASE_EW_GREEN else 0.0
        ph_ns  = 1.0 if phase == PHASE_NS_GREEN else 0.0
        t_norm = min(phase_timer[tl] / RL_MAX_PHASE_DUR,           1.0)
        thru   = min(_edge_throughput(edges["ew"] + edges["ns"])
                     / RL_MAX_THROUGHPUT,                           1.0)
        obs.extend([q_ew, q_ns, ph_ew, ph_ns, t_norm, thru])
    return np.array(obs, dtype=np.float32)


def _apply_rl_action(
    action:      int,
    phase_timer: dict[str, int],
    cur_phase:   dict[str, int],
    sim_step:    int,
) -> dict[str, dict]:
    """
    Decode 3-bit action and apply phase switches.
    Updates phase_timer and cur_phase in-place.
    Returns per-junction state dict for phase_log.
    """
    state = {}
    for i, tl in enumerate(TL_IDS):
        requested = bool(action & (1 << i))
        phase_timer[tl] += RL_CONTROL_STEP
        q_ew = _sum_queue(JUNCTION_EDGES[tl]["ew"])
        q_ns = _sum_queue(JUNCTION_EDGES[tl]["ns"])
        action_taken = "hold"

        cur = cur_phase[tl]
        # Don't interrupt yellow phases
        if cur not in (PHASE_EW_YELLOW, PHASE_NS_YELLOW):
            force   = phase_timer[tl] >= RL_MAX_PHASE_DUR
            allowed = requested and phase_timer[tl] >= RL_MIN_PHASE_DUR
            if force or allowed:
                _rl_switch_phase(tl, cur, cur_phase, phase_timer)
                action_taken = "rl_switch"

        state[tl] = {
            "phase":    cur_phase[tl],
            "queue_ew": q_ew,
            "queue_ns": q_ns,
            "action":   action_taken,
        }
    return state


def _rl_switch_phase(
    tl:          str,
    cur:         int,
    cur_phase:   dict[str, int],
    phase_timer: dict[str, int],
):
    """Toggle EW-green ↔ NS-green through a yellow phase."""
    try:
        if cur == PHASE_EW_GREEN:
            traci.trafficlight.setPhase(tl, PHASE_EW_YELLOW)
            traci.trafficlight.setPhaseDuration(tl, RL_YELLOW_DUR)
            cur_phase[tl] = PHASE_NS_GREEN
        elif cur == PHASE_NS_GREEN:
            traci.trafficlight.setPhase(tl, PHASE_NS_YELLOW)
            traci.trafficlight.setPhaseDuration(tl, RL_YELLOW_DUR)
            cur_phase[tl] = PHASE_EW_GREEN
        phase_timer[tl] = 0
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TraCI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_phase(tl: str) -> int:
    try:
        return traci.trafficlight.getPhase(tl)
    except Exception:
        return 0


def _sum_queue(edges: list[str]) -> float:
    total = 0.0
    for e in edges:
        try:
            vids   = traci.edge.getLastStepVehicleIDs(e)
            total += sum(1 for v in vids if traci.vehicle.getSpeed(v) < 0.5)
        except Exception:
            pass
    return total


def _edge_throughput(edges: list[str]) -> float:
    total = 0.0
    for e in edges:
        try:
            total += traci.edge.getLastStepVehicleNumber(e)
        except Exception:
            pass
    return total


def _accumulated_delay(vid: str) -> float:
    try:
        return traci.vehicle.getAccumulatedWaitingTime(vid)
    except Exception:
        return 0.0


def _is_stopped(vid: str) -> int:
    try:
        return 1 if traci.vehicle.getWaitingTime(vid) > 0 else 0
    except Exception:
        return 0


def _inject_accident():
    """Stall a random vehicle to simulate an accident."""
    try:
        vids = list(traci.vehicle.getIDList())
        if vids:
            victim = random.choice(vids)
            traci.vehicle.setSpeed(victim, 0.0)
            traci.vehicle.setSpeedMode(victim, 0)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Metrics builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_metrics(
    all_delays:     list[float],
    all_stops:      list[int],
    arrived:        int,
    mode:           str,
    baseline_delay: Optional[float],
    phase_log:      list[dict],
) -> dict:
    avg_delay = float(np.mean(all_delays)) if all_delays else 0.0
    avg_stops = float(np.mean(all_stops))  if all_stops  else 0.0
    improvement = 0.0
    if baseline_delay and baseline_delay > 0:
        improvement = round((baseline_delay - avg_delay) / baseline_delay * 100, 1)

    # Per-junction average queue from phase_log
    per_junction: dict[str, dict] = {}
    if phase_log:
        for tl in TL_IDS:
            ew_vals = [s.get(f"{tl}_queue_ew", 0) for s in phase_log]
            ns_vals = [s.get(f"{tl}_queue_ns", 0) for s in phase_log]
            per_junction[tl] = {
                "avg_queue_ew": round(float(np.mean(ew_vals)), 2),
                "avg_queue_ns": round(float(np.mean(ns_vals)), 2),
                "avg_queue":    round(float(np.mean(ew_vals)) +
                                      float(np.mean(ns_vals)), 2),
            }

    return {
        "mode":          mode,
        "avg_delay_s":   round(avg_delay, 2),
        "avg_stops":     round(avg_stops, 2),
        "throughput":    arrived,
        "improvement":   improvement,
        "per_junction":  per_junction,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Mock result (SUMO not installed)
# ══════════════════════════════════════════════════════════════════════════════

_MOCK_SEEDS = {"fixed": 1, "adaptive": 2, "rl": 3}

def _mock_result(mode: str, baseline_delay: Optional[float]) -> SimResult:
    """Realistic synthetic data so the Streamlit UI works without SUMO."""
    rng = np.random.default_rng(_MOCK_SEEDS.get(mode, 99))

    if mode == "fixed":
        avg_delay, avg_stops, throughput = (
            rng.uniform(45, 65), rng.uniform(3.5, 6.0), int(rng.integers(900, 1100))
        )
        improv = 0.0
    elif mode == "adaptive":
        avg_delay, avg_stops, throughput = (
            rng.uniform(28, 42), rng.uniform(2.0, 3.5), int(rng.integers(1050, 1250))
        )
        bl     = baseline_delay or 55.0
        improv = round((bl - avg_delay) / bl * 100, 1)
    else:  # rl
        avg_delay, avg_stops, throughput = (
            rng.uniform(18, 30), rng.uniform(1.2, 2.4), int(rng.integers(1150, 1400))
        )
        bl     = baseline_delay or 55.0
        improv = round((bl - avg_delay) / bl * 100, 1)

    # ── Synthetic GPS data ─────────────────────────────────────────────────────
    n  = 600
    xs = rng.uniform(-200, 1400, n)
    ys = rng.uniform(-300,  300, n)
    junction_x = {"J0": 0, "J1": 500, "J2": 1000}
    gps_df = pd.DataFrame({
        "time":    rng.integers(0, SIM_DURATION, n),
        "vehicle_id": [f"v{i}" for i in range(n)],
        "x":       xs,
        "y":       ys,
        "speed":   rng.uniform(0, 14, n),
        "vehicle_type": rng.choice(
            ["motorcycle", "car", "auto", "truck"],
            size=n, p=[0.6, 0.2, 0.1, 0.1]
        ),
        "junction_proximity": [
            min(TL_IDS, key=lambda j: abs(x - junction_x[j]))
            for x in xs
        ],
    })

    # ── Synthetic phase log ────────────────────────────────────────────────────
    phase_log = []
    phases    = {tl: PHASE_EW_GREEN for tl in TL_IDS}
    timers    = {tl: 0 for tl in TL_IDS}
    durations = {
        PHASE_EW_GREEN:  35, PHASE_EW_YELLOW: 5,
        PHASE_NS_GREEN:  30, PHASE_NS_YELLOW:  5,
    }
    next_phase_map = {
        PHASE_EW_GREEN: PHASE_EW_YELLOW, PHASE_EW_YELLOW: PHASE_NS_GREEN,
        PHASE_NS_GREEN: PHASE_NS_YELLOW, PHASE_NS_YELLOW: PHASE_EW_GREEN,
    }
    # Stagger for green-wave
    timers["J1"] = 36
    timers["J2"] = 72

    for step in range(0, SIM_DURATION, 5):
        snap: dict = {"step": step}
        for tl in TL_IDS:
            timers[tl] += 5
            if timers[tl] >= durations[phases[tl]]:
                phases[tl] = next_phase_map[phases[tl]]
                timers[tl] = 0

            # Adaptive / RL: vary queue lengths to look realistic
            q_ew = max(0, int(rng.normal(5 if phases[tl] == PHASE_NS_GREEN else 2, 2)))
            q_ns = max(0, int(rng.normal(5 if phases[tl] == PHASE_EW_GREEN else 2, 2)))
            snap[f"{tl}_phase"]    = phases[tl]
            snap[f"{tl}_label"]    = PHASE_LABELS[phases[tl]]
            snap[f"{tl}_queue_ew"] = q_ew
            snap[f"{tl}_queue_ns"] = q_ns
            snap[f"{tl}_action"]   = "hold"
        phase_log.append(snap)

    # ── Synthetic signal events ────────────────────────────────────────────────
    signal_events = [
        {"step": s["step"], "junction": tl,
         "from_phase": "EW Green", "to_phase": "EW Yellow", "mode": mode}
        for s in phase_log[::15]
        for tl in TL_IDS
    ]

    # ── Per-junction metrics ───────────────────────────────────────────────────
    per_junction = {}
    scale = {"fixed": 1.0, "adaptive": 0.65, "rl": 0.45}[mode]
    for tl in TL_IDS:
        per_junction[tl] = {
            "avg_queue_ew": round(float(rng.uniform(3, 8) * scale), 2),
            "avg_queue_ns": round(float(rng.uniform(2, 6) * scale), 2),
            "avg_queue":    round(float(rng.uniform(5, 14) * scale), 2),
        }

    metrics = {
        "mode":         mode,
        "avg_delay_s":  round(float(avg_delay), 2),
        "avg_stops":    round(float(avg_stops), 2),
        "throughput":   throughput,
        "improvement":  improv,
        "per_junction": per_junction,
    }
    return SimResult(
        mode          = mode,
        metrics       = metrics,
        gps_df        = gps_df,
        phase_log     = phase_log,
        signal_events = signal_events,
        raw_delays    = [float(rng.uniform(0, avg_delay * 2)) for _ in range(300)],
        raw_stops     = [int(rng.integers(0, int(avg_stops * 2) + 1)) for _ in range(300)],
    )
