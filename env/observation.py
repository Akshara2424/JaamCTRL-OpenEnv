"""
env/observation.py
───────────────────────────────────────────────────────────────────────────────
Observation construction and GPS probe density computation for JaamCTRL.

Responsibilities
----------------
1. collect_telemetry()  — pull raw metrics from TraCI (or generate mocks)
2. build_obs()          — assemble the Gymnasium observation dict from
                          telemetry + internal env state
3. compute_probe_density() — bin vehicle positions into an 8-cell heatmap
4. flatten_obs()        — produce the 46-dim float32 vector for the PPO head

Observation vector layout (46 floats, always)
──────────────────────────────────────────────
  [0:12]   queue_lengths  (3 intersections × 4 lanes)   — zero-padded if n_tl < 3
  [12:15]  current_phase  (3,)                           — zero-padded
  [15:18]  phase_elapsed  (3,)                           — zero-padded
  [18:42]  probe_density  (3 × 8 cells)                  — zero-padded + noisy in T3
  [42:45]  incident_flag  (3,)                           — zero-padded
  [45:46]  time_of_day_norm (1,)

All tasks use the same 46-dim flat vector so a single PPO model can be
evaluated across all three difficulty levels without architecture changes.
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Re-import shared constants — observation.py has no other intra-package deps.
from env import (
    LANE_CAPACITY,
    PROBE_GRID_CELLS,
    PLATOON_TRAVEL_TIME_S,
)

# Observation vector total length — must equal the sum in the layout above.
OBS_DIM = 46


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. Telemetry collection                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def collect_telemetry(
    traci,
    cfg:              Dict[str, Any],
    n_tl:             int,
    rng:              np.random.Generator,
    phase_history,    # collections.deque of (sim_time_s, tl_index, phase)
    active_incidents: List[Dict],
    incident_cleared_flag: bool,
) -> Dict[str, Any]:
    """
    Pull per-intersection metrics from an active TraCI connection.

    Returns a flat dict of numpy arrays / scalars consumed by
    build_obs() and reward.compute_reward().

    Parameters
    ----------
    traci             : live traci module (already connected)
    cfg               : task config dict (TASK_CONFIGS[task_id])
    n_tl              : number of active traffic lights
    rng               : numpy Generator for probe noise
    phase_history     : deque of recent phase changes for green-wave counting
    active_incidents  : list of current incident dicts (from IncidentManager)
    incident_cleared_flag : bool set by IncidentManager when a queue clears
    """
    queue_lengths     = np.zeros((3, 4), dtype=np.float32)
    throughput        = np.zeros(3,      dtype=np.float32)
    new_stops         = np.zeros(3,      dtype=np.float32)
    waiting_times     = np.zeros(3,      dtype=np.float32)
    probe_density_raw = np.zeros((3, 8), dtype=np.float32)
    overflow_lanes    = 0
    long_wait_count   = 0
    threshold_s       = cfg["long_wait_threshold_s"]

    all_vehicle_ids = traci.vehicle.getIDList()

    for i, tl_id in enumerate(cfg["tl_ids"]):
        lanes = traci.trafficlight.getControlledLanes(tl_id)

        for j, lane in enumerate(lanes[:4]):          # cap at 4 approach lanes
            q = traci.lane.getLastStepHaltingNumber(lane)
            queue_lengths[i, j] = float(q)

            if q > LANE_CAPACITY:
                overflow_lanes += 1

            for vid in traci.lane.getLastStepVehicleIDs(lane):
                if traci.vehicle.getWaitingTime(vid) > threshold_s:
                    long_wait_count += 1

        # Throughput — prefer induction loop, fall back to edge counts
        loop_id = f"loop_{tl_id}"
        edge_id = f"edge_out_{tl_id}"
        loop_ids = traci.inductionloop.getIDList()
        edge_ids = traci.edge.getIDList()

        if loop_id in loop_ids:
            throughput[i] = float(traci.inductionloop.getLastStepVehicleNumber(loop_id))
        elif edge_id in edge_ids:
            throughput[i] = float(traci.edge.getLastStepVehicleNumber(edge_id))
        else:
            # Heuristic: estimate from queue drain
            throughput[i] = max(0.0, float(queue_lengths[i].sum()) * 0.1)

        # New stops: vehicles that just decelerated to near-zero
        new_stops[i] = float(
            sum(
                1 for vid in all_vehicle_ids
                if traci.vehicle.getSpeed(vid) < 0.1
                and traci.vehicle.getAcceleration(vid) < -0.5
            )
        )

        # Total waiting time on controlled lanes
        waiting_times[i] = float(
            sum(traci.lane.getWaitingTime(l) for l in lanes[:4])
        )

        # GPS probe density heatmap
        probe_density_raw[i] = _compute_probe_density(
            traci, tl_id, all_vehicle_ids,
            noise_sigma=cfg["probe_noise_sigma"],
            rng=rng,
        )

    green_wave_hits = _count_green_wave_hits(phase_history, n_tl)

    return {
        "queue_lengths":        queue_lengths,
        "throughput":           throughput,
        "new_stops":            new_stops,
        "total_waiting_time_s": float(waiting_times.sum()),
        "overflow_lanes":       int(overflow_lanes),
        "long_wait_count":      int(long_wait_count),
        "probe_density_raw":    probe_density_raw,
        "incident_cleared":     bool(incident_cleared_flag),
        "green_wave_hits":      int(green_wave_hits),
    }


def mock_telemetry(
    rng: np.random.Generator,
    n_tl: int,
    noise_sigma: float = 0.0,
) -> Dict[str, Any]:
    """
    Synthetic telemetry for use when SUMO is not available.
    Produces plausible but random values; useful for unit-testing
    reward / observation pipelines without a SUMO installation.
    """
    queue_lengths = rng.integers(0, 12, size=(3, 4)).astype(np.float32)
    probe_raw     = rng.random((3, 8)).astype(np.float32)
    if noise_sigma > 0:
        probe_raw = np.clip(
            probe_raw + rng.normal(0, noise_sigma, probe_raw.shape), 0.0, 1.0
        ).astype(np.float32)

    return {
        "queue_lengths":        queue_lengths,
        "throughput":           rng.integers(2, 8, size=(3,)).astype(np.float32),
        "new_stops":            rng.integers(0, 4, size=(3,)).astype(np.float32),
        "total_waiting_time_s": float(queue_lengths[:n_tl].sum() * 5),
        "overflow_lanes":       int(rng.integers(0, 2)),
        "long_wait_count":      int(rng.integers(0, 3)),
        "probe_density_raw":    probe_raw,
        "incident_cleared":     False,
        "green_wave_hits":      int(rng.integers(0, 2)),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. Observation builder                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_obs(
    telemetry:        Dict[str, Any],
    current_phases:   np.ndarray,   # shape (3,) int32
    phase_elapsed:    np.ndarray,   # shape (3,) float32  seconds
    active_incidents: List[Dict],
    step_count:       int,
    max_steps:        int,
    n_tl:             int,
) -> Dict[str, Any]:
    """
    Assemble the full observation dict from telemetry + env state.

    Lower tasks zero-pad slots belonging to inactive intersections so the
    flat 46-dim vector always has the same shape regardless of task.

    Returns
    -------
    dict with keys:
        queue_lengths    np.ndarray (3,4)  float32
        current_phase    np.ndarray (3,)   int64
        phase_elapsed    np.ndarray (3,)   float32
        probe_density    np.ndarray (3,8)  float32
        incident_flag    np.ndarray (3,)   int8
        time_of_day_norm np.ndarray (1,)   float32
        flat             np.ndarray (46,)  float32   ← PPO policy input
    """
    # Full (3,4) and (3,8) arrays — inactive rows stay zero
    queue_lengths = np.zeros((3, 4), dtype=np.float32)
    probe_density = np.zeros((3, 8), dtype=np.float32)
    incident_flag = np.zeros(3,      dtype=np.int8)

    queue_lengths[:n_tl] = telemetry["queue_lengths"][:n_tl]
    probe_density[:n_tl] = telemetry["probe_density_raw"][:n_tl]

    for inc in active_incidents:
        if inc.get("active"):
            idx = inc.get("tl_index", 0)
            if idx < 3:
                incident_flag[idx] = 1

    time_norm = np.array(
        [step_count / max(1, max_steps)],
        dtype=np.float32,
    )

    obs = {
        "queue_lengths":    queue_lengths,
        "current_phase":    current_phases.copy().astype(np.int64),
        "phase_elapsed":    phase_elapsed.copy(),
        "probe_density":    probe_density,
        "incident_flag":    incident_flag,
        "time_of_day_norm": time_norm,
    }
    obs["flat"] = flatten_obs(obs)
    return obs


def flatten_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    Produce the canonical 46-dim float32 vector used as PPO policy input.

    Layout
    ------
    [0:12]   queue_lengths.flatten()
    [12:15]  current_phase (cast to float32)
    [15:18]  phase_elapsed
    [18:42]  probe_density.flatten()
    [42:45]  incident_flag (cast to float32)
    [45:46]  time_of_day_norm
    """
    flat = np.concatenate([
        obs["queue_lengths"].flatten(),                # 12
        obs["current_phase"].astype(np.float32),       #  3
        obs["phase_elapsed"].astype(np.float32),       #  3
        obs["probe_density"].flatten(),                # 24
        obs["incident_flag"].astype(np.float32),       #  3
        obs["time_of_day_norm"].astype(np.float32),    #  1
    ]).astype(np.float32)

    assert flat.shape == (OBS_DIM,), (
        f"flatten_obs: expected ({OBS_DIM},) got {flat.shape}"
    )
    return flat


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. GPS probe density                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _compute_probe_density(
    traci,
    tl_id:          str,
    all_vehicle_ids: List[str],
    noise_sigma:    float,
    rng:            np.random.Generator,
    radius_m:       float = 200.0,
) -> np.ndarray:
    """
    Bin vehicle positions within `radius_m` of a junction into 8 angular cells
    (each covering 45°).  Returns a normalised (0–1) density array of shape (8,).

    In Task 3, Gaussian noise (sigma = probe_noise_sigma from cfg) is added to
    simulate the sparsity and inaccuracy of real-world GPS probe feeds.

    Parameters
    ----------
    traci           : active traci module
    tl_id           : SUMO junction ID of the traffic light
    all_vehicle_ids : cached list from traci.vehicle.getIDList()
    noise_sigma     : std-dev of additive Gaussian noise (0 = clean)
    rng             : numpy Generator for reproducible noise
    radius_m        : detection radius around the junction centre
    """
    cells  = np.zeros(PROBE_GRID_CELLS, dtype=np.float32)
    jx, jy = traci.junction.getPosition(tl_id)

    for vid in all_vehicle_ids:
        vx, vy = traci.vehicle.getPosition(vid)
        dx, dy = vx - jx, vy - jy
        dist   = (dx * dx + dy * dy) ** 0.5
        if dist < radius_m:
            angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
            cell  = int(angle / 45.0) % PROBE_GRID_CELLS
            cells[cell] += 1.0

    # Normalise to [0, 1]
    max_val = cells.max()
    if max_val > 0:
        cells /= max_val

    # Additive probe noise (Task 3 only; sigma = 0 is a no-op)
    if noise_sigma > 0:
        noise  = rng.normal(0.0, noise_sigma, size=PROBE_GRID_CELLS).astype(np.float32)
        cells  = np.clip(cells + noise, 0.0, 1.0)

    return cells


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. Green-wave hit counter                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _count_green_wave_hits(
    phase_history,          # deque of (sim_time_s, tl_index, phase_int)
    n_tl: int,
    travel_window_s: float = 3.0,
) -> int:
    """
    Count how many upstream platoons hit a green light at the downstream signal.

    Logic
    -----
    For each NS_GREEN start at INT_1 (tl_index=0, phase=0) recorded in
    phase_history, check whether INT_2 (tl_index=1) was also in NS_GREEN
    within ±travel_window_s of (t_INT1_green + PLATOON_TRAVEL_TIME_S).

    Returns 0 for Task 1 (n_tl < 2) and is capped at 5 per step to prevent
    reward explosion when the full history is long.
    """
    if n_tl < 2:
        return 0

    hits = 0
    for t1, idx1, p1 in phase_history:
        if idx1 != 0 or p1 != 0:       # only INT_1 NS_GREEN starts
            continue
        target_t = t1 + PLATOON_TRAVEL_TIME_S
        for t2, idx2, p2 in phase_history:
            if idx2 != 1 or p2 != 0:   # only INT_2 NS_GREEN
                continue
            if abs(t2 - target_t) <= travel_window_s:
                hits += 1
                break                  # count each platoon once

    return min(hits, 5)                # cap to avoid per-step reward explosion