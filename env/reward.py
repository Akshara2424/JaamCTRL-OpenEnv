"""
───────────────────────────────────────────────────────────────────────────────
Reward computation for JaamCTRL.

All functions are pure (no side-effects, no SUMO calls).
base_env.py calls compute_reward() once per step.

Reward terms
────────────
  r_queue       penalise total vehicles waiting
  r_throughput  reward vehicles exiting intersections
  r_stops       penalise new stop events (deceleration to 0)
  r_long_wait   hard penalty for vehicles waiting beyond threshold
  r_green_wave  bonus for upstream/downstream green-wave alignment
  r_overflow    hard penalty when lane queue exceeds physical capacity
  r_incident    one-time bonus when queue behind an incident clears
  r_thrash      penalise rapid phase switching below MIN_GREEN_S

Coefficients for each term live entirely in TASK_CONFIGS["reward_coeffs"]
so this file never needs to change when tuning difficulty.
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

# Reward is clipped to this range every step to prevent exploding gradients.
REWARD_MIN = -10.0
REWARD_MAX =  +5.0


def compute_reward(
    telemetry:  Dict[str, Any],
    cfg:        Dict[str, Any],
    step_count: int,
    n_tl:       int,
) -> float:
    """
    Unified reward function.  Runs identically for all three task levels;
    difficulty scaling is achieved entirely through cfg["reward_coeffs"]
    and cfg["rush_hour_window"].

    Parameters
    ----------
    telemetry  : dict from observation.collect_telemetry()
                 Keys used:
                   queue_lengths     np.ndarray (3,4)
                   throughput        np.ndarray (3,)
                   new_stops         np.ndarray (3,)
                   long_wait_count   int
                   green_wave_hits   int
                   overflow_lanes    int
                   incident_cleared  bool
                   thrash_count      int
    cfg        : task config dict (TASK_CONFIGS[task_id])
    step_count : current episode step (int)
    n_tl       : number of active traffic lights this task (1, 2, or 3)

    Returns
    -------
    float  clipped to [REWARD_MIN, REWARD_MAX]
    """
    c = cfg["reward_coeffs"]

    # ── Term 1: Queue penalty ─────────────────────────────────────────────
    # Penalise every vehicle sitting in a queue at any active intersection.
    # Shape slice [:n_tl] zeros-out inactive intersections automatically.
    total_queue   = float(telemetry["queue_lengths"][:n_tl].sum())
    r_queue       = c["queue"] * total_queue

    # ── Term 2: Throughput reward ─────────────────────────────────────────
    # Reward vehicles that successfully exit an intersection this step.
    total_through  = float(telemetry["throughput"][:n_tl].sum())
    r_throughput   = c["throughput"] * total_through

    # ── Term 3: Stop penalty ──────────────────────────────────────────────
    # Penalise new stop events (vehicles decelerating to near-zero speed).
    total_stops   = float(telemetry["new_stops"][:n_tl].sum())
    r_stops       = c["stops"] * total_stops

    # ── Term 4: Long-wait hard penalty ───────────────────────────────────
    # Coefficient is 0.0 for Task 1 (threshold = 999 s, never triggered).
    r_long_wait   = c["long_wait"] * float(telemetry["long_wait_count"])

    # ── Term 5: Green-wave coordination bonus ────────────────────────────
    # Counts upstream platoons that found the downstream signal already green.
    # Coefficient is 0.0 for Task 1 (no multi-TL coordination required).
    r_green_wave  = c["green_wave"] * float(telemetry["green_wave_hits"])

    # ── Term 6: Overflow / spillover penalty ─────────────────────────────
    # Fires when queue depth exceeds LANE_CAPACITY on any lane.
    # Coefficient is 0.0 for Tasks 1 and 2.
    r_overflow    = c["overflow"] * float(telemetry["overflow_lanes"])

    # ── Term 7: Incident clearance one-time bonus ─────────────────────────
    # Fires once when the queue behind an active incident drops below 5 veh.
    # Coefficient is 0.0 for Tasks 1 and 2.
    r_incident    = c["incident"] * float(telemetry["incident_cleared"])

    # ── Term 8: Phase-thrash penalty ──────────────────────────────────────
    # Discourages epileptic phase switching before MIN_GREEN_S has elapsed.
    r_thrash      = c["thrash"] * float(telemetry.get("thrash_count", 0))

    # ── Rush-hour multiplier (Task 3 only) ────────────────────────────────
    # During the demand spike window the queue and throughput signals are
    # amplified so the agent feels stronger pressure to clear queues fast.
    rush_window = cfg.get("rush_hour_window")
    if rush_window and rush_window[0] <= step_count < rush_window[1]:
        r_queue      *= 1.5
        r_throughput *= 1.5

    # ── Assemble and clip ─────────────────────────────────────────────────
    total = (
        r_queue
        + r_throughput
        + r_stops
        + r_long_wait
        + r_green_wave
        + r_overflow
        + r_incident
        + r_thrash
    )
    return float(np.clip(total, REWARD_MIN, REWARD_MAX))


def reward_breakdown(
    telemetry:  Dict[str, Any],
    cfg:        Dict[str, Any],
    step_count: int,
    n_tl:       int,
) -> Dict[str, float]:
    """
    Same computation as compute_reward() but returns every term individually.
    Used by the Streamlit dashboard and inference.py for logging/debugging.

    Returns
    -------
    dict with keys matching the reward_coeffs keys, plus "total" and "clipped".
    """
    c = cfg["reward_coeffs"]

    total_queue   = float(telemetry["queue_lengths"][:n_tl].sum())
    total_through = float(telemetry["throughput"][:n_tl].sum())
    total_stops   = float(telemetry["new_stops"][:n_tl].sum())

    rush_window   = cfg.get("rush_hour_window")
    rush_active   = bool(
        rush_window and rush_window[0] <= step_count < rush_window[1]
    )
    rush_mult     = 1.5 if rush_active else 1.0

    terms = {
        "queue":      c["queue"]      * total_queue      * rush_mult,
        "throughput": c["throughput"] * total_through     * rush_mult,
        "stops":      c["stops"]      * total_stops,
        "long_wait":  c["long_wait"]  * float(telemetry["long_wait_count"]),
        "green_wave": c["green_wave"] * float(telemetry["green_wave_hits"]),
        "overflow":   c["overflow"]   * float(telemetry["overflow_lanes"]),
        "incident":   c["incident"]   * float(telemetry["incident_cleared"]),
        "thrash":     c["thrash"]     * float(telemetry.get("thrash_count", 0)),
    }
    raw    = sum(terms.values())
    clipped = float(np.clip(raw, REWARD_MIN, REWARD_MAX))
    return {**terms, "total_raw": raw, "total_clipped": clipped}