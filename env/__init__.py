"""
───────────────────────────────────────────────────────────────────────────────
Package entry point.

Exports
-------
- TASK_CONFIGS          : dict  — all per-task parameters
- PHASE_DEFINITIONS     : dict  — phase index → label
- constants             : module-level signal/physics constants
- JaamCTRLTrafficEnv    : the main Gymnasium env (from base_env)
- Task1Env / Task2Env / Task3Env : convenience thin wrappers
- register_envs()       : registers all three gymnasium IDs

Import pattern
--------------
    from env import JaamCTRLTrafficEnv, register_envs
    register_envs()
    env = gymnasium.make("JaamCTRL-Hard-v0")
───────────────────────────────────────────────────────────────────────────────
"""

# ── Import OpenEnv models and graders ────────────────────────────────────────
from env.models import (
    ObservationData,
    ActionData,
    RewardData,
    RewardBreakdown,
    StepOutput,
    TaskMetrics,
)
from env.graders import (
    TaskGrader,
    Task1Grader,
    Task2Grader,
    Task3Grader,
    GRADERS,
    grade_task,
    grade_all_tasks,
)

# ── Physics / simulation constants ───────────────────────────────────────────

LANE_CAPACITY          = 20     # vehicles per lane before overflow penalty fires
PROBE_GRID_CELLS       = 8      # spatial cells in GPS heatmap around each TL
PLATOON_TRAVEL_TIME_S  = 13     # seconds: INT-1 → INT-2 at ~23 km/h over 300 m
MIN_GREEN_S            = 10     # minimum seconds before a non-yellow phase switch
YELLOW_PHASES          = frozenset({1, 3})   # phase indices that are yellow
YELLOW_DURATION        = {1: 3, 3: 3}        # fixed yellow durations (seconds)
DECISION_INTERVAL_S    = 5      # simulation seconds advanced per RL step

# ── Phase encoding (same scheme for every intersection) ──────────────────────
#   0 = NS green  / EW red
#   1 = NS yellow (3 s transition)
#   2 = EW green  / NS red
#   3 = EW yellow (3 s transition)

PHASE_DEFINITIONS = {
    0: "NS_GREEN",
    1: "NS_YELLOW",
    2: "EW_GREEN",
    3: "EW_YELLOW",
}

# ── Task configurations ───────────────────────────────────────────────────────
# Every env behaviour is driven from this dict.
# base_env.py reads cfg = TASK_CONFIGS[task_id] and never hard-codes values.

TASK_CONFIGS = {
    # ── Task 1 ────────────────────────────────────────────────────────────────
    1: {
        "name":                 "Easy — Barakhamba Road (Single Intersection)",
        "active_intersections": 1,
        "tl_ids":               ["INT_1"],
        "max_steps":            300,
        "decision_interval_s":  DECISION_INTERVAL_S,
        "demand_veh_per_hr":    400,
        "vehicle_mix":          {"passenger": 1.0},
        "probe_noise_sigma":    0.0,
        "incident_schedule":    [],
        "rush_hour_window":     None,
        "long_wait_threshold_s": 999,           # effectively disabled at Task 1
        "route_file":           "sumo/easy.rou.xml",

        # Reward coefficients — inactive terms are exactly 0.0
        "reward_coeffs": {
            "queue":      -0.01,
            "throughput": +0.05,
            "stops":      -0.02,
            "long_wait":   0.00,
            "green_wave":  0.00,
            "overflow":    0.00,
            "incident":    0.00,
            "thrash":     -0.05,
        },

        "success_thresholds": {
            "delay_reduction_pct": 15.0,
        },
    },

    # ── Task 2 ────────────────────────────────────────────────────────────────
    2: {
        "name":                 "Medium — Barakhamba + CP Core (Two Intersections)",
        "active_intersections": 2,
        "tl_ids":               ["INT_1", "INT_2"],
        "max_steps":            500,
        "decision_interval_s":  DECISION_INTERVAL_S,
        "demand_veh_per_hr":    600,
        "vehicle_mix":          {
            "passenger":  0.20,
            "motorcycle": 0.60,
            "tuk_tuk":    0.20,
        },
        "probe_noise_sigma":    0.0,
        "incident_schedule":    [
            # Stochastic pedestrian crossings; evaluated every step
            {"type": "pedestrian", "poisson_lambda": 0.02},
        ],
        "rush_hour_window":     None,
        "long_wait_threshold_s": 60,
        "route_file":           "sumo/medium.rou.xml",

        "reward_coeffs": {
            "queue":      -0.01,
            "throughput": +0.05,
            "stops":      -0.02,
            "long_wait":  -0.50,
            "green_wave": +0.10,
            "overflow":    0.00,
            "incident":    0.00,
            "thrash":     -0.05,
        },

        "success_thresholds": {
            "delay_reduction_pct": 20.0,
            "stop_reduction_pct":  15.0,
        },
    },

    # ── Task 3 ────────────────────────────────────────────────────────────────
    3: {
        "name":                 "Hard — Full Corridor: Barakhamba → CP Core → Patel Chowk",
        "active_intersections": 3,
        "tl_ids":               ["INT_1", "INT_2", "INT_3"],
        "max_steps":            800,
        "decision_interval_s":  DECISION_INTERVAL_S,
        "demand_veh_per_hr":    900,
        "vehicle_mix":          {
            "passenger":  0.25,
            "motorcycle": 0.55,
            "tuk_tuk":    0.12,
            "bus":        0.08,
        },
        "probe_noise_sigma":    0.05,           # Gaussian noise on GPS probes
        "incident_schedule":    [
            {"type": "accident",  "tl_index": 1, "start_step": 400, "duration_steps": 50},
            {"type": "animal",    "tl_index": 2, "start_step": 200, "duration_steps": 10},
        ],
        "rush_hour_window":     (300, 400),     # episode steps, not sim seconds
        "long_wait_threshold_s": 90,
        "route_file":           "sumo/hard.rou.xml",

        "reward_coeffs": {
            "queue":      -0.01,
            "throughput": +0.05,
            "stops":      -0.02,
            "long_wait":  -2.00,
            "green_wave": +0.15,
            "overflow":   -3.00,
            "incident":   +2.00,
            "thrash":     -0.05,
        },

        "success_thresholds": {
            "delay_reduction_pct":          25.0,
            "throughput_improvement_pct":   20.0,
            "overflow_events":               0,
        },
    },
}

# ── Public re-exports ─────────────────────────────────────────────────────────
from env.base_env       import JaamCTRLTrafficEnv          # noqa: E402
from env.task_easy      import Task1Env                    # noqa: E402
from env.task_medium    import Task2Env                    # noqa: E402
from env.task_hard      import Task3Env                    # noqa: E402


def register_envs() -> None:
    """
    Register all three task variants in the Gymnasium global registry.

    Call once at programme start (or from your inference.py top-level):

        from env import register_envs
        register_envs()

    Then use standard gymnasium.make():

        env = gymnasium.make("JaamCTRL-Easy-v0")
        env = gymnasium.make("JaamCTRL-Medium-v0", use_gui=True)
        env = gymnasium.make("JaamCTRL-Hard-v0",   seed=42)
    """
    import gymnasium

    _IDS = [
        ("JaamCTRL-Easy-v0",   1),
        ("JaamCTRL-Medium-v0", 2),
        ("JaamCTRL-Hard-v0",   3),
    ]
    for env_id, task_id in _IDS:
        if env_id not in gymnasium.envs.registry:
            gymnasium.register(
                id=env_id,
                entry_point="env.base_env:JaamCTRLTrafficEnv",
                kwargs={"task_id": task_id},
                max_episode_steps=TASK_CONFIGS[task_id]["max_steps"],
            )


# Auto-register on package import
register_envs()