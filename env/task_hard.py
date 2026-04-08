"""
env/task_hard.py
───────────────────────────────────────────────────────────────────────────────
Task 3 — Hard: Full 3-intersection corridor (Barakhamba → CP Core → Patel Chowk).

Thin wrapper around JaamCTRLTrafficEnv(task_id=3).
See base_env.py for all implementation logic.

Task contract
─────────────
Intersections  : 3  (INT_1 Barakhamba, INT_2 CP Core, INT_3 Patel Chowk)
Vehicle mix    : 55% motorcycles, 25% passenger cars, 12% autos, 8% buses
Demand         : 900 veh / hr base; ×1.8 rush-hour spike at steps 300–400
Incidents:
  - Animal crossing  at INT_3, step 200, duration 10 steps
  - Accident lane block at INT_2, step 400, duration 50 steps
Probe noise    : Gaussian σ=0.05 on all GPS density readings
Obs dim        : 46 (full — all slots used)
Action space   : MultiDiscrete([4, 4, 4])
Episode steps  : 800  (66.7 sim-minutes)
Success        : ≥25% delay reduction
                 ≥20% throughput improvement
                 0 overflow events
                 vs fixed-time baseline

What makes this hard
────────────────────
1. Overflow penalty (-3.0 per lane)  — agent must actively prevent spillback
2. Incident clearance bonus (+2.0)   — agent rewarded for reactive queue drain
3. Rush-hour multiplier (×1.5)       — reward/penalty amplified during spike
4. Noisy probes                      — agent can't fully trust density readings;
                                       must rely on queue_lengths + phase_elapsed
5. Bus mix                           — slower queue discharge than Tasks 1 and 2;
                                       phase timing strategy must change
"""

from __future__ import annotations

from typing import Any, Optional

from env.base_env import JaamCTRLTrafficEnv


class Task3Env(JaamCTRLTrafficEnv):
    """
    Convenience wrapper for Task 3 (Hard).
    Identical to JaamCTRLTrafficEnv(task_id=3); task_id is locked.
    """

    def __init__(
        self,
        sumo_cfg_path: str           = "sumo/corridor.sumocfg",
        use_gui:       bool          = False,
        port:          int           = 0,
        seed:          Optional[int] = None,
        mock_sumo:     bool          = False,
    ) -> None:
        super().__init__(
            task_id       = 3,
            sumo_cfg_path = sumo_cfg_path,
            use_gui       = use_gui,
            port          = port,
            seed          = seed,
            mock_sumo     = mock_sumo,
        )

    def reset(self, *, seed=None, options: Optional[Any] = None):
        if options:
            options = {k: v for k, v in options.items()
                       if k not in ("task_id", "difficulty")}
        return super().reset(seed=seed, options=options)