"""
───────────────────────────────────────────────────────────────────────────────
Task 2 — Medium: Two intersections (Barakhamba + CP Core).

Thin wrapper around JaamCTRLTrafficEnv(task_id=2).
See base_env.py for all implementation logic.

Task contract
─────────────
Intersections  : 2  (INT_1 Barakhamba, INT_2 CP Core — linear corridor)
Vehicle mix    : 60% motorcycles, 20% autos, 20% passenger cars
Demand         : 600 veh / hr  (moderate variation, no hard spike)
Incidents      : stochastic pedestrian crossings (Poisson λ=0.02 / step)
Probe noise    : none
Obs dim        : 46 (slot for INT_3 zero-padded)
Action space   : MultiDiscrete([4, 4])
Episode steps  : 500  (41.7 sim-minutes)
Success        : ≥20% delay reduction  AND  ≥15% stop reduction
                 vs fixed-time baseline

Key learning signal added over Task 1
──────────────────────────────────────
  green_wave bonus (+0.10 per hit) — agent must learn that signalling INT_2
  green ~13 seconds after INT_1 catches the outgoing platoon and earns reward.
"""

from __future__ import annotations

from typing import Any, Optional

from env.base_env import JaamCTRLTrafficEnv


class Task2Env(JaamCTRLTrafficEnv):
    """
    Convenience wrapper for Task 2 (Medium).
    Identical to JaamCTRLTrafficEnv(task_id=2); task_id is locked.
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
            task_id       = 2,
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