""""
───────────────────────────────────────────────────────────────────────────────
Task 1 — Easy: Single intersection (Barakhamba Road).

This is a thin convenience wrapper around JaamCTRLTrafficEnv that:
  - Hard-codes task_id = 1 so callers never need to pass it
  - Documents the task contract clearly for judges and team members
  - Exposes no new logic — all behaviour lives in base_env / reward / observation

Usage
-----
    from env.task_easy import Task1Env

    env = Task1Env(mock_sumo=True)
    obs, info = env.reset()
    obs, r, term, trunc, info = env.step(env.action_space.sample())

    # Or via gymnasium.make after register_envs() is called:
    import gymnasium
    env = gymnasium.make("JaamCTRL-Easy-v0")
───────────────────────────────────────────────────────────────────────────────

Task contract
─────────────
Intersections  : 1  (INT_1 — Barakhamba Road)
Vehicle mix    : 100% passenger cars
Demand         : 400 veh / hr  (steady, no spike)
Incidents      : none
Probe noise    : none
Obs dim        : 46 (slots for INT_2 and INT_3 zero-padded)
Action space   : MultiDiscrete([4])
Episode steps  : 300  (25 sim-minutes at 5 s/step)
Success        : ≥15% delay reduction vs fixed-time baseline
"""

from __future__ import annotations

from typing import Any, Optional

from env.base_env import JaamCTRLTrafficEnv


class Task1Env(JaamCTRLTrafficEnv):
    """
    Convenience wrapper for Task 1 (Easy).
    Identical to JaamCTRLTrafficEnv(task_id=1); task_id is locked.
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
            task_id       = 1,
            sumo_cfg_path = sumo_cfg_path,
            use_gui       = use_gui,
            port          = port,
            seed          = seed,
            mock_sumo     = mock_sumo,
        )

    def reset(self, *, seed=None, options: Optional[Any] = None):
        # Strip any task_id from options — Task1Env is always task 1.
        if options:
            options = {k: v for k, v in options.items()
                       if k not in ("task_id", "difficulty")}
        return super().reset(seed=seed, options=options)