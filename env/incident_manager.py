"""
env/incident_manager.py
───────────────────────────────────────────────────────────────────────────────
Stateful incident scheduler for JaamCTRL.

Handles two incident categories:

  Deterministic  — accident / animal events scripted at fixed episode steps
                   (used in Task 3; defined in TASK_CONFIGS["incident_schedule"])

  Stochastic     — pedestrian crossings generated via a Poisson process
                   (used in Task 2; controlled by poisson_lambda in the schedule)

Public API
----------
  manager = IncidentManager(cfg, rng)
  manager.reset()                          # call at episode start
  events  = manager.tick(step, n_tl)       # call once per env step
  lanes   = manager.get_active_flags()     # returns np.ndarray (3,) binary
  cleared = manager.check_clearance(queue_lengths)

The manager also communicates with TraCI (when not in mock mode) to reduce
lane speeds or force a brief all-red phase during the event window.
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("JaamCTRL.IncidentManager")

# Speed caps injected into SUMO lanes during incident (m/s)
ACCIDENT_SPEED_CAP = 2.0    # ~7 km/h — near standstill
ANIMAL_SPEED_CAP   = 5.0    # ~18 km/h — slow crawl
CLEAR_QUEUE_THRESHOLD = 5   # vehicles: below this = incident considered cleared


class IncidentManager:
    """
    Manages all traffic incidents within a single episode.

    Parameters
    ----------
    cfg       : task config dict (TASK_CONFIGS[task_id])
    rng       : numpy Generator for stochastic events
    mock_sumo : if True, skip all TraCI calls
    """

    def __init__(
        self,
        cfg:        Dict[str, Any],
        rng:        np.random.Generator,
        mock_sumo:  bool = False,
    ) -> None:
        self.cfg        = cfg
        self.rng        = rng
        self.mock_sumo  = mock_sumo

        # Runtime state — reset at episode start
        self._active:   List[Dict[str, Any]] = []   # currently active incidents
        self._history:  List[Dict[str, Any]] = []   # all incidents this episode
        self._cleared:  bool = False                 # latched per-step flag

        # Track which deterministic incidents have already been activated
        # so we do not double-fire them.
        self._fired_deterministic: set = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all incident state.  Call at the start of every episode."""
        self._active.clear()
        self._history.clear()
        self._fired_deterministic.clear()
        self._cleared = False
        logger.debug("IncidentManager reset.")

    def tick(
        self,
        step:  int,
        n_tl:  int,
        traci: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the incident schedule for the current episode step.

        - Activates new incidents whose start_step matches `step`.
        - Deactivates expired incidents.
        - Fires stochastic pedestrian events via Poisson draw.

        Parameters
        ----------
        step  : current episode step counter
        n_tl  : number of active traffic lights (caps tl_index)
        traci : live traci module (None in mock mode)

        Returns
        -------
        List of newly activated incident dicts this step (may be empty).
        """
        self._cleared = False   # reset the per-step clearance flag
        new_events: List[Dict] = []

        for spec in self.cfg["incident_schedule"]:
            incident_type = spec["type"]

            # ── Stochastic: pedestrian crossings ─────────────────────────
            if incident_type == "pedestrian":
                lam = spec.get("poisson_lambda", 0.02)
                if self.rng.random() < lam:
                    tl_idx = int(self.rng.integers(0, n_tl))
                    inc = {
                        "type":            "pedestrian",
                        "tl_index":        tl_idx,
                        "active":          True,
                        "start_step":      step,
                        "end_step":        step + 1,    # one step duration
                        "speed_cap":       None,         # handled via phase
                    }
                    self._active.append(inc)
                    self._history.append(inc)
                    new_events.append(inc)
                    logger.debug(
                        "Pedestrian crossing triggered at TL %d, step %d", tl_idx, step
                    )
                    if traci and not self.mock_sumo:
                        self._inject_pedestrian(traci, tl_idx)
                continue

            # ── Deterministic: accident / animal ─────────────────────────
            start    = spec.get("start_step", -1)
            duration = spec.get("duration_steps", 10)
            tl_idx   = spec.get("tl_index", 0)
            uid      = (incident_type, tl_idx, start)   # unique key

            # Activate on the correct step (only once)
            if step == start and uid not in self._fired_deterministic:
                self._fired_deterministic.add(uid)
                inc = {
                    "type":       incident_type,
                    "tl_index":   tl_idx,
                    "active":     True,
                    "start_step": start,
                    "end_step":   start + duration,
                }
                self._active.append(inc)
                self._history.append(inc)
                new_events.append(inc)
                logger.info(
                    "Incident '%s' activated at INT_%d, steps %d–%d",
                    incident_type, tl_idx + 1, start, start + duration,
                )
                if traci and not self.mock_sumo:
                    self._inject_lane_restriction(traci, tl_idx, incident_type)

        # Deactivate expired incidents
        for inc in self._active:
            if inc["active"] and step >= inc.get("end_step", step + 1):
                inc["active"] = False
                logger.info(
                    "Incident '%s' expired at INT_%d, step %d",
                    inc["type"], inc["tl_index"] + 1, step,
                )
                if traci and not self.mock_sumo:
                    self._restore_lane(traci, inc)

        # Prune cleared incidents from active list
        self._active = [i for i in self._active if i["active"]]
        return new_events

    def check_clearance(self, queue_lengths: np.ndarray) -> bool:
        """
        Inspect queue depths at intersections with active incidents.
        Latch self._cleared = True (and return True) when the queue behind
        a recently expired incident drops below CLEAR_QUEUE_THRESHOLD.

        Called once per step by base_env after telemetry collection.
        """
        for inc in self._history:
            if inc.get("type") in ("accident", "animal"):
                tl_idx = inc["tl_index"]
                if queue_lengths[tl_idx].sum() < CLEAR_QUEUE_THRESHOLD:
                    if not self._cleared:
                        logger.debug(
                            "Queue cleared at INT_%d after '%s'.",
                            tl_idx + 1, inc["type"],
                        )
                    self._cleared = True
        return self._cleared

    def get_active_flags(self) -> np.ndarray:
        """
        Return a (3,) binary array: 1 where an incident is currently active.
        Used by observation.build_obs() to populate incident_flag.
        """
        flags = np.zeros(3, dtype=np.int8)
        for inc in self._active:
            idx = inc.get("tl_index", 0)
            if 0 <= idx < 3:
                flags[idx] = 1
        return flags

    @property
    def incident_cleared(self) -> bool:
        """True if a queue clearance was detected this step."""
        return self._cleared

    @property
    def active_incidents(self) -> List[Dict[str, Any]]:
        """Snapshot of currently active incidents (for state() / logging)."""
        return [dict(i) for i in self._active]

    # ── TraCI injection helpers ───────────────────────────────────────────────

    def _inject_lane_restriction(
        self,
        traci,
        tl_index: int,
        incident_type: str,
    ) -> None:
        """
        Reduce max speed on the first controlled lane of the affected TL
        to simulate a blocked or restricted lane.
        """
        tl_id  = self.cfg["tl_ids"][tl_index]
        lanes  = traci.trafficlight.getControlledLanes(tl_id)
        cap    = ACCIDENT_SPEED_CAP if incident_type == "accident" else ANIMAL_SPEED_CAP
        target = lanes[:1] if incident_type == "accident" else lanes[:2]
        for lane in target:
            traci.lane.setMaxSpeed(lane, cap)
            logger.debug("Lane %s speed capped to %.1f m/s", lane, cap)

    def _restore_lane(self, traci, inc: Dict[str, Any]) -> None:
        """
        Restore default lane speed after an incident expires.
        SUMO default is 13.89 m/s (50 km/h) for urban arterials.
        """
        tl_id  = self.cfg["tl_ids"][inc["tl_index"]]
        lanes  = traci.trafficlight.getControlledLanes(tl_id)
        for lane in lanes[:2]:
            traci.lane.setMaxSpeed(lane, 13.89)
            logger.debug("Lane %s speed restored.", lane)

    def _inject_pedestrian(self, traci, tl_index: int) -> None:
        """
        Force a 5-second yellow phase on a TL to simulate a pedestrian crossing.
        Uses SUMO's setPhase; base_env's phase tracking will resync after.
        """
        tl_id = self.cfg["tl_ids"][tl_index]
        traci.trafficlight.setPhase(tl_id, 1)   # phase 1 = NS_YELLOW as proxy
        logger.debug("Pedestrian all-red injected at %s", tl_id)

    # ── Utility ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a serialisable summary of all incidents this episode."""
        return {
            "total_incidents":      len(self._history),
            "cleared_count":        sum(
                1 for i in self._history
                if not i.get("active", True)
            ),
            "incidents":            self._history,
        }

    def __repr__(self) -> str:
        return (
            f"IncidentManager("
            f"active={len(self._active)}, "
            f"history={len(self._history)}, "
            f"cleared={self._cleared})"
        )