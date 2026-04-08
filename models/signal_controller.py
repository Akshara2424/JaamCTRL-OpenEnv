"""
signal_controller.py  -  Jaam Ctrl
====================================
Rule-based adaptive signal controller for 3 coordinated intersections.

Two controllers are provided:

  FixedTimeController
    No-op. SUMO runs its built-in fixed-time program.
    Used as the "before" baseline.

  RuleBasedController
    Per-step queue-aware green extension + coordinated green-wave.
    Logic per junction:
      - If E-W queue > threshold AND phase is E-W green → extend green
      - If N-S queue starved (N-S queue >> E-W queue) → cut E-W short
      - Green-wave offset pre-applied at t=1 so platoons hit J1 and J2 on green

    Central coordination:
      J0 runs at offset 0
      J1 runs at offset +GREEN_WAVE_OFFSET  (36 s @ 50 km/h, 500 m spacing)
      J2 runs at offset +2×GREEN_WAVE_OFFSET

Per-step signal state is returned for the phase timeline display in Streamlit.
"""

from __future__ import annotations

try:
    import traci
    TRACI_OK = True
except ImportError:
    TRACI_OK = False

# ── Junction / TL IDs (must match network.net.xml) ────────────────────────────
TL_IDS = ["J0", "J1", "J2"]

JUNCTION_EDGES: dict[str, dict[str, list[str]]] = {
    "J0": {
        "ew": ["W0J0", "J1J0"],
        "ns": ["N0J0", "S0J0"],
    },
    "J1": {
        "ew": ["J0J1", "J2J1"],
        "ns": ["N1J1", "S1J1"],
    },
    "J2": {
        "ew": ["J1J2"],
        "ns": ["N2J2", "S2J2"],
    },
}

# Phase indices (match tllogic.tll.xml)
PHASE_EW_GREEN  = 0
PHASE_EW_YELLOW = 1
PHASE_NS_GREEN  = 2
PHASE_NS_YELLOW = 3

# Timing (seconds)
MIN_GREEN          = 15
MAX_GREEN          = 60
DEFAULT_EW         = 35
DEFAULT_NS         = 30
YELLOW_DUR         = 5
QUEUE_THRESHOLD    = 5      # vehicles → trigger extension
GREEN_WAVE_OFFSET  = 36     # seconds between junctions (500 m / 13.89 m/s)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _queue_on_edge(edge_id: str) -> int:
    """Vehicles with speed < 0.5 m/s on an edge."""
    if not TRACI_OK:
        return 0
    try:
        vids = traci.edge.getLastStepVehicleIDs(edge_id)
        return sum(1 for v in vids if traci.vehicle.getSpeed(v) < 0.5)
    except Exception:
        return 0


def _total_queue(edges: list[str]) -> int:
    return sum(_queue_on_edge(e) for e in edges)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _safe_phase(tl: str) -> int:
    if not TRACI_OK:
        return 0
    try:
        return traci.trafficlight.getPhase(tl)
    except Exception:
        return 0


def _safe_next_switch(tl: str, sim_step: int) -> int:
    if not TRACI_OK:
        return 999
    try:
        return int(traci.trafficlight.getNextSwitch(tl)) - sim_step
    except Exception:
        return 999


# ══════════════════════════════════════════════════════════════════════════════
# Fixed-Time Baseline
# ══════════════════════════════════════════════════════════════════════════════

class FixedTimeController:
    """
    Pure no-op.  SUMO drives its built-in static TLS program.
    Provides a .step() interface identical to RuleBasedController
    so run_simulation.py can call them interchangeably.
    """

    def step(self, sim_step: int) -> dict[str, dict]:
        """Return current phase snapshot for each junction."""
        return {
            tl: {
                "phase":    _safe_phase(tl),
                "queue_ew": 0,
                "queue_ns": 0,
                "action":   "fixed",
            }
            for tl in TL_IDS
        }

    def reset(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Rule-Based Adaptive Controller
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedController:
    """
    Stateful queue-aware adaptive controller with green-wave coordination.

    Call reset() when the simulation resets, then step(sim_step) every second.
    step() returns a per-junction state dict for the Streamlit phase timeline.
    """

    def __init__(self):
        self._wave_applied = False

    def reset(self):
        self._wave_applied = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def step(self, sim_step: int) -> dict[str, dict]:
        """
        Run one adaptive control step.

        Returns
        -------
        dict[tl_id -> {phase, queue_ew, queue_ns, action}]
          action is one of: "extend_ew", "cut_ew", "extend_ns", "cut_ns", "hold"
        """
        # Apply green-wave offsets once at the start
        if not self._wave_applied and sim_step == 1:
            self._apply_green_wave()
            self._wave_applied = True

        state = {}
        for tl in TL_IDS:
            state[tl] = self._control_junction(tl, sim_step)
        return state

    # ── Internal ───────────────────────────────────────────────────────────────

    def _apply_green_wave(self):
        """
        Stagger J1 and J2 so a platoon released from J0
        arrives at J1 and J2 exactly when EW-green starts.
        """
        if not TRACI_OK:
            return
        offsets = {"J0": 0, "J1": GREEN_WAVE_OFFSET, "J2": GREEN_WAVE_OFFSET * 2}
        for tl, offset in offsets.items():
            if offset == 0:
                continue
            try:
                # Advance the phase timer so the cycle is offset
                remaining = max(MIN_GREEN, DEFAULT_EW - offset)
                traci.trafficlight.setPhase(tl, PHASE_EW_GREEN)
                traci.trafficlight.setPhaseDuration(tl, remaining)
            except Exception:
                pass

    def _control_junction(self, tl: str, sim_step: int) -> dict:
        edges     = JUNCTION_EDGES[tl]
        phase     = _safe_phase(tl)
        time_left = _safe_next_switch(tl, sim_step)
        q_ew      = _total_queue(edges["ew"])
        q_ns      = _total_queue(edges["ns"])
        action    = "hold"

        # Only act on green phases — leave yellow alone
        if phase == PHASE_EW_GREEN:
            if q_ew > QUEUE_THRESHOLD and time_left < 8:
                # Heavy E-W queue approaching switch → extend
                new_dur = _clamp(DEFAULT_EW + (q_ew - QUEUE_THRESHOLD) * 2,
                                 MIN_GREEN, MAX_GREEN)
                self._set_duration(tl, new_dur)
                action = "extend_ew"

            elif q_ns > q_ew * 1.8 and time_left > 10:
                # N-S side badly starved → cut E-W short
                self._set_duration(tl, MIN_GREEN)
                action = "cut_ew"

        elif phase == PHASE_NS_GREEN:
            if q_ns > QUEUE_THRESHOLD and time_left < 8:
                new_dur = _clamp(DEFAULT_NS + (q_ns - QUEUE_THRESHOLD) * 2,
                                 MIN_GREEN, MAX_GREEN)
                self._set_duration(tl, new_dur)
                action = "extend_ns"

            elif q_ew > q_ns * 2.0 and time_left > 8:
                # E-W arterial badly backed up → cut N-S short
                self._set_duration(tl, MIN_GREEN)
                action = "cut_ns"

        return {
            "phase":    phase,
            "queue_ew": q_ew,
            "queue_ns": q_ns,
            "action":   action,
        }

    @staticmethod
    def _set_duration(tl: str, dur: int):
        if not TRACI_OK:
            return
        try:
            traci.trafficlight.setPhaseDuration(tl, dur)
        except Exception:
            pass
