"""
signal_controller.py
Rule-based adaptive traffic signal controller for 3 coordinated intersections.

Logic:
  - Each junction independently monitors queue length on each incoming edge.
  - If the main arterial (E-W) queue > threshold → extend green on E-W phase.
  - If cross-street (N-S) queue > threshold → extend green on N-S phase.
  - Central coordinator applies a green-wave offset: J1 gets +offset, J2 gets +2*offset
    so a platoon cleared from J0 hits J1 green, then J2 green.

All calls use traci to read/set phase durations.
"""

import traci

# ── Constants ────────────────────────────────────────────────────────────────
TL_IDS = ["J0", "J1", "J2"]

# Edge groups per junction: (ew_edges, ns_edges)
JUNCTION_EDGES = {
    "J0": {
        "ew": ["W0J0", "J1J0"],   # main arterial approaches
        "ns": ["N0J0", "S0J0"],   # cross street approaches
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

# Phase indices (must match tlLogic in .net.xml)
PHASE_EW_GREEN = 0   # E-W green
PHASE_EW_YELLOW = 1
PHASE_NS_GREEN = 2   # N-S green
PHASE_NS_YELLOW = 3

# Timing limits (seconds)
MIN_GREEN   = 15
MAX_GREEN   = 60
DEFAULT_EW  = 35
DEFAULT_NS  = 30
YELLOW_DUR  = 5

# Queue threshold to trigger extension (vehicles)
QUEUE_THRESHOLD = 5

# Green-wave offset between consecutive junctions (seconds)
GREEN_WAVE_OFFSET = 12


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_queue(edge_id: str) -> int:
    """Number of vehicles with speed < 0.1 m/s on an edge (proxy for queue)."""
    try:
        vids = traci.edge.getLastStepVehicleIDs(edge_id)
        return sum(1 for v in vids if traci.vehicle.getSpeed(v) < 0.1)
    except traci.TraCIException:
        return 0


def _total_queue(edges: list[str]) -> int:
    return sum(_get_queue(e) for e in edges)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ── Main controller ───────────────────────────────────────────────────────────

class RuleBasedController:
    """
    Stateful rule-based controller.
    Call .step(sim_step) every simulation second.
    """

    def __init__(self):
        # Per-junction state: current phase duration remaining
        self._phase_timer = {tl: DEFAULT_EW for tl in TL_IDS}
        self._current_phase = {tl: PHASE_EW_GREEN for tl in TL_IDS}
        self._green_wave_active = False

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, sim_step: int) -> dict:
        """
        Run one control step.
        Returns a dict of {tl_id: current_phase} for dashboard display.
        """
        # Apply green-wave offset on first step
        if sim_step == 1:
            self._apply_green_wave_offsets()

        phase_info = {}
        for tl_id in TL_IDS:
            phase_info[tl_id] = self._control_junction(tl_id, sim_step)
        return phase_info

    # ── Internal ──────────────────────────────────────────────────────────────

    def _apply_green_wave_offsets(self):
        """Stagger initial phases so platoons hit consecutive junctions on green."""
        try:
            traci.trafficlight.setPhase("J1", PHASE_EW_GREEN)
            traci.trafficlight.setPhaseDuration(
                "J1", DEFAULT_EW - GREEN_WAVE_OFFSET
            )
            traci.trafficlight.setPhase("J2", PHASE_EW_GREEN)
            traci.trafficlight.setPhaseDuration(
                "J2", DEFAULT_EW - 2 * GREEN_WAVE_OFFSET
            )
        except traci.TraCIException:
            pass

    def _control_junction(self, tl_id: str, step: int) -> int:
        """Adaptive rule for a single junction; returns current phase index."""
        try:
            current = traci.trafficlight.getPhase(tl_id)
            time_in_phase = traci.trafficlight.getNextSwitch(tl_id) - step
        except traci.TraCIException:
            return 0

        # Only act on green phases; leave yellow phases alone
        if current not in (PHASE_EW_GREEN, PHASE_NS_GREEN):
            return current

        edges = JUNCTION_EDGES[tl_id]
        ew_q = _total_queue(edges["ew"])
        ns_q = _total_queue(edges["ns"])

        # Decide target green durations
        if current == PHASE_EW_GREEN:
            if ew_q > QUEUE_THRESHOLD and time_in_phase < 10:
                # Extend E-W green
                new_dur = _clamp(DEFAULT_EW + (ew_q - QUEUE_THRESHOLD) * 2,
                                 MIN_GREEN, MAX_GREEN)
                try:
                    traci.trafficlight.setPhaseDuration(tl_id, new_dur)
                except traci.TraCIException:
                    pass
            elif ns_q > ew_q * 1.5 and time_in_phase > 10:
                # Cut E-W short, give N-S a chance
                try:
                    traci.trafficlight.setPhaseDuration(tl_id, MIN_GREEN)
                except traci.TraCIException:
                    pass

        elif current == PHASE_NS_GREEN:
            if ns_q > QUEUE_THRESHOLD and time_in_phase < 10:
                new_dur = _clamp(DEFAULT_NS + (ns_q - QUEUE_THRESHOLD) * 2,
                                 MIN_GREEN, MAX_GREEN)
                try:
                    traci.trafficlight.setPhaseDuration(tl_id, new_dur)
                except traci.TraCIException:
                    pass
            elif ew_q > ns_q * 2 and time_in_phase > 8:
                try:
                    traci.trafficlight.setPhaseDuration(tl_id, MIN_GREEN)
                except traci.TraCIException:
                    pass

        return current


# ── Fixed-time controller (baseline) ─────────────────────────────────────────

class FixedTimeController:
    """
    Applies the default fixed-time program from the .net.xml (no changes).
    Used as the 'before' baseline.
    """

    def step(self, sim_step: int) -> dict:  # noqa: D401
        """No-op: SUMO runs its built-in fixed program."""
        return {tl: 0 for tl in TL_IDS}
