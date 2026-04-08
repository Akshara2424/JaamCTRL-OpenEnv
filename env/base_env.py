"""
───────────────────────────────────────────────────────────────────────────────
JaamCTRLTrafficEnv — core Gymnasium environment loop.

This class owns:
  - SUMO subprocess lifecycle (_launch_sumo, _close_sumo)
  - Action application and yellow-phase safety enforcement
  - Delegating to observation.py, reward.py, incident_manager.py
  - Episode metrics accumulation and success checking
  - The three Gymnasium contract methods: reset(), step(), state()

It does NOT define task configs, reward coefficients, or observation math —
those all live in their respective modules.
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── TraCI import ─────────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# ── Intra-package imports ─────────────────────────────────────────────────────
from env import (
    TASK_CONFIGS,
    YELLOW_PHASES,
    YELLOW_DURATION,
    MIN_GREEN_S,
)
from env.reward          import compute_reward, reward_breakdown
from env.observation     import (
    collect_telemetry,
    mock_telemetry,
    build_obs,
    OBS_DIM,
)
from env.incident_manager import IncidentManager

logger = logging.getLogger("JaamCTRL.BaseEnv")


class JaamCTRLTrafficEnv(gym.Env):
    """
    Gymnasium-compatible adaptive traffic signal control environment.

    Supports 3 progressive difficulty tasks via `task_id`.
    Designed for use with the OpenEnv judging harness and stable-baselines3.

    Parameters
    ----------
    task_id       : 1 = Easy, 2 = Medium, 3 = Hard
    sumo_cfg_path : path to .sumocfg (default "sumo/corridor.sumocfg")
    use_gui       : launch sumo-gui (True) or headless sumo (False)
    port          : TraCI port; 0 = auto-select a free port
    seed          : RNG seed for full reproducibility
    mock_sumo     : skip SUMO entirely, use synthetic observations
                    (useful for CI / unit tests without SUMO installed)
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 10}

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        task_id:       int  = 1,
        sumo_cfg_path: str  = "sumo/corridor.sumocfg",
        use_gui:       bool = False,
        port:          int  = 0,
        seed:          Optional[int] = None,
        mock_sumo:     bool = False,
    ) -> None:
        super().__init__()

        assert task_id in (1, 2, 3), f"task_id must be 1, 2 or 3; got {task_id}"

        self.task_id   = task_id
        self.cfg       = TASK_CONFIGS[task_id]
        self.use_gui   = use_gui
        self.port      = port
        self.mock_sumo = mock_sumo or not TRACI_AVAILABLE
        self.sumo_cfg  = Path(sumo_cfg_path)

        self._rng = np.random.default_rng(seed)
        self.n_tl = self.cfg["active_intersections"]

        # ── Spaces ────────────────────────────────────────────────────────
        # Action: phase index per active TL (0–3 each)
        self.action_space = spaces.MultiDiscrete([4] * self.n_tl)

        # Observation: Dict with a pre-flattened "flat" key for PPO
        self.observation_space = spaces.Dict({
            "queue_lengths":    spaces.Box(0.0, 50.0,  shape=(3, 4), dtype=np.float32),
            "current_phase":    spaces.MultiDiscrete([4, 4, 4]),
            "phase_elapsed":    spaces.Box(0.0, 120.0, shape=(3,),   dtype=np.float32),
            "probe_density":    spaces.Box(0.0, 1.0,   shape=(3, 8), dtype=np.float32),
            "incident_flag":    spaces.MultiBinary(3),
            "time_of_day_norm": spaces.Box(0.0, 1.0,   shape=(1,),   dtype=np.float32),
            "flat":             spaces.Box(-1.0, 50.0,  shape=(OBS_DIM,), dtype=np.float32),
        })

        # ── Internal state ────────────────────────────────────────────────
        self._step_count         = 0
        self._sim_time_s         = 0.0
        self._sumo_process       = None
        self._phase_elapsed      = np.zeros(3, dtype=np.float32)
        self._current_phases     = np.zeros(3, dtype=np.int32)
        self._episode_throughput = np.zeros(3, dtype=np.float32)
        self._episode_delay_sum  = 0.0
        self._episode_stops      = 0
        self._overflow_events    = 0
        self._last_telemetry: Dict[str, Any] = {}

        # Phase history for green-wave computation (observation.py)
        # Entries: (sim_time_s: float, tl_index: int, phase: int)
        self._phase_history: deque = deque(maxlen=200)

        # Baseline metrics for episode_summary comparison
        self._baseline_avg_delay:   Optional[float] = None
        self._baseline_throughput:  Optional[float] = None

        # Incident manager — owns all chaos events
        self._incident_mgr = IncidentManager(
            cfg=self.cfg,
            rng=self._rng,
            mock_sumo=self.mock_sumo,
        )

        logger.info(
            "JaamCTRLTrafficEnv | task=%d | %s | n_tl=%d | mock=%s",
            task_id, self.cfg["name"], self.n_tl, self.mock_sumo,
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed:    Optional[int]          = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset to the start of a new episode.

        `options` dict accepts:
          "task_id"  / "difficulty" : int  — switch task on reset
          "use_gui"                 : bool — override GUI flag

        Returns
        -------
        obs  : dict (see observation_space)
        info : dict (task metadata + reset flag)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Optional task switch
        if options:
            new_task = options.get("task_id") or options.get("difficulty")
            if new_task and int(new_task) != self.task_id:
                self.task_id  = int(new_task)
                self.cfg      = TASK_CONFIGS[self.task_id]
                self.n_tl     = self.cfg["active_intersections"]
                self.action_space = spaces.MultiDiscrete([4] * self.n_tl)
                self._incident_mgr = IncidentManager(
                    cfg=self.cfg, rng=self._rng, mock_sumo=self.mock_sumo
                )
                logger.info("Task switched to %d on reset.", self.task_id)
            if "use_gui" in options:
                self.use_gui = bool(options["use_gui"])

        self._close_sumo()

        # Reset episode counters
        self._step_count             = 0
        self._sim_time_s             = 0.0
        self._phase_elapsed[:]       = 0.0
        self._current_phases[:]      = 0
        self._episode_throughput[:]  = 0.0
        self._episode_delay_sum      = 0.0
        self._episode_stops          = 0
        self._overflow_events        = 0
        self._last_telemetry         = {}
        self._phase_history.clear()
        self._incident_mgr.reset()

        if not self.mock_sumo:
            self._launch_sumo()

        # Generate first observation via mock telemetry (no sim steps yet)
        tel = (
            mock_telemetry(self._rng, self.n_tl, self.cfg["probe_noise_sigma"])
            if self.mock_sumo
            else self._fetch_telemetry()
        )
        self._last_telemetry = tel

        obs  = build_obs(
            telemetry=tel,
            current_phases=self._current_phases,
            phase_elapsed=self._phase_elapsed,
            active_incidents=self._incident_mgr.active_incidents,
            step_count=self._step_count,
            max_steps=self.cfg["max_steps"],
            n_tl=self.n_tl,
        )
        info = self._build_info(reward=0.0, terminated=False, truncated=False)
        info["reset"] = True
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Apply phase actions and advance the simulation by one decision step.

        Parameters
        ----------
        action : np.ndarray  shape (n_tl,)  dtype int
            Phase index (0–3) for each active TL.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action_arr   = np.asarray(action, dtype=np.int64).flatten()
        padded_action = self._pad_action(action_arr)
        thrash_count  = 0

        # ── Apply phase actions ───────────────────────────────────────────
        for i in range(self.n_tl):
            desired = int(padded_action[i])
            current = int(self._current_phases[i])

            # Thrash guard: ignore switch if minimum green not yet elapsed
            if (
                desired != current
                and desired not in YELLOW_PHASES
                and self._phase_elapsed[i] < MIN_GREEN_S
            ):
                thrash_count += 1
                desired = current           # keep current phase

            # Auto-insert yellow transition between green phases
            if (
                current in (0, 2)
                and desired in (0, 2)
                and desired != current
            ):
                yellow = current + 1        # 0→1 (NS) or 2→3 (EW)
                self._set_phase(i, yellow)
                self._advance_sim(YELLOW_DURATION[yellow])

            self._set_phase(i, desired)
            self._phase_history.append((self._sim_time_s, i, desired))

        # ── Advance simulation ────────────────────────────────────────────
        self._advance_sim(self.cfg["decision_interval_s"])
        self._step_count += 1

        # ── Incident tick ─────────────────────────────────────────────────
        traci_ref = traci if not self.mock_sumo else None
        self._incident_mgr.tick(
            step=self._step_count,
            n_tl=self.n_tl,
            traci=traci_ref,
        )

        # ── Telemetry ─────────────────────────────────────────────────────
        tel = (
            mock_telemetry(self._rng, self.n_tl, self.cfg["probe_noise_sigma"])
            if self.mock_sumo
            else self._fetch_telemetry()
        )
        tel["thrash_count"] = thrash_count

        # Incident clearance check (updates incident_mgr internal flag)
        self._incident_mgr.check_clearance(tel["queue_lengths"])
        tel["incident_cleared"] = self._incident_mgr.incident_cleared

        self._last_telemetry = tel

        # ── Update phase elapsed ──────────────────────────────────────────
        for i in range(self.n_tl):
            if int(self._current_phases[i]) == int(padded_action[i]):
                self._phase_elapsed[i] += self.cfg["decision_interval_s"]
            else:
                self._phase_elapsed[i] = 0.0

        # ── Build obs ─────────────────────────────────────────────────────
        obs = build_obs(
            telemetry=tel,
            current_phases=self._current_phases,
            phase_elapsed=self._phase_elapsed,
            active_incidents=self._incident_mgr.active_incidents,
            step_count=self._step_count,
            max_steps=self.cfg["max_steps"],
            n_tl=self.n_tl,
        )

        # ── Reward ────────────────────────────────────────────────────────
        reward = compute_reward(
            telemetry=tel,
            cfg=self.cfg,
            step_count=self._step_count,
            n_tl=self.n_tl,
        )

        # ── Accumulate episode metrics ────────────────────────────────────
        self._episode_throughput += tel["throughput"]
        self._episode_delay_sum  += tel["total_waiting_time_s"]
        self._episode_stops      += int(tel["new_stops"].sum())
        if tel["overflow_lanes"] > 0:
            self._overflow_events += 1

        # ── Termination ───────────────────────────────────────────────────
        truncated  = self._step_count >= self.cfg["max_steps"]
        terminated = False          # no early-success termination

        info = self._build_info(reward, terminated, truncated, tel)
        if truncated or terminated:
            info["episode_summary"] = self._build_episode_summary()

        return obs, reward, terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        """
        Return a fully JSON-serialisable snapshot of current env state.
        Called by the OpenEnv grader after each episode.
        """
        return {
            "task_id":              self.task_id,
            "task_name":            self.cfg["name"],
            "step":                 int(self._step_count),
            "sim_time_s":           float(self._sim_time_s),
            "current_phases":       self._current_phases.tolist(),
            "phase_elapsed_s":      self._phase_elapsed.tolist(),
            "active_incidents":     self._incident_mgr.active_incidents,
            "episode_throughput":   self._episode_throughput.tolist(),
            "episode_delay_sum_s":  float(self._episode_delay_sum),
            "episode_stops":        int(self._episode_stops),
            "overflow_events":      int(self._overflow_events),
            "incident_cleared":     bool(self._incident_mgr.incident_cleared),
            "success_thresholds":   self.cfg["success_thresholds"],
        }

    def render(self, mode: str = "none") -> None:
        """SUMO-GUI handles rendering; this is a no-op for headless mode."""

    def close(self) -> None:
        self._close_sumo()

    # ── SUMO lifecycle ────────────────────────────────────────────────────────

    def _launch_sumo(self) -> None:
        """Start a SUMO subprocess and connect via TraCI."""
        binary = "sumo-gui" if self.use_gui else "sumo"

        if self.port == 0:
            import socket
            with socket.socket() as s:
                s.bind(("", 0))
                self.port = s.getsockname()[1]

        cmd = [
            binary,
            "--configuration-file", str(self.sumo_cfg),
            "--route-files",        str(self.cfg["route_file"]),
            "--remote-port",        str(self.port),
            "--no-step-log",        "true",
            "--no-warnings",        "true",
            "--collision.action",   "remove",
            "--seed",               str(int(self._rng.integers(0, 99_999))),
        ]
        logger.debug("SUMO cmd: %s", " ".join(cmd))

        self._sumo_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        traci.init(self.port)
        logger.debug("TraCI connected on port %d.", self.port)

    def _close_sumo(self) -> None:
        try:
            if TRACI_AVAILABLE:
                traci.close()
        except Exception:
            pass
        if self._sumo_process is not None:
            self._sumo_process.terminate()
            try:
                self._sumo_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._sumo_process.kill()
            self._sumo_process = None

    def _advance_sim(self, seconds: float) -> None:
        if self.mock_sumo:
            self._sim_time_s += seconds
            return
        for _ in range(max(1, int(seconds))):
            traci.simulationStep()
        self._sim_time_s += seconds

    def _set_phase(self, tl_index: int, phase: int) -> None:
        self._current_phases[tl_index] = phase
        if not self.mock_sumo:
            traci.trafficlight.setPhase(self.cfg["tl_ids"][tl_index], phase)

    # ── Telemetry via TraCI ───────────────────────────────────────────────────

    def _fetch_telemetry(self) -> Dict[str, Any]:
        """Collect metrics from live TraCI connection."""
        return collect_telemetry(
            traci=traci,
            cfg=self.cfg,
            n_tl=self.n_tl,
            rng=self._rng,
            phase_history=self._phase_history,
            active_incidents=self._incident_mgr.active_incidents,
            incident_cleared_flag=self._incident_mgr.incident_cleared,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pad_action(self, action: np.ndarray) -> np.ndarray:
        """Zero-pad a task-sized action to length 3 for internal loops."""
        padded = np.zeros(3, dtype=np.int64)
        padded[:len(action)] = action[:3]
        return padded

    def _build_info(
        self,
        reward:     float,
        terminated: bool,
        truncated:  bool,
        telemetry:  Optional[Dict] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "task_id":    self.task_id,
            "step":       int(self._step_count),
            "sim_time_s": float(self._sim_time_s),
            "reward":     float(reward),
            "terminated": terminated,
            "truncated":  truncated,
        }
        if telemetry:
            info.update({
                "queue_total":      float(telemetry["queue_lengths"][:self.n_tl].sum()),
                "throughput_total": float(telemetry["throughput"][:self.n_tl].sum()),
                "overflow_lanes":   int(telemetry["overflow_lanes"]),
                "long_wait_count":  int(telemetry["long_wait_count"]),
                "green_wave_hits":  int(telemetry["green_wave_hits"]),
                "incident_cleared": bool(telemetry["incident_cleared"]),
                "active_incidents": self._incident_mgr.active_incidents,
                "reward_breakdown": reward_breakdown(
                    telemetry, self.cfg, self._step_count, self.n_tl
                ),
            })
        return info

    def _build_episode_summary(self) -> Dict[str, Any]:
        """Compute end-of-episode metrics and check success thresholds."""
        total_steps   = max(1, self._step_count)
        avg_delay     = self._episode_delay_sum / total_steps
        total_through = float(self._episode_throughput[:self.n_tl].sum())

        # Fall back to heuristic baseline if set_baseline() was never called
        baseline_delay      = self._baseline_avg_delay   or max(1e-6, avg_delay * 1.30)
        baseline_throughput = self._baseline_throughput  or max(1e-6, total_through * 0.80)

        delay_reduction_pct = (
            100.0 * (baseline_delay - avg_delay) / baseline_delay
        )
        throughput_improvement_pct = (
            100.0 * (total_through - baseline_throughput) / baseline_throughput
        )

        t = self.cfg["success_thresholds"]
        passed = all([
            delay_reduction_pct      >= t.get("delay_reduction_pct",          0.0),
            throughput_improvement_pct >= t.get("throughput_improvement_pct", 0.0),
            self._overflow_events    <= t.get("overflow_events",              9999),
        ])

        summary = {
            "task_id":                     self.task_id,
            "total_steps":                 total_steps,
            "avg_delay_s":                 round(float(avg_delay),            4),
            "total_throughput":            round(float(total_through),         4),
            "delay_reduction_pct":         round(float(delay_reduction_pct),   4),
            "throughput_improvement_pct":  round(float(throughput_improvement_pct), 4),
            "overflow_events":             int(self._overflow_events),
            "success":                     bool(passed),
            "thresholds":                  t,
        }
        logger.info("Episode done | %s", json.dumps(
            {k: v for k, v in summary.items() if k != "thresholds"}
        ))
        return summary

    def set_baseline(self, avg_delay: float, throughput: float) -> None:
        """
        Inject fixed-time baseline metrics for episode_summary comparison.
        Called by inference.py after running a fixed-time reference episode.
        """
        self._baseline_avg_delay   = float(avg_delay)
        self._baseline_throughput  = float(throughput)

    def reward_breakdown_last(self) -> Dict[str, float]:
        """Return per-term reward breakdown for the most recent step."""
        if not self._last_telemetry:
            return {}
        return reward_breakdown(
            self._last_telemetry, self.cfg, self._step_count, self.n_tl
        )

    def __repr__(self) -> str:
        return (
            f"JaamCTRLTrafficEnv("
            f"task={self.task_id}, "
            f"n_tl={self.n_tl}, "
            f"step={self._step_count}/{self.cfg['max_steps']})"
        )