"""
inference.py
───────────────────────────────────────────────────────────────────────────────
JaamCTRL — OpenEnv Hackathon Submission Inference Script
Meta PyTorch OpenEnv Hackathon x Scaler School of Technology 2026

CRITICAL FORMAT RULES (from Scaler dashboard spec):
  - stdout logs MUST follow [START] / [STEP] / [END] format EXACTLY.
  - Field names, ordering, and JSON structure must not deviate.
  - This file must be named inference.py and live in the project root.
  - Uses OpenAI-compatible client for any LLM calls (env var driven).

Usage
-----
  # Run all 3 tasks with PPO agent (requires trained models)
  python inference.py

  # Run a specific task only
  python inference.py --task 1
  python inference.py --task 2
  python inference.py --task 3

  # Run in mock mode (no SUMO required — for CI / judging sandbox)
  python inference.py --mock

  # Run with rule-based fallback instead of PPO
  python inference.py --agent rule_based

Environment variables
---------------------
  MODEL_PATH      path to PPO model zip (default: agents/models/ppo_jaamctrl.zip)
  MOCK_SUMO       set to "1" to force mock mode
  API_BASE_URL    LLM API endpoint (OpenAI-compatible, unused in RL inference)
  MODEL_NAME      LLM model name (unused in RL inference)
  HF_TOKEN        Hugging Face token (unused in RL inference)
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ── Env import ────────────────────────────────────────────────────────────────
from env import JaamCTRLTrafficEnv, TASK_CONFIGS, register_envs
import gymnasium as gym
from gymnasium import spaces

register_envs()

# ── Optional SB3 PPO import ───────────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ── Constants matching the grader's expected field names ─────────────────────
LOG_VERSION     = "1.0"
ENV_NAME        = "jaamctrl-traffic"
TEAM_NAME       = os.getenv("TEAM_NAME", "JaamCTRL")

# Per-task model paths — fall back to shared model if task-specific not found
MODEL_PATHS = {
    1: os.getenv("MODEL_PATH_1", str(ROOT / "models/ppo_task1.zip")),
    2: os.getenv("MODEL_PATH_2", str(ROOT / "models/ppo_task2.zip")),
    3: os.getenv("MODEL_PATH_3", str(ROOT / "models/ppo_jaamctrl.zip")),
}
SHARED_MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models/ppo_jaamctrl.zip"))

MOCK_SUMO  = os.getenv("MOCK_SUMO", "0") == "1"
EPISODES_PER_TASK = 1          # grader runs 1 episode per task


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Stdout logging — EXACT format required by Scaler grader               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def log_start(task_id: int, episode: int, agent_type: str) -> None:
    """
    Emit [START] line.  Must be the very first output for each episode.
    The grader parses this to initialise its episode tracking.
    """
    payload = {
        "event":       "START",
        "env":         ENV_NAME,
        "task_id":     task_id,
        "episode":     episode,
        "agent":       agent_type,
        "max_steps":   TASK_CONFIGS[task_id]["max_steps"],
        "n_tl":        TASK_CONFIGS[task_id]["active_intersections"],
        "timestamp":   time.time(),
    }
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(
    task_id:   int,
    episode:   int,
    step:      int,
    action:    List[int],
    reward:    float,
    info:      Dict[str, Any],
    truncated: bool,
    terminated: bool,
) -> None:
    """
    Emit [STEP] line for every environment step.
    Field names MUST match exactly — the grader reads these to build
    per-episode reward curves and metric timeseries.
    """
    payload = {
        "event":           "STEP",
        "task_id":         task_id,
        "episode":         episode,
        "step":            step,
        "action":          action,
        "reward":          round(float(reward), 6),
        "queue_total":     round(float(info.get("queue_total",      0.0)), 4),
        "throughput":      round(float(info.get("throughput_total", 0.0)), 4),
        "overflow_lanes":  int(info.get("overflow_lanes",  0)),
        "long_wait_count": int(info.get("long_wait_count", 0)),
        "green_wave_hits": int(info.get("green_wave_hits", 0)),
        "truncated":       bool(truncated),
        "terminated":      bool(terminated),
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(
    task_id:  int,
    episode:  int,
    summary:  Dict[str, Any],
    success:  bool,
    duration_s: float,
) -> None:
    """
    Emit [END] line.  Must be the final output for each episode.
    The grader reads `success`, `delay_reduction_pct`, and
    `throughput_improvement_pct` to score the submission.
    """
    payload = {
        "event":                        "END",
        "task_id":                      task_id,
        "episode":                      episode,
        "success":                      bool(success),
        "total_steps":                  int(summary.get("total_steps",                0)),
        "avg_delay_s":                  round(float(summary.get("avg_delay_s",        0.0)), 4),
        "total_throughput":             round(float(summary.get("total_throughput",   0.0)), 4),
        "delay_reduction_pct":          round(float(summary.get("delay_reduction_pct",         0.0)), 4),
        "throughput_improvement_pct":   round(float(summary.get("throughput_improvement_pct",  0.0)), 4),
        "overflow_events":              int(summary.get("overflow_events", 0)),
        "episode_duration_s":           round(float(duration_s), 3),
        "thresholds":                   summary.get("thresholds", {}),
    }
    print(f"[END] {json.dumps(payload)}", flush=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Flat observation wrapper (SB3 needs Box, not Dict)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class FlatObsWrapper:
    """
    Thin shim that presents env.observation_space["flat"] to SB3
    and unwraps Dict observations to the flat numpy vector.
    Not a full gymnasium.Wrapper to avoid any gymnasium version conflicts.
    
    NOTE: The trained PPO model expects 18-dim observations, but the environment
    produces 46-dim. This wrapper extracts the first 18 dims:
      [0:12]   queue_lengths (3×4)
      [12:15]  current_phase (3,)
      [15:18]  phase_elapsed (3,)
    """

    def __init__(self, env: JaamCTRLTrafficEnv) -> None:
        self._env = env
        self.action_space      = env.action_space
        # Override observation space to be 18-dim instead of 46-dim
        flat_space = env.observation_space["flat"]
        self.observation_space = spaces.Box(
            low=flat_space.low[:18],
            high=flat_space.high[:18],
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return obs["flat"][:18], info

    def step(self, action):
        obs, r, term, trunc, info = self._env.step(action)
        return obs["flat"][:18], r, term, trunc, info

    def close(self):
        self._env.close()

    # Forward state() so callers can still call it on the wrapper
    def state(self):
        return self._env.state()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Agent loaders                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_ppo_agent(task_id: int) -> Optional[Any]:
    """
    Load the saved PPO model for a given task.
    Tries task-specific path first, then shared path, then returns None.
    """
    if not SB3_AVAILABLE:
        print(
            f"[WARN] stable-baselines3 not installed. "
            f"Falling back to rule-based agent for task {task_id}.",
            file=sys.stderr,
            flush=True,
        )
        return None

    candidates = [MODEL_PATHS[task_id], SHARED_MODEL_PATH]
    for path in candidates:
        if Path(path).exists():
            try:
                model = PPO.load(path)
                print(
                    f"[INFO] Loaded PPO model for task {task_id}: {path}",
                    file=sys.stderr, flush=True,
                )
                return model
            except Exception as exc:
                print(
                    f"[WARN] Failed to load {path}: {exc}",
                    file=sys.stderr, flush=True,
                )

    print(
        f"[WARN] No PPO model found for task {task_id}. "
        f"Searched: {candidates}. Using rule-based fallback.",
        file=sys.stderr, flush=True,
    )
    return None


def rule_based_action(obs_flat: np.ndarray, n_tl: int) -> np.ndarray:
    """
    Simple heuristic controller: for each active TL, extend green if
    the corresponding approach queue length exceeds a threshold,
    otherwise switch to the other green phase.

    obs_flat layout (from observation.py):
      [0:12]  queue_lengths (3×4) — we use [0:4] per TL
      [12:15] current_phase (3,)
      [15:18] phase_elapsed (3,)
    """
    # Parse relevant slices from the flat vector
    queue_lengths  = obs_flat[0:12].reshape(3, 4)    # (3,4)
    current_phases = obs_flat[12:15].astype(int)     # (3,)
    phase_elapsed  = obs_flat[15:18]                 # (3,)

    QUEUE_THRESHOLD = 8     # vehicles; extend green if above
    MIN_GREEN       = 10    # seconds; don't switch before this

    actions = np.zeros(n_tl, dtype=np.int64)

    for i in range(n_tl):
        cp = int(current_phases[i])
        elapsed = float(phase_elapsed[i])

        # Never override a yellow phase — let it finish
        if cp in (1, 3):
            actions[i] = cp
            continue

        # Current green direction queue sum
        if cp == 0:   # NS green — lanes 0,1
            current_q  = float(queue_lengths[i, 0] + queue_lengths[i, 1])
            opposite_q = float(queue_lengths[i, 2] + queue_lengths[i, 3])
        else:          # EW green — lanes 2,3
            current_q  = float(queue_lengths[i, 2] + queue_lengths[i, 3])
            opposite_q = float(queue_lengths[i, 0] + queue_lengths[i, 1])

        # Logic: extend green if current queue still large and not too long;
        #        switch if opposite queue is worse and minimum green elapsed.
        if elapsed < MIN_GREEN:
            actions[i] = cp   # always respect minimum green
        elif opposite_q > current_q + QUEUE_THRESHOLD:
            actions[i] = 2 if cp == 0 else 0   # switch direction
        else:
            actions[i] = cp   # extend current green

    return actions


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Baseline runner (fixed-time controller)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_fixed_time_baseline(
    task_id:    int,
    mock_sumo:  bool,
    green_s:    int = 30,
) -> Dict[str, float]:
    """
    Run one episode with a naive fixed-time controller (alternating 30 s green)
    and return {avg_delay, throughput} for comparison.
    Used to set env.set_baseline() before the PPO/rule-based episode.
    """
    env = JaamCTRLTrafficEnv(
        task_id   = task_id,
        mock_sumo = mock_sumo,
        seed      = 999,
    )
    wrapper = FlatObsWrapper(env)
    obs, _ = wrapper.reset()

    cfg        = TASK_CONFIGS[task_id]
    n_tl       = cfg["active_intersections"]
    interval   = cfg["decision_interval_s"]    # seconds per step
    steps_per_cycle = (green_s * 2) // interval  # full NS+EW cycle

    total_delay    = 0.0
    total_through  = 0.0
    step           = 0

    while True:
        # Cycle: NS green for green_s, EW green for green_s
        cycle_pos   = step % steps_per_cycle
        phase_index = 0 if cycle_pos < (steps_per_cycle // 2) else 2
        action      = np.full(n_tl, phase_index, dtype=np.int64)

        obs, r, term, trunc, info = wrapper.step(action)
        total_delay   += info.get("queue_total", 0.0) * interval
        total_through += info.get("throughput_total", 0.0)
        step          += 1

        if trunc or term:
            break

    wrapper.close()
    avg_delay = total_delay / max(1, step)
    return {"avg_delay": avg_delay, "throughput": total_through}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Core episode runner                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_episode(
    task_id:    int,
    episode:    int,
    agent_type: str,
    mock_sumo:  bool,
    ppo_model:  Optional[Any],
    baseline:   Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Run a single episode, emitting [START] / [STEP]... / [END] to stdout.

    Returns the episode summary dict (same as info["episode_summary"]).
    """
    env = JaamCTRLTrafficEnv(
        task_id   = task_id,
        mock_sumo = mock_sumo,
        seed      = episode,   # different seed per episode
    )
    wrapper = FlatObsWrapper(env)
    n_tl    = TASK_CONFIGS[task_id]["active_intersections"]

    # Inject baseline so episode_summary computes % improvement correctly
    if baseline:
        env.set_baseline(
            avg_delay  = baseline["avg_delay"],
            throughput = baseline["throughput"],
        )

    obs_flat, info = wrapper.reset()

    log_start(task_id=task_id, episode=episode, agent_type=agent_type)

    step           = 0
    total_reward   = 0.0
    episode_start  = time.time()
    summary        = {}

    while True:
        # ── Select action ─────────────────────────────────────────────────
        if ppo_model is not None and agent_type == "ppo":
            action, _ = ppo_model.predict(obs_flat, deterministic=True)
        else:
            action = rule_based_action(obs_flat, n_tl)

        # ── Step ──────────────────────────────────────────────────────────
        obs_flat, reward, terminated, truncated, info = wrapper.step(action)

        step         += 1
        total_reward += reward

        log_step(
            task_id    = task_id,
            episode    = episode,
            step       = step,
            action     = action.tolist(),
            reward     = reward,
            info       = info,
            truncated  = truncated,
            terminated = terminated,
        )

        if truncated or terminated:
            summary = info.get("episode_summary", {})
            break

    duration_s = time.time() - episode_start

    # Patch summary with total reward (not in base episode_summary)
    summary["total_reward"] = round(float(total_reward), 4)

    log_end(
        task_id    = task_id,
        episode    = episode,
        summary    = summary,
        success    = bool(summary.get("success", False)),
        duration_s = duration_s,
    )

    wrapper.close()
    return summary


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Results file writer                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def write_results_json(all_results: Dict[int, List[Dict]], output_dir: Path) -> None:
    """
    Write a machine-readable results JSON that the grader can also consume
    if it expects a file in addition to stdout.

    File: logs/results.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"

    serialisable = {}
    for task_id, episodes in all_results.items():
        serialisable[str(task_id)] = episodes

    with open(out_path, "w") as f:
        json.dump(
            {
                "version":    LOG_VERSION,
                "env":        ENV_NAME,
                "team":       TEAM_NAME,
                "results":    serialisable,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Results written to {out_path}", file=sys.stderr, flush=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main entry point                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="JaamCTRL OpenEnv Inference Script",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific task only (default: run all 3)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["ppo", "rule_based"],
        default="ppo",
        help="Agent type to use (default: ppo, falls back to rule_based if model missing)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=MOCK_SUMO,
        help="Run in mock mode (no SUMO required)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES_PER_TASK,
        help="Number of episodes to run per task (default: 1)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        default=False,
        help="Skip baseline run (faster; improvement metrics will use heuristic baseline)",
    )
    args = parser.parse_args()

    tasks      = [args.task] if args.task else [1, 2, 3]
    mock_sumo  = args.mock
    agent_type = args.agent

    print(
        f"[INFO] JaamCTRL inference | tasks={tasks} | agent={agent_type} | "
        f"mock={mock_sumo} | episodes={args.episodes}",
        file=sys.stderr, flush=True,
    )

    all_results: Dict[int, List[Dict]] = {}

    for task_id in tasks:
        task_name = TASK_CONFIGS[task_id]["name"]
        print(
            f"[INFO] ── Task {task_id}: {task_name} ──",
            file=sys.stderr, flush=True,
        )

        # ── Load PPO model ────────────────────────────────────────────────
        ppo_model = None
        if agent_type == "ppo":
            ppo_model = load_ppo_agent(task_id)
        effective_agent = "ppo" if ppo_model is not None else "rule_based"

        # ── Run fixed-time baseline ───────────────────────────────────────
        baseline = None
        if not args.no_baseline:
            print(
                f"[INFO] Running fixed-time baseline for task {task_id}...",
                file=sys.stderr, flush=True,
            )
            baseline = run_fixed_time_baseline(
                task_id   = task_id,
                mock_sumo = mock_sumo,
            )
            print(
                f"[INFO] Baseline | avg_delay={baseline['avg_delay']:.2f}s | "
                f"throughput={baseline['throughput']:.1f}",
                file=sys.stderr, flush=True,
            )

        # ── Run episodes ──────────────────────────────────────────────────
        task_results = []
        for ep in range(1, args.episodes + 1):
            summary = run_episode(
                task_id    = task_id,
                episode    = ep,
                agent_type = effective_agent,
                mock_sumo  = mock_sumo,
                ppo_model  = ppo_model,
                baseline   = baseline,
            )
            task_results.append(summary)

        all_results[task_id] = task_results

        # ── Per-task aggregate summary to stderr (not parsed by grader) ───
        if task_results:
            avg_delay_pct  = np.mean([r.get("delay_reduction_pct",         0) for r in task_results])
            avg_thru_pct   = np.mean([r.get("throughput_improvement_pct",  0) for r in task_results])
            success_rate   = np.mean([float(r.get("success", False))           for r in task_results])
            print(
                f"[INFO] Task {task_id} aggregate | "
                f"delay_reduction={avg_delay_pct:.1f}% | "
                f"throughput_improvement={avg_thru_pct:.1f}% | "
                f"success_rate={success_rate*100:.0f}%",
                file=sys.stderr, flush=True,
            )

    # ── Write results JSON ────────────────────────────────────────────────
    write_results_json(all_results, output_dir=ROOT / "logs")

    # ── Final overall summary to stderr ──────────────────────────────────
    all_summaries = [s for eplist in all_results.values() for s in eplist]
    if all_summaries:
        overall_success = np.mean([float(s.get("success", False)) for s in all_summaries])
        print(
            f"[INFO] Overall success rate: {overall_success*100:.0f}% "
            f"({sum(s.get('success',False) for s in all_summaries)}/{len(all_summaries)} episodes)",
            file=sys.stderr, flush=True,
        )


if __name__ == "__main__":
    main()