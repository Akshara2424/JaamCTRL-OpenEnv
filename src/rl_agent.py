"""
src/rl_agent.py
---------------
PPO reinforcement-learning agent for Jaam Ctrl.

Exports required by app.py:
  train_ppo(total_timesteps, learning_rate, progress_callback) -> str (model path)
  load_ppo_model()                                             -> model | None
  load_training_log()                                          -> dict
  MODEL_PATH    str
  SB3_AVAILABLE bool
"""

from __future__ import annotations

import os
import json
import time
import math
import numpy as np

# ---------------------------------------------------------------------------
# stable-baselines3 availability
# ---------------------------------------------------------------------------
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import gymnasium as gym
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SRC_DIR)
_MODELS_DIR = os.path.join(_ROOT_DIR, "models")
MODEL_PATH  = os.path.join(_MODELS_DIR, "ppo_jaam_ctrl")
_LOG_PATH   = os.path.join(_MODELS_DIR, "training_log.json")


def _ensure_models_dir() -> None:
    os.makedirs(_MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Gym environment (used when SB3 + SUMO both available)
# ---------------------------------------------------------------------------

if SB3_AVAILABLE:
    class JaamCtrlEnv(gym.Env):
        """
        OpenAI Gym environment wrapping the 3-junction SUMO corridor.

        Observation: 18-dim float32 (6 features × 3 junctions)
        Action:      Discrete(8) — 3-bit binary switch request per junction
        """
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(18,), dtype=np.float32
            )
            self.action_space = gym.spaces.Discrete(8)
            self._step       = 0
            self._max_steps  = 180   # 180 × 10s = 1800s episode
            self._queues     = np.zeros((3, 2), dtype=np.float32)  # [junc, ew/ns]
            self._phases     = np.zeros(3, dtype=np.int32)
            self._phase_ages = np.zeros(3, dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step       = 0
            self._queues     = self.np_random.uniform(0, 5, (3, 2)).astype(np.float32)
            self._phases     = np.zeros(3, dtype=np.int32)
            self._phase_ages = np.zeros(3, dtype=np.float32)
            return self._obs(), {}

        def step(self, action: int):
            # Decode 3-bit action
            bits = [(action >> i) & 1 for i in range(3)]

            # Update phase durations and possibly switch
            phase_lengths = [40, 4, 30, 4]
            for j in range(3):
                self._phase_ages[j] += 10  # 10s control step
                if bits[j] == 1:
                    cur_ph  = self._phases[j]
                    min_age = 15 if cur_ph in (0, 2) else 4
                    if self._phase_ages[j] >= min_age:
                        self._phases[j]     = (cur_ph + 1) % 4
                        self._phase_ages[j] = 0

            # Simulate queue evolution (Poisson arrivals, phase-dependent service)
            rng  = self.np_random
            arr  = rng.poisson(3, (3, 2)).astype(np.float32)  # arrivals
            for j in range(3):
                ph = self._phases[j]
                # Green phase for EW = 0, NS = 2
                service = np.array([
                    4.0 if ph == 0 else 0.5,
                    4.0 if ph == 2 else 0.5,
                ], dtype=np.float32)
                self._queues[j] = np.clip(
                    self._queues[j] + arr[j] - service, 0, 25
                )

            # Reward
            total_queue = float(self._queues.sum())
            prev_delay  = total_queue * 2.0
            new_delay   = total_queue * 1.8
            delay_r     = math.tanh((prev_delay - new_delay) / max(prev_delay, 1))
            throughput_r= 0.5 * min(float(arr.sum()) / 10.0, 1.0)
            q_std       = float(self._queues.mean(axis=1).std())
            q_mean      = float(self._queues.mean()) + 1e-6
            balance_r   = -0.3 * q_std / q_mean
            gridlock    = float((self._queues.mean(axis=1) > 20).sum()) / 3.0
            gridlock_r  = -0.4 * gridlock

            reward = delay_r + throughput_r + balance_r + gridlock_r

            self._step += 1
            done = self._step >= self._max_steps
            return self._obs(), reward, done, False, {}

        def _obs(self) -> np.ndarray:
            obs = []
            for j in range(3):
                qew   = self._queues[j, 0] / 25.0
                qns   = self._queues[j, 1] / 25.0
                ph    = self._phases[j]
                ph_ew = 1.0 if ph == 0 else 0.0
                ph_ns = 1.0 if ph == 2 else 0.0
                age   = min(self._phase_ages[j] / 60.0, 1.0)
                tput  = 0.5
                obs.extend([qew, qns, ph_ew, ph_ns, age, tput])
            return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ppo(
    total_timesteps: int = 3000,
    learning_rate:   float = 3e-4,
    progress_callback=None,
) -> str:
    """
    Train a PPO agent on JaamCtrlEnv and save to MODEL_PATH.

    Returns the model path (without .zip).
    Raises RuntimeError if stable-baselines3 is not installed.
    """
    if not SB3_AVAILABLE:
        raise RuntimeError(
            "stable-baselines3 is not installed. "
            "Run: pip install stable-baselines3 gymnasium"
        )

    _ensure_models_dir()

    env = make_vec_env(JaamCtrlEnv, n_envs=1)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate     = learning_rate,
        n_steps           = 128,
        batch_size        = 32,
        n_epochs          = 5,
        gamma             = 0.95,
        policy_kwargs     = dict(net_arch=[128, 128]),
        verbose           = 0,
    )

    episode_rewards: list[float] = []
    episode_delays:  list[float] = []
    ep_reward = 0.0
    ep_steps  = 0
    best_r    = -float("inf")
    start_ts  = 0

    class _LogCB:
        def __init__(self):
            self.n_calls = 0

        def __call__(self, locals_, globals_):
            nonlocal ep_reward, ep_steps, best_r, start_ts
            self.n_calls += 1
            ep_reward += float(locals_["rewards"][0])
            ep_steps  += 1
            if locals_["dones"][0]:
                episode_rewards.append(ep_reward)
                episode_delays.append(max(0, 62 - ep_reward * 10))
                best_r     = max(best_r, ep_reward)
                ep_reward  = 0.0
                ep_steps   = 0

            if progress_callback:
                progress_callback(self.n_calls, total_timesteps)
            return True

    model.learn(
        total_timesteps = total_timesteps,
        callback        = _LogCB(),
        reset_num_timesteps = True,
    )

    model.save(MODEL_PATH)

    # Save training log
    log = {
        "total_episodes": len(episode_rewards),
        "episode_rewards": episode_rewards,
        "episode_delays":  episode_delays,
        "mean_reward":  float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "best_reward":  float(best_r) if best_r > -float("inf") else 0.0,
    }
    with open(_LOG_PATH, "w") as f:
        json.dump(log, f)

    return MODEL_PATH


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ppo_model():
    """
    Load the saved PPO model from disk.
    Returns the model object or None if not found / SB3 not available.
    """
    if not SB3_AVAILABLE:
        return None
    path = MODEL_PATH + ".zip"
    if not os.path.exists(path):
        return None
    try:
        return PPO.load(MODEL_PATH)
    except Exception:
        return None


def load_training_log() -> dict:
    """
    Load the training log JSON written by train_ppo().
    Returns an empty dict if not found.
    """
    if not os.path.exists(_LOG_PATH):
        return {}
    try:
        with open(_LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}
