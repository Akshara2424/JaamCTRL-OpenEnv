# JaamCTRL — जाम Ctrl
## AI Adaptive Traffic Signal Control for Indian Urban Roads

> **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology 2026**  
> **Open Innovation Track**  
> **OpenEnv-Compliant Gymnasium Environment**

---

## Executive Summary

JaamCTRL is a **production-ready RL environment** for adaptive traffic signal control trained on realistic Indian urban traffic. It implements the full **OpenEnv specification** with:

- ✅ Typed Pydantic models (Observation, Action, Reward)
- ✅ Deterministic task graders with clear success criteria
- ✅ Baseline inference script using OpenAI-compatible API
- ✅ Full Docker containerization for HF Spaces
- ✅ Comprehensive documentation and baseline metrics

---

## The Problem

**Indian cities lose $22 billion annually to traffic congestion.** Delhi alone wastes **1.5 billion person-hours** yearly sitting at red lights.

**The cause:** Fixed-time signals (30s green / 30s red) everywhere, regardless of actual demand. No coordination between intersections means a platoon cleared at one signal hits red at the next.

**The solution:** JaamCTRL learns to coordinate three signals on a real Delhi corridor using reinforcement learning.

---

## What We're Solving

**Corridor:** Barakhamba Rd → CP Core → Patel Chowk, New Delhi (600 meters)

```
[INT-1: Barakhamba] --300m-- [INT-2: CP Core] --300m-- [INT-3: Patel Chowk]
```

**Simulation fidelity:**
- SUMO (Simulation of Urban MObility) with realistic Delhi traffic
- Vehicle mix: 55% motorcycles, 25% cars, 12% autos (tuk-tuks), 8% buses
- Dynamic incidents: accidents, animal crossings, pedestrian waves
- Peak demand: 900 vehicles/hour across corridor
- GPS probe noise and realistic sensor failures

**Baseline:** Fixed-time 30s/30s signal (standard deployment everywhere)

**Target:** PPO agent learning to reduce delay by 25–47% while maintaining zero lane overflow.

---

## Three Progressive Difficulty Levels

All tasks use **identical 46-dimensional observation vector** (zero-padded) and **same PPO architecture**, enabling fair skill evaluation across difficulty levels.

### Task 1 — Easy (Gymnasium ID: `JaamCTRL-Easy-v0`)
**Single intersection, cars only, no incidents**

| Metric | Value |
|--------|-------|
| **Intersections** | 1 (Barakhamba only) |
| **Traffic mix** | 100% cars (homogeneous) |
| **Demand** | 400 vehicles/hour |
| **Incidents** | None |
| **Episode length** | 300 steps (25 minutes simulation) |
| **Action space** | `MultiDiscrete([4])` — one 4-phase decision |
| **Observation** | 46-dim vector (7 lanes zero-padded) |
| **Success threshold** | ≥15% delay reduction vs fixed-time |

### Task 2 — Medium (Gymnasium ID: `JaamCTRL-Medium-v0`)
**Two intersections, realistic traffic mix, pedestrian incidents**

| Metric | Value |
|--------|-------|
| **Intersections** | 2 (Barakhamba + CP Core) |
| **Traffic mix** | 20% cars, 60% motorcycles, 20% autos |
| **Demand** | 600 vehicles/hour |
| **Incidents** | Stochastic pedestrian crossings (Poisson λ=0.02/step) |
| **Episode length** | 500 steps (42 minutes simulation) |
| **Action space** | `MultiDiscrete([4, 4])` — two coordinated decisions |
| **Observation** | 46-dim vector (same structure) |
| **Success threshold** | ≥20% delay reduction + ≥15% throughput gain |

### Task 3 — Hard (Gymnasium ID: `JaamCTRL-Hard-v0`)
**Full corridor, chaotic traffic, multiple incident types**

| Metric | Value |
|--------|-------|
| **Intersections** | 3 (full corridor) |
| **Traffic mix** | 25% cars, 55% motorcycles, 12% autos, 8% buses |
| **Demand** | 900 vehicles/hour + rush-hour spike |
| **Incidents** | Accident (50 steps @ INT-2) + animal crossing (10 steps @ INT-3) |
| **Probe noise** | σ=0.05 Gaussian on GPS heatmap |
| **Episode length** | 800 steps (67 minutes simulation) |
| **Action space** | `MultiDiscrete([4, 4, 4])` — three coordinated decisions |
| **Observation** | 46-dim vector (same structure) |
| **Success threshold** | ≥25% delay reduction + ≥20% throughput + zero overflow |

---

## Observation Space

**Type:** Gymnasium Dict space with flat 46-dimensional float32 vector

**Structure (always 46 floats, zero-padded to 3 intersections):**

```
[0:12]   Queue lengths          3 intersections × 4 lanes
[12:15]  Current phase          phase ID (0-3) per TL  
[15:18]  Phase elapsed (sec)    time in current phase per TL
[18:42]  Probe density          3 TL × 8 spatial bins (GPS heatmap)
[42:45]  Incident flags         1.0 if active, 0.0 otherwise
[45:46]  Time of day (norm)     0.0-1.0 (24-hour cycle)
```

### Example Observation
```json
{
  "queue_lengths": [[10, 8, 5, 3], [0, 0, 0, 0], [0, 0, 0, 0]],
  "current_phase": [0, 2, 0],
  "phase_elapsed": [7.5, 3.2, 0.0],
  "probe_density": [
    [0.5, 0.6, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ],
  "incident_active": [false, false, false],
  "time_of_day_norm": 0.5
}
```

**Note:** Inactive intersections' slots are always zero. Task 1 always has zeros in positions [12:24] (INT-2 and INT-3).

---

## Action Space

**Type:** Gymnasium MultiDiscrete, variable size per task

**Encoding:**  
- `0` = NS green / EW red (north-south primary)
- `1` = NS yellow (3 sec transition)
- `2` = EW green / NS red (east-west primary)
- `3` = EW yellow (3 sec transition)

### Examples

**Task 1:** `[2]` → phase 2 (EW green) at INT-1

**Task 2:** `[0, 2]` → NS green at INT-1, EW green at INT-2 (green-wave setup)

**Task 3:** `[1, 0, 2]` → NS yellow @ INT-1, NS green @ INT-2, EW green @ INT-3

---

## Reward Function

**Design:** Reward partial progress toward task completion, penalizing clearly undesirable behavior.

**Components (all normalized per active intersection):**

| Component | Equation | Type | Tasks |
|-----------|----------|------|-------|
| **Queue penalty** | `-0.01 × sum(queue_lengths)` | Ongoing | All |
| **Throughput bonus** | `+0.05 × vehicles_exiting` | Ongoing | All |
| **Stop penalty** | `-0.02 × new_stop_events` | Ongoing | All |
| **Long-wait hard** | `-0.50 × vehicles_waiting_>60s` | Ongoing | T2, T3 |
| **Green-wave bonus** | `+0.10 × platoons_found_green` | Ongoing | T2, T3 |
| **Overflow penalty** | `-3.00 × overflow_lane_events` | Ongoing | T3 |
| **Incident bonus** | `+2.00` (one-time) | Once/incident | T3 |
| **Thrash penalty** | `-0.05 × rapid_phase_switches` | Ongoing | All |

**Clipping:** Final reward clipped to `[-10.0, +5.0]` per step to prevent gradient explosion.

**Example episode reward:** ~100–150 cumulative over 300–800 steps

---

## Task Grading Criteria

All tasks are graded deterministically against a **fixed-time 30s/30s baseline**.

### Success Metrics

**Metric 1: Delay Reduction**
- Computes average vehicle delay at all intersections
- Compares to fixed-time baseline  
- Score: `% reduction / (threshold + 10%)`

**Metric 2: Throughput Improvement**
- Counts vehicles exiting network per hour
- Relative to fixed-time baseline
- Score: `% improvement / (threshold + 10%)`

**Metric 3: Lane Overflow Events**
- Hard constraint: must be ≤ max allowed
- Lane overflow = queue depth > 20 vehicles
- Failure if even one overflow occurs (Task 3)

### Passing Criteria by Task

| Task | Delay Reduction | Throughput Gain | Max Overflow | Score Formula |
|------|-----------------|-----------------|--------------|---------------|
| **Easy** | ≥15% | ≥10% | 0 | (d_score + t_score + o_score) / 3 |
| **Medium** | ≥20% | ≥15% | 0 | (d_score + t_score + o_score) / 3 |
| **Hard** | ≥25% | ≥20% | 0 | (d_score + t_score + o_score) / 3 |

**Output format:**
```json
{
  "task_id": 3,
  "task_name": "Task 3 (Hard)",
  "passed": true,
  "score": 0.92,
  "details": {
    "delay_reduction": {
      "achieved": 28.5,
      "required": 25.0,
      "passed": true
    },
    "throughput_improvement": {
      "achieved": 22.3,
      "required": 20.0,
      "passed": true
    },
    "overflow_events": {
      "achieved": 0,
      "max_allowed": 0,
      "passed": true
    }
  },
  "message": "Task 3 (Hard): delay=28.5% (need 25%), throughput=22.3% (need 20%), overflow=0 events (max 0). Score: 0.92 — PASS ✓"
}
```

---

## Baseline Performance

### Fixed-Time Signal (30s/30s, deployed everywhere)
- Avg delay: **~45 seconds**
- Throughput: **400 veh/hr** (baseline)
- Lane overflow: **~2–3 events per episode**
- Delay reduction: **0%** (by definition)

### Rule-Based Heuristic
- Avg delay: **~36 seconds**
- Throughput: **+12%** improvement
- Delay reduction: **~20%**
- Episode can fail if incidents occur

### PPO Agent Results (Mock-Mode Baseline)

| Task | Avg Delay (s) | Throughput Gain | Delay Reduction | Overflow Events | Score |
|------|---------------|-----------------|-----------------|-----------------|-------|
| **Easy** | 32 | +18% | 29% | 0 | 0.95 |
| **Medium** | 28 | +25% | 38% | 0 | 0.91 |
| **Hard** | 24 | +34% | 47% | 0 | 0.87 |

**Note:** All results from mock-mode (synthetic observations). SUMO-based results vary by `seed` and network conditions. These baselines represent ~50K training steps per task.

---

## Setup & Installation

### Local Development

**Prerequisites:**
- Python 3.11+
- SUMO 1.14+ (optional; mock-mode works without it)
- pip

**Install:**

```bash
# Clone repo
git clone https://github.com/Akshara2424/JaamCTRL-OpenEnv.git
cd JaamCTRL-OpenEnv

# Create venv
python3.11 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Docker (HF Spaces)

Pre-configured to run on HF Spaces:

```bash
docker build -t jaamctrl .
docker run -p 7860:7860 \
  -e MOCK_SUMO=1 \
  -e OPENAI_API_KEY="sk-..." \
  jaamctrl
```

---

## Usage

### Run All 3 Tasks (Inference)

```bash
export O PENAI_API_KEY="sk-..."
python inference.py
```

Output follows OpenEnv grader format:
```
[START] {"event": "START", "env": "jaamctrl-traffic", "task_id": 1, ...}
[STEP]  {"event": "STEP", "step": 1, "action": [2], "reward": 0.35, ...}
[END]   {"event": "END", "task_id": 1, "success": true, "delay_reduction": 27.4, ...}
```

### Mock Mode (No SUMO)

```bash
python inference.py --mock
```

### Gymnasium API

```python
import gymnasium
from env import register_envs

register_envs()

# Easy task
env = gymnasium.make("JaamCTRL-Easy-v0", mock_sumo=True)
obs, info = env.reset(seed=42)

for step in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

Visualizes:
- Live signal phase animations
- Queue history and throughput graphs
- Incident detection & GPS heatmaps
- Reward breakdown per step

---

## File Structure

```
JaamCTRL-OpenEnv/
├── app.py                      # Streamlit dashboard
├── inference.py                # OpenEnv grader format inference
├── requirements.txt
├── Dockerfile
├── openenv.yaml                # OpenEnv spec metadata
│
├── env/
│   ├── __init__.py             # Package exports + TASK_CONFIGS
│   ├── base_env.py             # Core Gymnasium environment
│   ├── observation.py          # Observation construction
│   ├── reward.py               # Reward computation
│   ├── incident_manager.py     # Incident scheduling
│   ├── models.py               # Pydantic models (OpenEnv)
│   ├── graders.py              # Task-specific graders
│   ├── task_easy.py            # Task 1 constants
│   ├── task_medium.py          # Task 2 constants
│   └── task_hard.py            # Task 3 constants
│
├── sumo/
│   ├── corridor.net.xml        # Network definition
│   ├── config.sumocfg          # SUMO configuration
│   ├── easy.rou.xml            # Task 1 routes
│   ├── medium.rou.xml          # Task 2 routes
│   └── hard.rou.xml            # Task 3 routes
│
├── models/
│   ├── ppo_jaamctrl.zip        # Trained PPO model (SB3)
│   └── training_log.json       # Training metrics
│
└── assets/
    ├── header.png
    ├── logo.jpeg
    └── poster.png
```

---

## Real-World Impact

**If deployed on Delhi's 200 key arterial corridors:**
- 14 million daily commuters would benefit
- **~63 million person-hours saved annually**
- **~8% fuel savings** (less stop-and-go)
- **~12% reduction in tailpipe emissions**

---

## OpenEnv Compliance Checklist

- ✅ **Real-world task simulation:** Adaptive traffic signal control
- ✅ **Typed models:** Pydantic `Observation`, `Action`, `Reward` classes
- ✅ **Three progressive tasks:** Easy (1 TL) → Medium (2 TL) → Hard (3 TL)  
- ✅ **Deterministic graders:** `grade_task(task_id, metrics)` with clear thresholds
- ✅ **Meaningful rewards:** 8 terms, partial progress, trajectory-based signal
- ✅ **Baseline inference:** `inference.py` with OpenAI API integration
- ✅ **Dockerfile:** Runs on HF Spaces, includes SUMO + Python 3.11
- ✅ **Documentation:** Complete API, task specs, setup, baselines

---

## License

MIT

---

## Citation

```bibtex
@software{jaamctrl_2026,
  title = {JaamCTRL: AI Adaptive Traffic Signal Control},
  author = {JaamCTRL Team},
  year = {2026},
  howpublished = {\url{https://github.com/Akshara2424/JaamCTRL-OpenEnv}},
  note = {OpenEnv-Compliant Gymnasium Environment}
}
```
