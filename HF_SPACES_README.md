# JaamCTRL — AI Adaptive Traffic Signal Control

![license badge](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.11-blue)
![streamlit](https://img.shields.io/badge/streamlit-1.35.0-red)

> **AI-Powered Traffic Signal Optimization for Indian Urban Corridors**
>
> Deployed on Hugging Face Spaces · Meta PyTorch OpenEnv Hackathon 2026

---

## Overview

JaamCTRL simulates a realistic 3-intersection arterial corridor in **Connaught Place, Delhi** with authentic Indian traffic (motorcycles, autos, cars, buses) and applies two levels of AI control:

| **Level** | **Approach** | **Performance** |
|---|---|---|
| **Rule-Based** | Queue-aware green extension + coordination | ~20–25% delay reduction |
| **PPO RL Agent** | Coordinated multi-intersection control | ~28–35% delay reduction |

---

## Quick Start

This Space runs the **interactive Streamlit dashboard** where you can:
- View live traffic metrics (queues, delays, throughput)
- Switch between rule-based and RL control strategies
- Visualize signal phases and vehicle heatmaps
- Compare performance metrics

No installation needed — **it runs instantly in your browser!**

---

## Architecture

```
JaamCTRL/
├── app.py                    ← Streamlit dashboard (this runs in the Space)
├── inference.py              ← OpenEnv compliance script
├── env/                      ← Gymnasium environment
│   ├── base_env.py
│   ├── observation.py
│   ├── reward.py
│   └── incident_manager.py
├── src/
│   ├── rl_agent.py          ← PPO agent training/loading
│   └── ...
├── models/
│   └── ppo_jaam_ctrl.zip    ← Trained PPO model
└── sumo/
    └── corridor.sumocfg     ← Traffic network definition
```

---

## How It Works

### Traffic Environment
- **Network**: 3-signal corridor (Barakhamba Rd → CP Core → Patel Chowk)
- **Vehicles**: 60% motorcycles, 30% autos/cars, 10% buses (realistic Indian mix)
- **Chaos**: Random animal crossings, mid-episode accidents, rush-hour spikes

### RL Agent (PPO)
- **Observation**: Queue lengths, phase timers, GPS probe density (46-dim vector)
- **Action**: Switch/extend green phases per junction (8 discrete actions)
- **Reward**: Minimize average delay + queue length + violations
- **Architecture**: Stable-Baselines3 PPO with fully-connected policy

### Baseline Controller
- Queue-aware phase extension (if queue above threshold → extend green)
- Green-wave offset optimization (reduce car stops via coordinated phases)

---

## Performance Metrics

The dashboard displays:
- **Delay Reduction**: % improvement over fixed-time signals
- **Throughput**: Vehicles discharged per minute
- **Queue Length**: Real-time queue size per approach lane
- **Overflow Events**: Number of times capacity exceeded
- **Phase Timeline**: Visual signal timing and vehicle arrivals

---

## Configuration

### Task Difficulty (via sidebar)
- **Easy (Task 1)**: Single intersection (Barakhamba Rd)
- **Medium (Task 2)**: Two intersections (Barakhamba + CP Core)
- **Hard (Task 3)**: Full 3-intersection corridor + chaos

### Control Strategy
- **Fixed-Time**: 60s green per direction (baseline)
- **Rule-Based**: Queue-aware extension + green-wave coordination
- **PPO RL**: Learned multi-intersection control

---

## Educational Value

This Space demonstrates:
- Real-world RL application (traffic signal control)
- Multi-agent coordination (3 independent signals)
- Dealing with chaos (accidents, animals, noise)
- Comparing heuristic vs. learned policies
- Indian urban traffic modeling

**Perfect for**: Students, researchers, and practitioners learning RL and optimization.

---

## OpenEnv Compliance

Submitted to **Meta PyTorch OpenEnv Hackathon** (Scaler School of Technology 2026):
- Full Gymnasium API support
- `inference.py` script with standardized logging
- Mock mode for CI/judging (no SUMO needed)
- Three progressive difficulty tasks
- Quantifiable metrics (delay, throughput, success rate)

**OpenEnv Spec**: [openenv.yaml](openenv.yaml)

---

## Repository & Model

- **GitHub**: [your-org/jaamctrl](https://github.com/your-org/jaamctrl)  
- **Model**: Trained PPO agent (`models/ppo_jaam_ctrl.zip`)  
- **License**: MIT  
- **Authors**: JaamCTRL Team

---

## Visual Design

- **Color Palette**: Neon Noir (dark mode optimized)
  - **Yellow** (`#f7f43c`): Primary accent, headers
  - **Pink** (`#ff8f96`): RL agent, secondary accent
  - **Mint** (`#b4feb2`): Success, positive metrics

---

## Citation

If you use JaamCTRL in your research:

```bibtex
@hackathon{jaamctrl2026,
  title={JaamCTRL: AI Adaptive Traffic Signal Control},
  author={JaamCTRL Team},
  year={2026},
  organization={Meta PyTorch OpenEnv Hackathon},
  url={https://huggingface.co/spaces/WitchedWick/jaamctrl}
}
```

---

## Questions & Feedback

- **Report Issues**: [GitHub Issues](https://github.com/your-org/jaamctrl/issues)
- **Suggest Features**: [GitHub Discussions](https://github.com/your-org/jaamctrl/discussions)
- **Contact**: [team email]

---

**Made for safer Indian cities.**
