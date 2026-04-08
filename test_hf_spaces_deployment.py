#!/usr/bin/env python3
"""
test_hf_spaces_deployment.py
Test that all dependencies are available and the app can start in HF Spaces mock mode.

Usage:
  python test_hf_spaces_deployment.py
"""

import os
import sys
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────
os.environ["MOCK_SUMO"] = "1"  # Force mock mode like HF Spaces
ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
if SRC not in sys.path:
    sys.path.insert(0, str(SRC))

print("=" * 80)
print("JaamCTRL HF Spaces Deployment Test")
print("=" * 80)

# ── Test 1: Check Python version ──────────────────────────────────────────────
print("\n1. Python Version")
py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
required_version = "3.11"
status = "[OK]" if sys.version_info >= (3, 11) else "[WARN]"
print(f"{status} Python {py_version} (required: {required_version}+)")
if sys.version_info < (3, 11):
    print("   Warning: Python 3.11+ recommended for full compatibility")

# ── Test 2: Check Core Dependencies ───────────────────────────────────────────
print("\n2. Core Dependencies")
deps = [
    ("numpy", "NumPy"),
    ("gymnasium", "Gymnasium"),
    ("stable_baselines3", "Stable-Baselines3"),
    ("torch", "PyTorch"),
    ("streamlit", "Streamlit"),
    ("folium", "Folium"),
    ("altair", "Altair"),
    ("pandas", "Pandas"),
]

all_ok = True
for module_name, display_name in deps:
    try:
        __import__(module_name)
        print(f"[OK] {display_name:20} OK")
    except ImportError:
        print(f"[FAIL] {display_name:20} MISSING")
        all_ok = False

# ── Test 3: Optional Dependencies ─────────────────────────────────────────────
print("\n3. Optional Dependencies (SUMO — Not needed for HF Spaces)")
optional_deps = [
    ("traci", "TraCI"),
    ("sumolib", "SUMO Library"),
]

for module_name, display_name in optional_deps:
    try:
        __import__(module_name)
        print(f"[OK] {display_name:20} OK (SUMO available locally)")
    except ImportError:
        print(f"[WARN] {display_name:20} Not available (OK for HF Spaces)")

# ── Test 4: Environment Setup ─────────────────────────────────────────────────
print("\n4. Environment Setup")
mock_sumo = os.environ.get("MOCK_SUMO", "0") == "1"
print(f"{'[OK]' if mock_sumo else '[FAIL]'} MOCK_SUMO: {mock_sumo} (should be True for HF Spaces)")

# ── Test 5: Load JaamCTRL Environment ─────────────────────────────────────────
print("\n5. JaamCTRL Gymnasium Registration")
try:
    from env import register_envs, TASK_CONFIGS
    register_envs()
    print(f"[OK] Environment registration successful")
    print(f"   Available tasks: {list(TASK_CONFIGS.keys())}")
except Exception as e:
    print(f"[FAIL] Failed to register environments: {e}")
    all_ok = False

# ── Test 6: Load Trained Model ────────────────────────────────────────────────
print("\n6. Trained PPO Model")
model_path = ROOT / "models" / "ppo_jaam_ctrl.zip"
if model_path.exists():
    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path))
        print(f"[OK] Model loaded: {model_path.name}")
        print(f"   Policy type: {type(model.policy).__name__}")
        print(f"   Observation shape: {model.policy.observation_space.shape}")
    except Exception as e:
        print(f"[WARN] Model loading warning: {e}")
else:
    print(f"[FAIL] Model not found: {model_path}")
    all_ok = False

# ── Test 7: Mock Environment Test ─────────────────────────────────────────────
print("\n7. Mock Environment Test")
try:
    import gymnasium as gym
    from env import JaamCTRLTrafficEnv
    
    env = JaamCTRLTrafficEnv(task_id=1, mock_sumo=True)
    obs, info = env.reset()
    print(f"[OK] Mock environment created & reset successfully")
    print(f"   Observation type: {type(obs)}")
    print(f"   Observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
    
    # Take a step
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"[OK] Step executed successfully")
    print(f"   Reward: {reward:.4f}")
    
    env.close()
except Exception as e:
    print(f"[FAIL] Mock environment test failed: {e}")
    import traceback
    traceback.print_exc()
    all_ok = False

# ── Test 8: File Structure ────────────────────────────────────────────────────
print("\n8. Required Files for HF Spaces")
required_files = [
    "app.py",
    "requirements.txt",
    "README.md",
    "openenv.yaml",
    "models/ppo_jaam_ctrl.zip",
    "env/__init__.py",
    "src/__init__.py",
]

for file_path in required_files:
    full_path = ROOT / file_path
    status = "[OK]" if full_path.exists() else "[FAIL]"
    print(f"{status} {file_path:40} {'OK' if full_path.exists() else 'MISSING'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
if all_ok:
    print("[OK] All tests passed! Ready for HF Spaces deployment.")
else:
    print("[WARN] Some issues detected. See above for details.")

print("="*80)
print("\nNext steps:")
print("  1. Read: DEPLOY_TO_HF_SPACES.md")
print("  2. Create Space: https://huggingface.co/spaces/create")
print("  3. Upload files to the Space")
print("  4. Visit: https://huggingface.co/spaces/WitchedWick/jaamctrl")
print()
