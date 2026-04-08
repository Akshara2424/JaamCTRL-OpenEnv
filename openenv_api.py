"""
openenv_api.py
──────────────────────────────────────────────────────────────────────────────
HTTP API wrapper for JaamCTRL environment.
Exposes Gymnasium reset/step/state endpoints as POST routes.

Supports OpenEnv hackathon submission checker.

Usage:
  python openenv_api.py                    # Start server on 0.0.0.0:5000
  python openenv_api.py --port 8000        # Custom port
  FLASK_DEBUG=1 python openenv_api.py      # Debug mode

Environment variables:
  FLASK_PORT     Server port (default: 5000)
  MOCK_SUMO      Force mock mode (default: "1")
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from typing import Any, Dict, Tuple

import numpy as np
from flask import Flask, request, jsonify

# ── Setup paths ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Force mock SUMO for API mode (unless explicitly disabled)
if "MOCK_SUMO" not in os.environ:
    os.environ["MOCK_SUMO"] = "1"

# ── Imports ─────────────────────────────────────────────────────────────────
from env import JaamCTRLTrafficEnv, register_envs

register_envs()

# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Global environment instance (reset on each request)
_env = None
_current_task = 1


def _to_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-JSON types to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "version": "1.0.0"}), 200


@app.route("/reset", methods=["POST"])
def reset():
    """
    Reset the environment.
    
    Request body (JSON):
    {
      "task_id": 1,
      "seed": 42,
      "options": {}
    }
    
    Response:
    {
      "observation": {...},
      "info": {...}
    }
    """
    global _env, _current_task
    
    try:
        data = request.get_json() or {}
        task_id = data.get("task_id", 1)
        seed = data.get("seed")
        options = data.get("options", {})
        
        # Create/recreate environment if task changed
        if _env is None or task_id != _current_task:
            if _env:
                _env.close()
            _env = JaamCTRLTrafficEnv(task_id=task_id, mock_sumo=True)
            _current_task = task_id
        
        obs, info = _env.reset(seed=seed, options=options)
        
        return jsonify({
            "observation": _to_serializable(obs),
            "info": _to_serializable(info),
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/step", methods=["POST"])
def step():
    """
    Execute one step in the environment.
    
    Request body (JSON):
    {
      "action": [0, 1, 2]  or [1]  depending on task
    }
    
    Response:
    {
      "observation": {...},
      "reward": 12.5,
      "terminated": false,
      "truncated": false,
      "info": {...}
    }
    """
    global _env
    
    if _env is None:
        return jsonify({"error": "Environment not initialized. Call /reset first."}), 400
    
    try:
        data = request.get_json() or {}
        action = data.get("action")
        
        if action is None:
            return jsonify({"error": "Missing 'action' in request body"}), 400
        
        action = np.array(action, dtype=np.int64)
        obs, reward, terminated, truncated, info = _env.step(action)
        
        return jsonify({
            "observation": _to_serializable(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _to_serializable(info),
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/state", methods=["GET"])
def state():
    """
    Get the current environment state.
    
    Response:
    {
      "task_id": 1,
      "step": 42,
      "sim_time_s": 210.0,
      ...
    }
    """
    global _env
    
    if _env is None:
        return jsonify({"error": "Environment not initialized. Call /reset first."}), 400
    
    try:
        state_dict = _env.state()
        return jsonify(_to_serializable(state_dict)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found", "path": request.path}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed", "method": request.method}), 405


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error", "details": str(error)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv HTTP API server")
    parser.add_argument("--port", type=int, default=int(os.getenv("FLASK_PORT", 5000)),
                        help="Port to run server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print(f"Starting OpenEnv API server on {args.host}:{args.port}")
    print(f"Health check: GET http://localhost:{args.port}/health")
    print(f"Reset env: POST http://localhost:{args.port}/reset")
    print(f"Step env: POST http://localhost:{args.port}/step")
    print(f"Get state: GET http://localhost:{args.port}/state")
    print()
    
    app.run(host=args.host, port=args.port, debug=False)
