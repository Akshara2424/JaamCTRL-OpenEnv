"""
env/models.py
───────────────────────────────────────────────────────────────────────────────
OpenEnv-compliant Pydantic models for JaamCTRL.

Defines:
  - Observation: typed, structured observation from the environment
  - Action: agent action (multi-discrete phase selection)
  - Reward: reward breakdown with detailed metrics
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ObservationData(BaseModel):
    """
    Structured environment observation.
    
    All vectors are zero-padded to size 3 (max intersections).
    Inactive intersections have zeros in their slots.
    """
    queue_lengths: List[List[float]] = Field(
        ..., description="Queue length per lane (3 intersections × 4 lanes)"
    )
    current_phase: List[int] = Field(
        ..., description="Current traffic light phase (0-3) per intersection"
    )
    phase_elapsed: List[float] = Field(
        ..., description="Seconds elapsed in current phase per intersection"
    )
    probe_density: List[List[float]] = Field(
        ..., description="GPS probe vehicle density in 8 spatial bins per intersection"
    )
    incident_active: List[bool] = Field(
        ..., description="Whether an incident is active per intersection"
    )
    time_of_day_norm: float = Field(
        ..., description="Normalized time of day (0.0 to 1.0, 24-hour cycle)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "queue_lengths": [[10.0, 8.0, 5.0, 3.0], [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]],
                "current_phase": [0, 2, 0],
                "phase_elapsed": [7.5, 3.2, 0.0],
                "probe_density": [[0.5, 0.6, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                "incident_active": [False, False, False],
                "time_of_day_norm": 0.5,
            }
        }


class ActionData(BaseModel):
    """
    Agent action: one phase (0-3) per active traffic light.
    
    - Task 1: 1 phase (Barakhamba only)
    - Task 2: 2 phases (Barakhamba + CP Core)
    - Task 3: 3 phases (full corridor)
    """
    phases: List[int] = Field(
        ..., description="Phase selection (0-3) per traffic light, zero-padded to size 3"
    )
    
    def validate_phases(self):
        """All phase values must be 0-3."""
        for phase in self.phases:
            assert 0 <= phase <= 3, f"Invalid phase {phase}: must be 0-3"
    
    class Config:
        json_schema_extra = {
            "example": {"phases": [0, 2, 1]}
        }


class RewardBreakdown(BaseModel):
    """
    Detailed breakdown of reward components for transparency.
    """
    r_queue: float = Field(..., description="Penalty from total queue length")
    r_throughput: float = Field(..., description="Reward from vehicles exiting")
    r_stops: float = Field(..., description="Penalty from new stop events")
    r_long_wait: float = Field(..., description="Hard penalty for excessive waits")
    r_green_wave: float = Field(..., description="Bonus for green-wave coordination")
    r_overflow: float = Field(..., description="Penalty for lane overflow")
    r_incident: float = Field(..., description="Bonus when incident queue clears")
    r_thrash: float = Field(..., description="Penalty for rapid phase switching")
    
    class Config:
        json_schema_extra = {
            "example": {
                "r_queue": -2.5,
                "r_throughput": 1.2,
                "r_stops": -0.3,
                "r_long_wait": 0.0,
                "r_green_wave": 0.5,
                "r_overflow": 0.0,
                "r_incident": 0.0,
                "r_thrash": 0.0,
            }
        }


class RewardData(BaseModel):
    """
    Final reward and breakdown.
    """
    total: float = Field(..., description="Total reward for this step (clipped to [-10, +5])")
    breakdown: RewardBreakdown = Field(..., description="Reward component breakdown")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total": -0.1,
                "breakdown": {
                    "r_queue": -2.5,
                    "r_throughput": 1.2,
                    "r_stops": -0.3,
                    "r_long_wait": 0.0,
                    "r_green_wave": 0.5,
                    "r_overflow": 0.0,
                    "r_incident": 0.0,
                    "r_thrash": 0.0,
                }
            }
        }


class StepOutput(BaseModel):
    """
    Complete output of a single environment step.
    """
    observation: ObservationData
    reward: float
    done: bool
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional info: episode_metrics, metrics summary, etc."
    )


class TaskMetrics(BaseModel):
    """
    Final task performance metrics for grading.
    """
    avg_delay_reduction: float = Field(
        ..., description="% reduction in avg vehicle delay vs. fixed-time baseline"
    )
    avg_throughput_improvement: float = Field(
        ..., description="% improvement in vehicles exiting intersections"
    )
    overflow_events: int = Field(
        ..., description="Number of lane overflow events (lanes exceeded capacity)"
    )
    total_reward: float = Field(
        ..., description="Accumulated reward over full episode"
    )
    episode_length: int = Field(
        ..., description="Number of steps completed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "avg_delay_reduction": 28.5,
                "avg_throughput_improvement": 15.2,
                "overflow_events": 0,
                "total_reward": 125.3,
                "episode_length": 300,
            }
        }
