"""
env/graders.py
───────────────────────────────────────────────────────────────────────────────
Task-specific graders for JaamCTRL.

Each task has:
  1. Clear success metrics
  2. Deterministic grading logic
  3. Baseline (fixed-time) and target (RL agent) performance thresholds

Graders compute a score (0.0–1.0) based on how well the agent performed
relative to the fixed-time baseline and target difficulty.
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Dict

from env.models import TaskMetrics


class TaskGrader:
    """Base class for task graders."""
    
    task_id: int
    task_name: str
    
    # Baseline performance (fixed-time signal 30s/30s green/red)
    baseline_delay_reduction: float = 0.0  # by definition
    baseline_throughput: float = 100.0     # vehicles/hour (normalized)
    baseline_overflow_events: int = 0
    
    # Success thresholds (what the agent must beat)
    min_delay_reduction: float
    min_throughput_improvement: float
    max_overflow_events: int
    
    def __init__(self):
        pass
    
    def grade(self, metrics: TaskMetrics) -> Dict[str, Any]:
        """
        Grade a completed episode.
        
        Returns
        -------
        dict with keys:
          'task_id' (int)
          'task_name' (str)
          'passed' (bool) — True if all criteria met
          'score' (float) — 0.0 to 1.0
          'details' (dict) — breakdown of each criterion
          'message' (str) — human-readable summary
        """
        passed = True
        details = {}
        
        # ── Criterion 1: Delay Reduction ────────────────────────────────────
        delay_ok = metrics.avg_delay_reduction >= self.min_delay_reduction
        details['delay_reduction'] = {
            'achieved': metrics.avg_delay_reduction,
            'required': self.min_delay_reduction,
            'passed': delay_ok,
        }
        passed = passed and delay_ok
        
        # ── Criterion 2: Throughput Improvement ─────────────────────────────
        throughput_ok = (
            metrics.avg_throughput_improvement >= self.min_throughput_improvement
        )
        details['throughput_improvement'] = {
            'achieved': metrics.avg_throughput_improvement,
            'required': self.min_throughput_improvement,
            'passed': throughput_ok,
        }
        passed = passed and throughput_ok
        
        # ── Criterion 3: No Lane Overflow (Hard Constraint) ──────────────────
        overflow_ok = metrics.overflow_events <= self.max_overflow_events
        details['overflow_events'] = {
            'achieved': metrics.overflow_events,
            'max_allowed': self.max_overflow_events,
            'passed': overflow_ok,
        }
        passed = passed and overflow_ok
        
        # ── Score Computation ──────────────────────────────────────────────
        # Scores are incremental: each criterion contributes to the final score.
        # Full score (1.0) requires beating all three thresholds by a margin.
        
        delay_score = min(1.0, metrics.avg_delay_reduction / (self.min_delay_reduction + 10.0))
        throughput_score = min(1.0, metrics.avg_throughput_improvement / (self.min_throughput_improvement + 10.0))
        overflow_score = 1.0 if overflow_ok else 0.0
        
        score = (delay_score + throughput_score + overflow_score) / 3.0
        
        message = (
            f"{self.task_name}: "
            f"delay={metrics.avg_delay_reduction:.1f}% "
            f"(need {self.min_delay_reduction}%), "
            f"throughput={metrics.avg_throughput_improvement:.1f}% "
            f"(need {self.min_throughput_improvement}%), "
            f"overflow={metrics.overflow_events} events "
            f"(max {self.max_overflow_events}). "
            f"Score: {score:.2f} — "
            f"{'PASS ✓' if passed else 'FAIL ✗'}"
        )
        
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'passed': passed,
            'score': score,
            'details': details,
            'message': message,
        }


class Task1Grader(TaskGrader):
    """
    Task 1 — Easy: Single intersection (Barakhamba), cars only, no incidents.
    
    Success criteria:
      - ≥15% reduction in average vehicle delay
      - ≥10% throughput improvement
      - 0 lane overflow events
    """
    task_id = 1
    task_name = "Task 1 (Easy)"
    
    min_delay_reduction = 15.0
    min_throughput_improvement = 10.0
    max_overflow_events = 0


class Task2Grader(TaskGrader):
    """
    Task 2 — Medium: Two intersections, mixed traffic, pedestrians/congestion.
    
    Success criteria:
      - ≥20% reduction in average vehicle delay
      - ≥15% throughput improvement (multi-intersection coordination)
      - 0 lane overflow events
    """
    task_id = 2
    task_name = "Task 2 (Medium)"
    
    min_delay_reduction = 20.0
    min_throughput_improvement = 15.0
    max_overflow_events = 0


class Task3Grader(TaskGrader):
    """
    Task 3 — Hard: Full corridor (3 intersections), realistic chaos, incidents.
    
    Success criteria:
      - ≥25% reduction in average vehicle delay
      - ≥20% throughput improvement (full-corridor coordination)
      - 0 lane overflow events (strict constraint)
    """
    task_id = 3
    task_name = "Task 3 (Hard)"
    
    min_delay_reduction = 25.0
    min_throughput_improvement = 20.0
    max_overflow_events = 0


GRADERS = {
    1: Task1Grader(),
    2: Task2Grader(),
    3: Task3Grader(),
}


def grade_task(task_id: int, metrics: TaskMetrics) -> Dict[str, Any]:
    """Convenience function: grade a task by ID."""
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id {task_id}. Valid: 1, 2, 3.")
    return GRADERS[task_id].grade(metrics)


def grade_all_tasks(
    metrics_dict: Dict[int, TaskMetrics]
) -> Dict[int, Dict[str, Any]]:
    """Grade all tasks from a metrics dict."""
    return {
        task_id: grade_task(task_id, metrics)
        for task_id, metrics in metrics_dict.items()
    }
