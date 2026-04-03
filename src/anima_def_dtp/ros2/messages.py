"""Python-side message contracts for the DEF-DTP ROS2 node.

These dataclasses mirror the message types that would be published
over ROS2 topics.  When rclpy is available the node converts these
to actual ROS2 messages; otherwise they serve as typed contracts for
unit testing and offline replay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrajectoryWindowMsg:
    """Incoming scenario window for the attack node."""

    dataset_name: str = "nuscenes"
    predictor_name: str = "replay"
    objective_name: str = "ade"
    target_object_id: str = ""
    window_json: str = ""  # serialised repo-style JSON


@dataclass
class AttackResultMsg:
    """Outgoing attack result published by the node."""

    target_object_id: str = ""
    objective: str = ""
    is_adversarial: bool = False
    query_count: int = 0
    distance_to_original: float = 0.0
    perturbation_json: str = ""  # serialised list[list[float]]
    metrics_json: str = ""  # serialised dict[str, float]


@dataclass
class EvaluationSummaryMsg:
    """Optional summary published after a batch run."""

    total_runs: int = 0
    attack_success_rate: float = 0.0
    max_query_count: int = 0
    records_json: str = ""  # serialised list of protocol records
