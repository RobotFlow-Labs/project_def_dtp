"""ROS2 node wrapper for the DEF-DTP attack engine.

When ``rclpy`` is available the class acts as a proper ROS2 Node.
Otherwise it works in *headless* mode for unit testing and offline replay.
"""

from __future__ import annotations

import json
from typing import Any

from anima_def_dtp.attack.engine import DtpAttackEngine
from anima_def_dtp.cli import create_predictor
from anima_def_dtp.config import get_settings
from anima_def_dtp.data import scenario_window_from_repo_dict
from anima_def_dtp.ros2.messages import AttackResultMsg, TrajectoryWindowMsg

try:
    from rclpy.node import Node as RclpyNode  # noqa: F401

    _HAS_RCLPY = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_RCLPY = False
    RclpyNode = object  # type: ignore[assignment,misc]


class DefDtpNode(RclpyNode):  # type: ignore[misc]
    """Thin ROS2 wrapper that delegates to the validated attack engine."""

    def __init__(self, node_name: str = "def_dtp", **kwargs: Any) -> None:
        if _HAS_RCLPY:
            super().__init__(node_name, **kwargs)  # type: ignore[call-arg]
        self.settings = get_settings()
        self.engine = DtpAttackEngine(self.settings)
        self._results: list[AttackResultMsg] = []

    def handle_window(self, msg: TrajectoryWindowMsg) -> AttackResultMsg:
        """Process a single trajectory window and return the attack result."""
        window = scenario_window_from_repo_dict(json.loads(msg.window_json))
        predictor = create_predictor(
            msg.predictor_name,
            window,
            checkpoint=None,
            device="auto",
        )
        target = msg.target_object_id or sorted(window.objects)[0]
        result = self.engine.run_case(
            dataset_name=msg.dataset_name,
            predictor=predictor,
            window=window,
            target_object_id=target,
            objective_name=msg.objective_name,
        )
        out = AttackResultMsg(
            target_object_id=result.target_object_id,
            objective=result.objective,
            is_adversarial=result.is_adversarial,
            query_count=result.query_count,
            distance_to_original=result.distance_to_original,
            perturbation_json=json.dumps(result.perturbation),
            metrics_json=json.dumps(result.metrics),
        )
        self._results.append(out)
        return out
