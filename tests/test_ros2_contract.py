"""Contract tests for the DEF-DTP ROS2 bridge — no rclpy required."""

import json

from anima_def_dtp.ros2.messages import AttackResultMsg, TrajectoryWindowMsg
from anima_def_dtp.ros2.node import DefDtpNode

WINDOW_JSON = json.dumps(
    {
        "observe_length": 2,
        "predict_length": 2,
        "time_step": 0.5,
        "objects": {
            "1": {
                "type": 0,
                "observe_trace": [[0.0, 0.0], [1.0, 0.0]],
                "future_trace": [[2.0, 0.0], [3.0, 0.0]],
            }
        },
    }
)


def test_message_round_trip() -> None:
    msg = TrajectoryWindowMsg(
        dataset_name="apolloscape",
        predictor_name="replay",
        objective_name="left",
        target_object_id="1",
        window_json=WINDOW_JSON,
    )
    assert msg.dataset_name == "apolloscape"
    assert msg.window_json


def test_node_handle_window_returns_result() -> None:
    node = DefDtpNode.__new__(DefDtpNode)
    node.settings = None  # avoid rclpy init
    from anima_def_dtp.attack.engine import DtpAttackEngine
    from anima_def_dtp.config import get_settings

    node.settings = get_settings()
    node.engine = DtpAttackEngine(node.settings)
    node._results = []

    msg = TrajectoryWindowMsg(
        dataset_name="apolloscape",
        predictor_name="replay",
        objective_name="left",
        target_object_id="1",
        window_json=WINDOW_JSON,
    )
    result = node.handle_window(msg)
    assert isinstance(result, AttackResultMsg)
    assert result.query_count > 0
    assert result.target_object_id == "1"


def test_launch_file_compiles() -> None:
    """Verify the launch file is valid Python."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "def_dtp_launch",
        "launch/def_dtp.launch.py",
    )
    assert spec is not None
