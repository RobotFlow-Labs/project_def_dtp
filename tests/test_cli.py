import json

from anima_def_dtp.cli import build_parser, main


def test_cli_help_smoke() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "anima-def-dtp" in help_text
    assert "evaluate-case" in help_text


def test_cli_evaluate_case_replay(tmp_path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
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
    )
    output_path = tmp_path / "result.json"
    rc = main(
        [
            "evaluate-case",
            "--input",
            str(case_path),
            "--predictor",
            "replay",
            "--dataset",
            "apolloscape",
            "--objective",
            "ade",
            "--output",
            str(output_path),
        ]
    )
    assert rc == 0
    result = json.loads(output_path.read_text())
    assert result["is_adversarial"] is False
