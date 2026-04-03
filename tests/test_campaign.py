"""Smoke tests for the campaign runner."""

import json

from scripts.run_campaign import build_parser


def test_campaign_parser_has_manifest_arg() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "--manifest" in help_text
    assert "--predictor" in help_text


def test_campaign_dry_run(tmp_path) -> None:
    case_file = tmp_path / "case.json"
    case_file.write_text(
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
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps([{"path": str(case_file), "target_object_id": "1"}])
    )
    from scripts.run_campaign import main

    rc = main(
        [
            "--manifest",
            str(manifest_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--objectives",
            "ade",
            "--dry-run",
        ]
    )
    assert rc == 0
    results_dir = list((tmp_path / "out").iterdir())
    assert len(results_dir) == 1
