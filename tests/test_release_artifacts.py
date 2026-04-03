"""Release artifact validation tests."""

import json

from scripts.export_report import build_parser


def test_export_parser_has_results_arg() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "--results" in help_text


def test_export_report_generates_parity(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case": "test",
                        "objective": "ade",
                        "is_adversarial": True,
                        "query_count": 42,
                        "distance_to_original": 0.3,
                        "metrics": {"ade": 8.0},
                    }
                ),
                json.dumps(
                    {
                        "case": "test2",
                        "objective": "ade",
                        "is_adversarial": False,
                        "query_count": 100,
                        "distance_to_original": 1.0,
                        "metrics": {"ade": 2.0},
                    }
                ),
            ]
        )
    )
    output = tmp_path / "report.json"
    from scripts.export_report import main

    rc = main(["--results", str(results), "--output", str(output)])
    assert rc == 0
    report = json.loads(output.read_text())
    assert report["total_cases"] == 2
    assert report["attack_success_rate"] == 0.5
    assert report["max_query_count"] == 100
