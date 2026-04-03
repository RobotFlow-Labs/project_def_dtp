from pathlib import Path

from scripts.reproduce_paper import build_report_paths


def test_report_files_are_named_consistently() -> None:
    paths = build_report_paths(Path("artifacts/reproduction"))
    assert paths["table_i"].name == "table_i.json"
    assert paths["table_ii"].name == "table_ii.json"
    assert paths["parity_summary"].name == "parity_summary.json"
