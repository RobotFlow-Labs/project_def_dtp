"""Generate DEF-DTP paper reproduction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anima_def_dtp.evaluation.baselines import run_pso_baseline, run_sba_baseline


def build_report_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "table_i": output_dir / "table_i.json",
        "table_ii": output_dir / "table_ii.json",
        "parity_summary": output_dir / "parity_summary.json",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce DEF-DTP paper tables.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reproduction"))
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_paths = build_report_paths(args.output_dir)
    payload = {
        "table_i": {"status": "pending-real-data"},
        "table_ii": {"status": "pending-real-data"},
        "parity_summary": {
            "status": "pending-real-data",
            "baselines": [run_pso_baseline(), run_sba_baseline()],
            "dry_run": args.dry_run,
        },
    }
    for key, path in report_paths.items():
        path.write_text(json.dumps(payload[key], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
