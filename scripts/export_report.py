"""Export parity and audit reports from campaign results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anima_def_dtp.constants import PAPER_ARXIV_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export DEF-DTP parity report.")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results.jsonl from run_campaign.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/artifacts-datai/reports/def_dtp/parity_report.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        return 1

    records = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    total = len(records)
    successes = sum(1 for r in records if r.get("is_adversarial", False))
    asr = successes / total if total else 0.0
    max_queries = max((r.get("query_count", 0) for r in records), default=0)

    report = {
        "paper_arxiv": PAPER_ARXIV_ID,
        "total_cases": total,
        "attack_success_rate": round(asr, 4),
        "max_query_count": max_queries,
        "paper_asr_range": [0.41, 0.81],
        "within_paper_range": 0.41 <= asr <= 0.81,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Report exported: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
