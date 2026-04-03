"""Batch robustness campaign runner for DEF-DTP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from anima_def_dtp.attack.engine import DtpAttackEngine
from anima_def_dtp.cli import create_predictor
from anima_def_dtp.config import get_settings
from anima_def_dtp.constants import ATTACK_GOALS, PAPER_ARXIV_ID
from anima_def_dtp.data import scenario_window_from_repo_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DEF-DTP attack campaign.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a JSON manifest listing case files.",
    )
    parser.add_argument("--predictor", default="replay")
    parser.add_argument("--dataset", default="nuscenes")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/artifacts-datai/campaigns/def_dtp"),
    )
    parser.add_argument("--objectives", nargs="+", default=list(ATTACK_GOALS))
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    engine = DtpAttackEngine(settings)

    run_id = f"campaign_{int(time.time())}"
    out_dir = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    results_path = out_dir / "results.jsonl"

    with results_path.open("w") as fh:
        for case_entry in manifest:
            case_path = Path(case_entry["path"])
            if not case_path.exists():
                fh.write(
                    json.dumps({"case": str(case_path), "error": "file not found"}) + "\n"
                )
                continue
            raw = json.loads(case_path.read_text())
            window = scenario_window_from_repo_dict(raw)
            predictor = create_predictor(
                args.predictor,
                window,
                checkpoint=args.checkpoint,
                device=args.device,
            )
            target = case_entry.get("target_object_id") or sorted(window.objects)[0]
            for objective in args.objectives:
                if args.dry_run:
                    record = {
                        "case": str(case_path),
                        "objective": objective,
                        "dry_run": True,
                    }
                else:
                    result = engine.run_case(
                        dataset_name=args.dataset,
                        predictor=predictor,
                        window=window,
                        target_object_id=target,
                        objective_name=objective,
                    )
                    record = {
                        "case": str(case_path),
                        "paper_arxiv": PAPER_ARXIV_ID,
                        "predictor": args.predictor,
                        "objective": objective,
                        "is_adversarial": result.is_adversarial,
                        "query_count": result.query_count,
                        "distance_to_original": result.distance_to_original,
                        "metrics": result.metrics,
                    }
                fh.write(json.dumps(record) + "\n")
                fh.flush()

    print(f"Campaign complete: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
