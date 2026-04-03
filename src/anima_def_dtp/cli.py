"""Command-line interface for local DEF-DTP runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anima_def_dtp.attack.engine import DtpAttackEngine
from anima_def_dtp.config import get_settings
from anima_def_dtp.criteria import AdversarialCriterion
from anima_def_dtp.data import scenario_window_from_repo_dict
from anima_def_dtp.predictors.grip import GripAdapter
from anima_def_dtp.predictors.replay import ReplayAdapter
from anima_def_dtp.predictors.trajectron import TrajectronAdapter
from anima_def_dtp.types import ScenarioWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anima-def-dtp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("normal", "attack", "evaluate-case"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--input", required=True, type=Path)
        subparser.add_argument(
            "--predictor",
            default="replay",
            choices=("replay", "grip", "trajectron", "trajectron_map"),
        )
        subparser.add_argument("--checkpoint", type=Path, default=None)
        subparser.add_argument("--device", default="auto")
        subparser.add_argument("--target-object-id", default=None)
        subparser.add_argument("--objective", default="ade")
        subparser.add_argument("--dataset", default="apolloscape")
        subparser.add_argument("--output", type=Path, default=None)
    return parser


def load_window(path: Path) -> ScenarioWindow:
    payload = json.loads(path.read_text())
    return scenario_window_from_repo_dict(payload)


def create_predictor(
    predictor_name: str,
    window: ScenarioWindow,
    *,
    checkpoint: Path | None,
    device: str,
):
    if predictor_name == "replay":
        return ReplayAdapter(
            obs_length=window.observe_length,
            pred_length=window.predict_length,
        )
    if predictor_name == "grip":
        return GripAdapter(
            obs_length=window.observe_length,
            pred_length=window.predict_length,
            checkpoint_path=checkpoint,
            device=device,
        )
    if predictor_name in {"trajectron", "trajectron_map"}:
        return TrajectronAdapter(
            obs_length=window.observe_length,
            pred_length=window.predict_length,
            checkpoint_path=checkpoint,
            use_map=predictor_name.endswith("_map"),
            device=device,
        )
    raise ValueError(f"unsupported predictor: {predictor_name}")


def command_normal(args: argparse.Namespace) -> dict[str, Any]:
    window = load_window(args.input)
    predictor = create_predictor(
        args.predictor,
        window,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    target_object_id = args.target_object_id or sorted(window.objects)[0]
    bundle = predictor.predict(window, target_object_id=target_object_id)
    result = bundle.model_dump()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    return result


def command_attack(args: argparse.Namespace) -> dict[str, Any]:
    window = load_window(args.input)
    predictor = create_predictor(
        args.predictor,
        window,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    target_object_id = args.target_object_id or sorted(window.objects)[0]
    engine = DtpAttackEngine(get_settings())
    result = engine.run_case(
        dataset_name=args.dataset,
        predictor=predictor,
        window=window,
        target_object_id=target_object_id,
        objective_name=args.objective,
    ).model_dump()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    return result


def command_evaluate_case(args: argparse.Namespace) -> dict[str, Any]:
    window = load_window(args.input)
    predictor = create_predictor(
        args.predictor,
        window,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    target_object_id = args.target_object_id or sorted(window.objects)[0]
    bundle = predictor.predict(window, target_object_id=target_object_id)
    value, is_adversarial = AdversarialCriterion(args.dataset).evaluate(
        bundle,
        target_object_id,
        args.objective,
    )
    result = {
        "target_object_id": target_object_id,
        "objective": args.objective,
        "metric_value": value,
        "is_adversarial": is_adversarial,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "normal":
        command_normal(args)
    elif args.command == "attack":
        command_attack(args)
    elif args.command == "evaluate-case":
        command_evaluate_case(args)
    else:  # pragma: no cover
        parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
