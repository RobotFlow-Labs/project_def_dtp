"""AnimaNode subclass for DEF-DTP serving."""

from __future__ import annotations

from typing import Any

from anima_def_dtp.attack.engine import DtpAttackEngine
from anima_def_dtp.cli import create_predictor
from anima_def_dtp.config import get_settings
from anima_def_dtp.data import scenario_window_from_repo_dict


class DefDtpServeNode:
    """Lightweight serve node that wraps the attack engine.

    Implements the AnimaNode contract: setup_inference + process + get_status.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine = DtpAttackEngine(self.settings)
        self._ready = True

    def setup_inference(self) -> None:
        """Load any required model weights."""
        pass  # attack engine is CPU-only by default

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run a single attack case.

        Parameters
        ----------
        input_data : dict with keys:
            window: repo-style scenario JSON
            dataset_name: str (default "nuscenes")
            predictor_name: str (default "replay")
            objective_name: str (default "ade")
            target_object_id: str | None

        Returns
        -------
        dict — AttackResult fields
        """
        window = scenario_window_from_repo_dict(input_data["window"])
        dataset = input_data.get("dataset_name", "nuscenes")
        predictor_name = input_data.get("predictor_name", "replay")
        objective = input_data.get("objective_name", "ade")
        target = input_data.get("target_object_id") or sorted(window.objects)[0]

        predictor = create_predictor(
            predictor_name, window, checkpoint=None, device="auto",
        )
        result = self.engine.run_case(
            dataset_name=dataset,
            predictor=predictor,
            window=window,
            target_object_id=target,
            objective_name=objective,
        )
        return result.model_dump()

    def get_status(self) -> dict:
        return {
            "module": "def_dtp",
            "ready": self._ready,
            "paper_arxiv": self.settings.paper_arxiv,
        }

    @property
    def ready(self) -> bool:
        return self._ready
