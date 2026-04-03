"""Attack orchestration layer."""

from __future__ import annotations

from anima_def_dtp.attack.boundary import BoundaryWalker
from anima_def_dtp.config import DefDtpSettings
from anima_def_dtp.criteria import AdversarialCriterion
from anima_def_dtp.predictors.base import PredictorAdapter
from anima_def_dtp.types import AttackResult, ScenarioWindow


class DtpAttackEngine:
    """Thin orchestration wrapper over the boundary walker."""

    def __init__(self, settings: DefDtpSettings):
        self.settings = settings

    def build_walker(self) -> BoundaryWalker:
        attack = self.settings.attack
        return BoundaryWalker(
            orthogonal_step=attack.orthogonal_step,
            forward_step=attack.forward_step,
            orthogonal_decay=attack.orthogonal_decay,
            forward_decay=attack.forward_decay,
            max_iter=attack.max_iter,
            tolerance=attack.tolerance,
        )

    def run_case(
        self,
        *,
        dataset_name: str,
        predictor: PredictorAdapter,
        window: ScenarioWindow,
        target_object_id: str,
        objective_name: str,
    ) -> AttackResult:
        criterion = AdversarialCriterion(dataset_name)
        walker = self.build_walker()
        return walker.run(window, predictor, criterion, target_object_id, objective_name)
