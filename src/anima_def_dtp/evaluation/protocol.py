"""Evaluation protocol for DTP-Attack reproduction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from anima_def_dtp.constants import ATTACK_GOALS
from anima_def_dtp.types import AttackResult, ScenarioWindow

AttackRunner = Callable[..., AttackResult]


@dataclass(frozen=True)
class ProtocolCase:
    """A single evaluation case inside the 100-sample benchmark loop."""

    case_id: str
    dataset_name: str
    target_object_id: str
    window: ScenarioWindow


@dataclass
class ProtocolRecord:
    """One objective run inside the evaluation protocol."""

    case_id: str
    objective: str
    query_count: int
    is_adversarial: bool
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ProtocolSummary:
    """Structured protocol result for downstream reports."""

    records: list[ProtocolRecord]

    @property
    def total_runs(self) -> int:
        return len(self.records)

    @property
    def attack_success_rate(self) -> float:
        if not self.records:
            return 0.0
        successes = sum(1 for record in self.records if record.is_adversarial)
        return successes / len(self.records)

    @property
    def max_query_count(self) -> int:
        if not self.records:
            return 0
        return max(record.query_count for record in self.records)


def run_protocol(
    cases: Iterable[ProtocolCase],
    predictor,
    attacker: AttackRunner,
    *,
    objectives: Sequence[str] = ATTACK_GOALS,
    max_queries: int = 1000,
) -> ProtocolSummary:
    """Run the paper-style objective loop with hard query-budget enforcement."""

    records: list[ProtocolRecord] = []
    for case in cases:
        for objective in objectives:
            result = attacker(
                dataset_name=case.dataset_name,
                predictor=predictor,
                window=case.window,
                target_object_id=case.target_object_id,
                objective_name=objective,
            )
            if result.query_count > max_queries:
                raise ValueError(
                    f"query budget exceeded for {case.case_id}/{objective}: "
                    f"{result.query_count} > {max_queries}"
                )
            records.append(
                ProtocolRecord(
                    case_id=case.case_id,
                    objective=objective,
                    query_count=result.query_count,
                    is_adversarial=result.is_adversarial,
                    metrics=dict(result.metrics),
                )
            )
    return ProtocolSummary(records=records)
