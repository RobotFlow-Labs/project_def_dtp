"""Decision-based boundary walking attack implementation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from anima_def_dtp.criteria import AdversarialCriterion
from anima_def_dtp.predictors.base import PredictorAdapter
from anima_def_dtp.types import AttackResult, PredictionBundle, ScenarioWindow


def _copy_trace(trace: list[list[float]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in trace]


def _distance(trace_a: list[list[float]], trace_b: list[list[float]]) -> float:
    if len(trace_a) != len(trace_b):
        raise ValueError("trace lengths must match")
    if not trace_a:
        return 0.0
    total = 0.0
    for point_a, point_b in zip(trace_a, trace_b, strict=True):
        total += math.dist(point_a, point_b)
    return total / len(trace_a)


@dataclass
class BoundaryState:
    trace: list[list[float]]
    query_count: int = 0
    objective_value: float = 0.0


class BoundaryWalker:
    """Synthetic-testable implementation of Algorithm 1."""

    def __init__(
        self,
        *,
        orthogonal_step: float = 1.0,
        forward_step: float = 0.1,
        orthogonal_decay: float = 0.95,
        forward_decay: float = 0.9,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        seed: int = 7,
    ):
        self.orthogonal_step = orthogonal_step
        self.forward_step = forward_step
        self.orthogonal_decay = orthogonal_decay
        self.forward_decay = forward_decay
        self.max_iter = max_iter
        self.tolerance = tolerance
        self._random = random.Random(seed)

    def run(
        self,
        window: ScenarioWindow,
        predictor: PredictorAdapter,
        criterion: AdversarialCriterion,
        target_object_id: str,
        objective_name: str,
    ) -> AttackResult:
        original = _copy_trace(window.objects[target_object_id].observe_trace)
        state = self._random_adversarial_init(
            window,
            predictor,
            criterion,
            target_object_id,
            objective_name,
        )
        best = BoundaryState(
            trace=_copy_trace(state.trace),
            query_count=state.query_count,
            objective_value=state.objective_value,
        )
        delta = self.orthogonal_step
        epsilon = self.forward_step

        for _ in range(self.max_iter):
            orthogonal_candidate = self._orthogonal_step(state.trace, original, delta)
            bundle, value, is_adversarial = self._query(
                window, predictor, criterion, target_object_id, objective_name, orthogonal_candidate
            )
            state.query_count += 1
            if is_adversarial:
                state.trace = orthogonal_candidate
                state.objective_value = value
            else:
                delta *= self.orthogonal_decay

            forward_candidate = self._forward_step(state.trace, original, epsilon)
            bundle, value, is_adversarial = self._query(
                window, predictor, criterion, target_object_id, objective_name, forward_candidate
            )
            state.query_count += 1
            if is_adversarial:
                state.trace = forward_candidate
                state.objective_value = value
                epsilon = min(self.forward_step, epsilon / max(self.forward_decay, 1e-9))
            else:
                epsilon *= self.forward_decay

            current_distance = _distance(state.trace, original)
            best_distance = _distance(best.trace, original)
            if current_distance < best_distance:
                best = BoundaryState(
                    trace=_copy_trace(state.trace),
                    query_count=state.query_count,
                    objective_value=state.objective_value,
                )
            if epsilon < self.tolerance:
                break

        perturbation = [
            [adv[0] - base[0], adv[1] - base[1]]
            for adv, base in zip(best.trace, original, strict=True)
        ]
        return AttackResult(
            target_object_id=target_object_id,
            objective=objective_name,
            is_adversarial=True,
            query_count=best.query_count,
            distance_to_original=_distance(best.trace, original),
            perturbation=perturbation,
            metrics={objective_name: best.objective_value},
        )

    def _random_adversarial_init(
        self,
        window: ScenarioWindow,
        predictor: PredictorAdapter,
        criterion: AdversarialCriterion,
        target_object_id: str,
        objective_name: str,
    ) -> BoundaryState:
        original = _copy_trace(window.objects[target_object_id].observe_trace)
        fallback: tuple[list[list[float]], float] | None = None
        for _ in range(128):
            candidate = [
                [
                    point[0] + self._random.uniform(-10.0, 10.0),
                    point[1] + self._random.uniform(-10.0, 10.0),
                ]
                for point in original
            ]
            _, value, is_adversarial = self._query(
                window, predictor, criterion, target_object_id, objective_name, candidate
            )
            if fallback is None or value > fallback[1]:
                fallback = (_copy_trace(candidate), value)
            if is_adversarial:
                return BoundaryState(trace=candidate, query_count=1, objective_value=value)
        if fallback is None:
            raise RuntimeError("failed to initialize attack state")
        return BoundaryState(trace=fallback[0], query_count=1, objective_value=fallback[1])

    def _orthogonal_step(
        self,
        current: list[list[float]],
        original: list[list[float]],
        delta: float,
    ) -> list[list[float]]:
        candidate: list[list[float]] = []
        for current_point, original_point in zip(current, original, strict=True):
            diff_x = current_point[0] - original_point[0]
            diff_y = current_point[1] - original_point[1]
            orth_x, orth_y = -diff_y, diff_x
            norm = math.sqrt(orth_x * orth_x + orth_y * orth_y)
            if norm == 0:
                orth_x = self._random.uniform(-1.0, 1.0)
                orth_y = self._random.uniform(-1.0, 1.0)
                norm = math.sqrt(orth_x * orth_x + orth_y * orth_y) or 1.0
            scale = delta / norm
            jitter = self._random.uniform(0.5, 1.0)
            candidate.append(
                [
                    current_point[0] + orth_x * scale * jitter,
                    current_point[1] + orth_y * scale * jitter,
                ]
            )
        return candidate

    def _forward_step(
        self,
        current: list[list[float]],
        original: list[list[float]],
        epsilon: float,
    ) -> list[list[float]]:
        candidate: list[list[float]] = []
        for current_point, original_point in zip(current, original, strict=True):
            step_x = original_point[0] - current_point[0]
            step_y = original_point[1] - current_point[1]
            candidate.append(
                [
                    current_point[0] + step_x * epsilon,
                    current_point[1] + step_y * epsilon,
                ]
            )
        return candidate

    def _query(
        self,
        window: ScenarioWindow,
        predictor: PredictorAdapter,
        criterion: AdversarialCriterion,
        target_object_id: str,
        objective_name: str,
        perturbation_trace: list[list[float]],
    ) -> tuple[PredictionBundle, float, bool]:
        original = window.objects[target_object_id].observe_trace
        perturbation = [
            [cand[0] - base[0], cand[1] - base[1]]
            for cand, base in zip(perturbation_trace, original, strict=True)
        ]
        bundle = predictor.predict(
            window,
            target_object_id=target_object_id,
            perturbation=perturbation,
        )
        value, is_adversarial = criterion.evaluate(bundle, target_object_id, objective_name)
        return bundle, value, is_adversarial
