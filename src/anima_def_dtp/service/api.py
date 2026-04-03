"""FastAPI application for DEF-DTP."""

from __future__ import annotations

from fastapi import FastAPI

from anima_def_dtp.criteria import AdversarialCriterion
from anima_def_dtp.data import scenario_window_from_repo_dict
from anima_def_dtp.service.deps import get_engine, resolve_predictor
from anima_def_dtp.service.schemas import (
    AttackRequest,
    AttackResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
)
from anima_def_dtp.version import __version__

app = FastAPI(title="DEF-DTP", version=__version__)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(version=__version__)


@app.post("/attack", response_model=AttackResponse)
def run_attack(req: AttackRequest) -> AttackResponse:
    window = scenario_window_from_repo_dict(req.window)
    predictor = resolve_predictor(req.predictor_name, window)
    target = req.target_object_id or sorted(window.objects)[0]
    engine = get_engine()
    result = engine.run_case(
        dataset_name=req.dataset_name,
        predictor=predictor,
        window=window,
        target_object_id=target,
        objective_name=req.objective_name,
    )
    return AttackResponse.from_result(result)


@app.post("/evaluate", response_model=EvaluateResponse)
def run_evaluate(req: EvaluateRequest) -> EvaluateResponse:
    window = scenario_window_from_repo_dict(req.window)
    predictor = resolve_predictor(req.predictor_name, window)
    target = req.target_object_id or sorted(window.objects)[0]
    bundle = predictor.predict(window, target_object_id=target)
    value, is_adversarial = AdversarialCriterion(req.dataset_name).evaluate(
        bundle, target, req.objective_name
    )
    return EvaluateResponse(
        target_object_id=target,
        objective=req.objective_name,
        metric_value=value,
        is_adversarial=is_adversarial,
    )
