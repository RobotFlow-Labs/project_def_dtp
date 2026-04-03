"""Smoke tests for the DEF-DTP FastAPI endpoints."""

from fastapi.testclient import TestClient

from anima_def_dtp.service.api import app

client = TestClient(app)

WINDOW = {
    "observe_length": 2,
    "predict_length": 2,
    "time_step": 0.5,
    "objects": {
        "1": {
            "type": 0,
            "observe_trace": [[0.0, 0.0], [1.0, 0.0]],
            "future_trace": [[2.0, 0.0], [3.0, 0.0]],
        }
    },
}


def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["module"] == "def-dtp"


def test_evaluate_returns_result() -> None:
    resp = client.post(
        "/evaluate",
        json={
            "dataset_name": "apolloscape",
            "predictor_name": "replay",
            "objective_name": "ade",
            "window": WINDOW,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "is_adversarial" in data
    assert data["target_object_id"] == "1"


def test_attack_returns_adversarial_result() -> None:
    resp = client.post(
        "/attack",
        json={
            "dataset_name": "apolloscape",
            "predictor_name": "replay",
            "objective_name": "left",
            "window": WINDOW,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_count"] > 0
    assert isinstance(data["perturbation"], list)
