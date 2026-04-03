"""Container contract tests — verify health and env expectations."""

from pathlib import Path

from anima_def_dtp.service.schemas import HealthResponse
from anima_def_dtp.version import __version__


def test_health_response_has_version() -> None:
    resp = HealthResponse(version=__version__)
    assert resp.status == "ok"
    assert resp.module == "def-dtp"
    assert resp.version == __version__


def test_dockerfile_exists() -> None:
    root = Path(__file__).resolve().parent.parent
    assert (root / "docker" / "Dockerfile.cuda").exists()


def test_compose_exists() -> None:
    root = Path(__file__).resolve().parent.parent
    assert (root / "docker" / "docker-compose.yaml").exists()
