"""AnimaNode subclass for DEF-DTP serving."""

from __future__ import annotations

from anima_def_dtp.attack.engine import DtpAttackEngine
from anima_def_dtp.config import get_settings


class DefDtpServeNode:
    """Lightweight serve node that wraps the attack engine."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine = DtpAttackEngine(self.settings)
        self._ready = True

    def setup_inference(self) -> None:
        """Load any required model weights."""
        pass  # attack engine is CPU-only by default

    def get_status(self) -> dict:
        return {
            "module": "def_dtp",
            "ready": self._ready,
            "paper_arxiv": self.settings.paper_arxiv,
        }

    @property
    def ready(self) -> bool:
        return self._ready
