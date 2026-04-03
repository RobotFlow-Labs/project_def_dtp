"""Typed settings for DEF-DTP."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import DATASET_THRESHOLDS, PAPER_ARXIV_ID, PAPER_TITLE


class AttackSettings(BaseModel):
    max_iter: int = 1000
    orthogonal_step: float = 1.0
    forward_step: float = 0.1
    orthogonal_decay: float = 0.95
    forward_decay: float = 0.9
    tolerance: float = 1e-6


class RuntimePaths(BaseModel):
    pretrained_root_linux: Path = Path("/mnt/forge-data/models/def_dtp")
    dataset_root_linux: Path = Path("/mnt/forge-data/datasets/def_dtp")
    pretrained_root_mac: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/models/def_dtp")
    dataset_root_mac: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/def_dtp")


class DefDtpSettings(BaseSettings):
    """Environment-driven runtime settings layered over static project config."""

    model_config = SettingsConfigDict(env_prefix="ANIMA_DEF_DTP_", extra="ignore")

    project_name: str = "anima-def-dtp"
    codename: str = "DEF-DTP"
    functional_name: str = "DEF-dtp"
    paper_arxiv: str = PAPER_ARXIV_ID
    paper_title: str = PAPER_TITLE
    backend: Literal["auto", "mlx", "cuda", "cpu"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    torch_device: str = "auto"
    attack: AttackSettings = Field(default_factory=AttackSettings)
    runtime: RuntimePaths = Field(default_factory=RuntimePaths)
    config_path: Path = Path("configs/default.toml")

    @property
    def dataset_thresholds(self):
        return DATASET_THRESHOLDS


@lru_cache(maxsize=1)
def get_settings() -> DefDtpSettings:
    """Return cached settings instance."""

    return DefDtpSettings()
