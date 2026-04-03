"""Runtime guards for missing assets and configuration errors."""

from __future__ import annotations

from pathlib import Path

from anima_def_dtp.config import DefDtpSettings


class AssetError(RuntimeError):
    """Raised when a required asset is missing or invalid."""


def check_dataset_path(path: Path, label: str = "dataset") -> None:
    """Raise :class:`AssetError` if *path* does not exist or is empty."""
    if not path.exists():
        raise AssetError(f"{label} path does not exist: {path}")
    if path.is_dir() and not any(path.iterdir()):
        raise AssetError(f"{label} directory is empty: {path}")


def check_checkpoint_path(path: Path, label: str = "checkpoint") -> None:
    """Raise :class:`AssetError` if the checkpoint file is missing."""
    if not path.exists():
        raise AssetError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise AssetError(f"{label} is zero-length: {path}")


def preflight(settings: DefDtpSettings) -> list[str]:
    """Run all preflight checks and return a list of warning strings.

    Raises :class:`AssetError` for hard blockers.
    """
    warnings: list[str] = []
    rt = settings.runtime
    for label, path in [
        ("pretrained_root_linux", rt.pretrained_root_linux),
        ("dataset_root_linux", rt.dataset_root_linux),
    ]:
        if not path.exists():
            warnings.append(f"[WARN] {label} not found: {path}")
    if settings.paper_arxiv != "2603.26462":
        raise AssetError(
            f"Paper arXiv ID mismatch: expected 2603.26462, got {settings.paper_arxiv}"
        )
    return warnings
