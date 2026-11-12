"""Repo-anchored default paths, so commands work from any working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKBOOK = REPO_ROOT / "Ardook_data.xlsx"
