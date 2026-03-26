"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def load_config(config_path: str | Path) -> Dict:
    """Load yaml config into dict."""
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)
