from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml


def configure_logging(config_path: Path | None = None, level: str | None = None) -> None:
    """Configure logging from YAML when available, else fall back to basic config."""

    if config_path and config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        logging.config.dictConfig(data)
    else:
        logging.basicConfig(
            level=getattr(logging, (level or "INFO").upper(), logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )


def get_logger(name: str, **extra: Any) -> logging.LoggerAdapter[logging.Logger]:
    return logging.LoggerAdapter(logging.getLogger(name), extra)

