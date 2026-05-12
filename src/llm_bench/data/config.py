"""Config file IO + log setup."""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional


def setup_logging(log_path: Optional[str]) -> None:
    handlers: list = [logging.StreamHandler(sys.stderr)]
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=handlers)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    if not cfg.get("models"):
        raise SystemExit("config: 'models' must be a non-empty list")
    if not cfg.get("questions"):
        raise SystemExit("config: 'questions' must be a non-empty list")
    if "email" not in cfg:
        raise SystemExit("config: 'email' section is required")
    return cfg
