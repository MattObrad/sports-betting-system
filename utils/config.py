"""
utils/config.py -- Config file loader for the MLB totals pipeline.

Reads D:\\models\\mlb\\config.json.  All pipeline scripts import from here
so the config path is never duplicated across modules.

Secrets (Twilio auth token, API keys) are NOT stored in config.json.
They are read from environment variables at call time in the modules
that need them.

Usage:
    from utils.config import load_config, cfg_get

    cfg = load_config()
    min_edge = cfg_get(cfg, "betting", "min_edge_runs", default=1.5)
"""

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config(path: str = None) -> dict:
    """
    Load and return the parsed config.json as a nested dict.
    Returns an empty dict if the file is not found (callers use defaults).
    """
    p = Path(path) if path else _CONFIG_PATH
    try:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        log.debug("Config loaded from %s", p)
        return cfg
    except FileNotFoundError:
        log.warning("config.json not found at %s -- using built-in defaults", p)
        return {}
    except json.JSONDecodeError as exc:
        log.error("config.json parse error: %s -- using built-in defaults", exc)
        return {}


def cfg_get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely traverse a nested config dict.

    Example:
        cfg_get(cfg, "betting", "min_edge_runs", default=1.5)
    Returns default if any key is missing or the value is None.
    """
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k)
        if val is None:
            return default
    return val
