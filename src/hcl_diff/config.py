from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigurationError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {source}")
    return config


def require(config: dict[str, Any], keys: Iterable[str]) -> None:
    missing: list[str] = []
    for dotted in keys:
        value: Any = config
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                missing.append(dotted)
                break
            value = value[part]
        else:
            if value is None:
                missing.append(dotted)
    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise ConfigurationError(f"Required configuration values are missing: {joined}")


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping")
    return value

