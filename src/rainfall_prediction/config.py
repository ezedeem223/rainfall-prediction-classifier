"""Configuration loading helpers."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .utils import ConfigError, resolve_path

ENVIRONMENT_OVERRIDES = {
    "RAINFALL_DATASET_PATH": ("data", "dataset", "path"),
    "RAINFALL_MODEL_PATH": ("paths", "model_path"),
    "RAINFALL_RESULTS_DIR": ("paths", "results_dir"),
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Config file must contain a mapping: {config_path}")
    return payload


def _set_nested_value(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def apply_environment_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply optional environment-variable overrides."""
    for env_name, key_path in ENVIRONMENT_OVERRIDES.items():
        value = os.getenv(env_name)
        if value:
            _set_nested_value(config, key_path, value)
    return config


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a config file and resolve referenced defaults."""
    resolved_path = Path(config_path).resolve()
    raw_config = load_yaml(resolved_path)
    defaults = raw_config.pop("defaults", {})
    base_dir = resolved_path.parent

    merged_config: dict[str, Any] = {}

    base_config = defaults.get("base_config")
    if base_config:
        base_payload = load_config(resolve_path(base_config, base_dir))
        merged_config = deep_merge(merged_config, base_payload)

    data_config = defaults.get("data_config")
    if data_config:
        merged_config = deep_merge(
            merged_config,
            {"data": load_yaml(resolve_path(data_config, base_dir))},
        )

    config = deep_merge(merged_config, raw_config)
    config["meta"] = {"config_path": str(resolved_path)}
    return apply_environment_overrides(config)
