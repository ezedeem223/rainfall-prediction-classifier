"""Utility helpers for the rainfall prediction project."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class RainfallPredictionError(Exception):
    """Base exception for project-specific failures."""


class ConfigError(RainfallPredictionError):
    """Raised when configuration loading fails."""


class DataNotFoundError(RainfallPredictionError):
    """Raised when the expected dataset is missing."""


class ModelArtifactNotFoundError(RainfallPredictionError):
    """Raised when the trained model artifact is missing."""


class MissingDependencyError(RainfallPredictionError):
    """Raised when an optional model dependency is unavailable."""


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(path_value: str | Path, base_path: str | Path | None = None) -> Path:
    """Resolve a path against an optional base directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_path is None:
        return path
    return Path(base_path).joinpath(path).resolve()


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(data: Mapping[str, Any] | list[Any], path: str | Path) -> Path:
    """Write JSON to disk with a stable UTF-8 encoding."""
    destination = Path(path)
    ensure_directory(destination.parent)
    destination.write_text(
        json.dumps(_normalize_for_json(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


def write_text(text: str, path: str | Path) -> Path:
    """Write plain text content to disk."""
    destination = Path(path)
    ensure_directory(destination.parent)
    destination.write_text(text, encoding="utf-8")
    return destination
