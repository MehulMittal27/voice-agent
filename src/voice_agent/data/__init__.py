"""Recognition data-provider package."""

from __future__ import annotations

from typing import Any

from .base import Authority, DataProvider, Document, Occupation

__all__ = [
    "Authority",
    "DataProvider",
    "Document",
    "MockDataProvider",
    "Occupation",
]


def __getattr__(name: str) -> Any:
    if name == "MockDataProvider":
        from .mock import MockDataProvider

        return MockDataProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
