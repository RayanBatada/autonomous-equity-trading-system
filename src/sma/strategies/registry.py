"""Sleeve name -> Strategy class. config.yaml's `strategies.sleeves[].name`
resolves through here.

Each registered class exposes `from_build(build: SleeveBuild) -> Strategy`,
so a sleeve can pull what it needs (universe, store, settings, a predictor
override for replay) without every constructor taking every argument.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sma.strategies.base import SESSIONS, Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


@dataclass
class SleeveBuild:
    """Everything a sleeve factory may need. Mirrors the arguments the decide
    job has always passed to its strategy builder."""

    universe: list[str]
    use_theses: bool
    db: str
    store: Any  # sma.ingest.store.Store or anything with `.conn`
    settings: Any = None
    predictor: Any = None  # replay swaps in StoredPredictionsPredictor
    # Builder for the incumbent XGBoostTopKStrategy, same signature as
    # sma.strategies.xgb_momentum.build_incumbent_strategy. The decide CLI
    # passes its own `_build_strategy` so there is exactly one seam to patch.
    incumbent_factory: Callable[..., Any] | None = None


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator. Names are unique; sessions must be known."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} has no `name`")
    if cls.session not in SESSIONS:
        raise ValueError(f"{name}: unknown session {cls.session!r}; expected one of {SESSIONS}")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"sleeve name {name!r} already registered by {existing.__name__}")
    _REGISTRY[name] = cls
    return cls


def _load_builtins() -> None:
    # Import for the @register side effect. Kept lazy so importing the
    # registry never drags the model stack in by itself.
    import sma.strategies.xgb_momentum  # noqa: F401


def get(name: str) -> type[Strategy]:
    _load_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown sleeve {name!r}; registered: {known}") from None


def names() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)


def build(name: str, build_ctx: SleeveBuild) -> Strategy:
    cls = get(name)
    factory: Callable[[SleeveBuild], Strategy] | None = getattr(cls, "from_build", None)
    return factory(build_ctx) if factory is not None else cls()
