"""Backtest engine, cost model, and baseline strategies."""

from dsky.research.backtest.baselines import buy_and_hold, random_entry_null
from dsky.research.backtest.costs import CostModel
from dsky.research.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    BacktestSpec,
    LookAheadViolation,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestSpec",
    "CostModel",
    "LookAheadViolation",
    "buy_and_hold",
    "random_entry_null",
]
