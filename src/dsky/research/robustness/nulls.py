"""Null comparison: strategy metric vs random-entry null over N seeds.

This module is the **only** place in the robustness suite that reads
``HypothesisSpec.holdout_window``. The holdout slice is consumed exactly
once per robustness run when building the null-model backtest window.
"""
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from dsky.clock import Clock
from dsky.data.bars import next_nyse_open
from dsky.research.backtest.baselines import random_entry_null
from dsky.research.backtest.engine import BacktestSpec
from dsky.research.spec import HypothesisSpec, TimeWindow

# One-sided significance threshold for rejecting a noise strategy.
_P_VALUE_THRESHOLD = 0.05
# Strategy must also clear a minimum material edge over the null cloud.
_MIN_MATERIAL_EDGE = 0.01


@dataclass(frozen=True)
class NullResult:
    """Empirical p-value from comparing strategy to random-entry nulls."""

    strategy_metric: float
    null_metrics: tuple[float, ...]
    p_value: float
    significant: bool
    holdout_start: str
    holdout_end: str


def _holdout_backtest_spec(asset: str, holdout: TimeWindow) -> BacktestSpec:
    """Build a :class:`BacktestSpec` from the holdout date window.

    The only consumer of ``HypothesisSpec.holdout_window`` in the
    robustness package.
    """
    start_day = datetime.fromisoformat(holdout.start).replace(tzinfo=UTC)
    end_day = datetime.fromisoformat(holdout.end).replace(tzinfo=UTC)
    first_decision = next_nyse_open(start_day)
    last_decision = next_nyse_open(end_day)
    return BacktestSpec(
        symbol=asset,
        training_window=(first_decision, last_decision),
    )


def analyze_nulls(  # noqa: PLR0913
    conn: sqlite3.Connection,
    hypothesis_spec: HypothesisSpec,
    clock: Clock,
    *,
    strategy_metric: float,
    trade_count: int,
    avg_holding_period: timedelta,
    n_seeds: int = 50,
) -> NullResult:
    """Compare ``strategy_metric`` to random-entry nulls on the holdout window.

    Reads ``hypothesis_spec.holdout_window`` exactly once to build the
    null backtest slice. For each seed in ``0 .. n_seeds-1``, runs
    :func:`~dsky.research.backtest.baselines.random_entry_null` and
    collects ``total_return``. The empirical one-sided p-value is the
    fraction of null draws with metric ``>= strategy_metric``.

    ``significant`` is ``True`` when ``p_value <= 0.05`` (strategy
    beats the null distribution at the 5% level).
    """
    # --- holdout window: read once, here only ---
    holdout = hypothesis_spec.holdout_window
    backtest_spec = _holdout_backtest_spec(hypothesis_spec.asset, holdout)

    null_metrics: list[float] = []
    for seed in range(n_seeds):
        result = random_entry_null(
            conn,
            spec=backtest_spec,
            clock=clock,
            trade_count=trade_count,
            avg_holding_period=avg_holding_period,
            seed=seed,
        )
        null_metrics.append(result.total_return)

    n = len(null_metrics)
    count_ge = sum(1 for m in null_metrics if m >= strategy_metric)
    p_value = (count_ge + 1) / (n + 1)  # conservative +1/+1 correction
    significant = (
        p_value <= _P_VALUE_THRESHOLD
        and strategy_metric >= _MIN_MATERIAL_EDGE
    )

    return NullResult(
        strategy_metric=strategy_metric,
        null_metrics=tuple(null_metrics),
        p_value=p_value,
        significant=significant,
        holdout_start=holdout.start,
        holdout_end=holdout.end,
    )


__all__ = ["NullResult", "analyze_nulls"]
