"""Robustness suite: subsample, outlier, regime, and null checks."""

import sqlite3
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import polars as pl

from dsky.clock import Clock
from dsky.db.events import append_event
from dsky.research.backtest.engine import Fill
from dsky.research.robustness.nulls import NullResult, analyze_nulls
from dsky.research.robustness.outlier_drop import OutlierDropResult, analyze_outlier_drop
from dsky.research.robustness.regime_split import RegimeSplitResult, analyze_regime_split
from dsky.research.robustness.subsample import SubsampleResult, analyze_subsample
from dsky.research.spec import HypothesisSpec


@dataclass(frozen=True)
class RobustnessResult:
    """Aggregated output of the full robustness battery."""

    subsample: SubsampleResult
    outlier_drop: OutlierDropResult
    regime_split: RegimeSplitResult
    nulls: NullResult
    passed: bool
    run_event_id: int


def run_robustness_suite(  # noqa: PLR0913
    conn: sqlite3.Connection,
    entity_id: str,
    hypothesis_spec: HypothesisSpec,
    clock: Clock,
    *,
    strategy_metric: float,
    trades: list[Fill],
    bars: pl.DataFrame,
    position_size: float,
    trade_count: int,
    avg_holding_period: timedelta,
    n_null_seeds: int = 50,
    actor: str = "code:robustness",
) -> RobustnessResult:
    """Run the anti-overfitting battery and append ``robustness.run.recorded``.

    The holdout date slice is consumed exactly once inside :mod:`nulls`.
    """
    subsample = analyze_subsample(
        key_metric=strategy_metric,
        trades=trades,
        position_size=position_size,
    )
    outlier = analyze_outlier_drop(
        key_metric=strategy_metric,
        trades=trades,
        position_size=position_size,
    )
    regime = analyze_regime_split(
        trades=trades,
        bars=bars,
        position_size=position_size,
    )
    nulls = analyze_nulls(
        conn,
        hypothesis_spec,
        clock,
        strategy_metric=strategy_metric,
        trade_count=trade_count,
        avg_holding_period=avg_holding_period,
        n_seeds=n_null_seeds,
    )

    passed = (
        not subsample.unstable
        and not outlier.edge_lost
        and nulls.significant
    )

    payload: dict[str, Any] = {
        "entity_id": entity_id,
        "strategy_metric": strategy_metric,
        "subsample": asdict(subsample),
        "outlier_drop": asdict(outlier),
        "regime_split": asdict(regime),
        "nulls": {
            **asdict(nulls),
            "null_metrics": list(nulls.null_metrics),
        },
        "passed": passed,
        "position_size": position_size,
        "trade_count": trade_count,
        "avg_holding_period_seconds": avg_holding_period.total_seconds(),
        "n_null_seeds": n_null_seeds,
    }
    run_event_id = append_event(
        conn=conn,
        event_type="robustness.run.recorded",
        entity_id=entity_id,
        actor=actor,
        payload=payload,
        clock=clock,
    )

    return RobustnessResult(
        subsample=subsample,
        outlier_drop=outlier,
        regime_split=regime,
        nulls=nulls,
        passed=passed,
        run_event_id=run_event_id,
    )


__all__ = [
    "RobustnessResult",
    "run_robustness_suite",
]
