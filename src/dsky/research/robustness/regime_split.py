"""Regime split: performance by a volatility-regime proxy."""

import math
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from dsky.research.backtest.engine import Fill
from dsky.research.robustness._math import metric_from_contributions, round_trip_contributions

_ROLLING_WINDOW = 5
_MIN_BARS_FOR_VOL = 2


@dataclass(frozen=True)
class RegimeSplitResult:
    """Key metric in low- and high-volatility regimes."""

    low_vol_metric: float
    high_vol_metric: float
    low_vol_trades: int
    high_vol_trades: int


def _vol_by_event_time(bars: pl.DataFrame) -> dict[datetime, float]:
    """Map each bar's ``event_time`` to a trailing realised-vol proxy."""
    if bars.height < _MIN_BARS_FOR_VOL:
        return {}
    sorted_bars = bars.sort("event_time")
    rets = sorted_bars.select(
        (pl.col("close").pct_change()).alias("ret"),
        pl.col("event_time"),
    )
    vols = rets.with_columns(
        pl.col("ret").rolling_std(_ROLLING_WINDOW).alias("vol"),
    )
    out: dict[datetime, float] = {}
    for row in vols.to_dicts():
        et = row["event_time"]
        vol = row["vol"]
        if et is not None and vol is not None and not math.isnan(vol):
            out[et] = float(vol)
    return out


def _nearest_vol(
    event_time: datetime, vol_map: dict[datetime, float],
) -> float | None:
    """Vol at ``event_time`` or the latest bar at or before it."""
    if not vol_map:
        return None
    candidates = [(et, v) for et, v in vol_map.items() if et <= event_time]
    if not candidates:
        return vol_map[min(vol_map)]
    return max(candidates, key=lambda x: x[0])[1]


def analyze_regime_split(
    *,
    trades: list[Fill],
    bars: pl.DataFrame,
    position_size: float,
) -> RegimeSplitResult:
    """Split the holdout by median trailing vol and report each regime's metric.

    The volatility proxy is a 5-bar rolling standard deviation of close-
    to-close returns. Each round trip is assigned to the regime at entry.
    """
    contribs = round_trip_contributions(trades, position_size=position_size)
    vol_map = _vol_by_event_time(bars)
    if not contribs or not vol_map:
        return RegimeSplitResult(0.0, 0.0, 0, 0)

    vols_at_entry = [
        _nearest_vol(et, vol_map) for et, _ in contribs
    ]
    valid = [v for v in vols_at_entry if v is not None]
    if not valid:
        return RegimeSplitResult(0.0, 0.0, 0, 0)

    median_vol = sorted(valid)[len(valid) // 2]
    low: list[tuple[datetime, float]] = []
    high: list[tuple[datetime, float]] = []
    for (et, pnl), vol in zip(contribs, vols_at_entry, strict=True):
        if vol is None:
            continue
        if vol <= median_vol:
            low.append((et, pnl))
        else:
            high.append((et, pnl))

    return RegimeSplitResult(
        low_vol_metric=metric_from_contributions(low),
        high_vol_metric=metric_from_contributions(high),
        low_vol_trades=len(low),
        high_vol_trades=len(high),
    )


__all__ = ["RegimeSplitResult", "analyze_regime_split"]
