"""Reference strategies: buy-and-hold and seeded random-entry null.

These baselines are the comparison points for any new strategy. A
strategy that can't beat ``buy_and_hold`` over its training window
isn't an edge. A strategy that can't beat the ``random_entry_null``
(matched on trade count and average holding period) on the SAME
data isn't capturing anything beyond random luck.

The random null is the strict one. Given the strategy's trade count
and average holding period, it generates random round-trips within
the same training window and computes the PnL with the same cost
model. Reproducible from the seed.
"""
import random
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from dsky.clock import Clock
from dsky.data.bars import load_bars
from dsky.db.events import append_event
from dsky.research.backtest.costs import CostModel
from dsky.research.backtest.engine import (
    BacktestResult,
    BacktestSpec,
    EquityPoint,
    Fill,
    Side,
)

# Minimum bars needed for any baseline run. Below this, the baseline
# can't produce a meaningful round-trip.
_MIN_BARS_FOR_RUN = 2


def _bars_in_window(
    conn: sqlite3.Connection, spec: BacktestSpec,
) -> pl.DataFrame:
    """All bars whose ``event_time`` is within the spec's training window.

    Uses ``load_bars`` at ``spec.training_window[1]`` plus a few
    days -- by then every bar in the window is known. A daily bar's
    available_time is the next market open, which is up to 4
    calendar days later (over a weekend), so we add a buffer.
    """
    start, end = spec.training_window
    all_bars = load_bars(conn, spec.symbol, end + timedelta(days=4))
    return all_bars.filter(
        (pl.col("event_time") >= start) & (pl.col("event_time") <= end),
    ).sort("event_time")
    ).sort("event_time")


def _write_run_event(  # noqa: PLR0913 -- internal helper, kw-only-args
    *,
    spec: BacktestSpec,
    trades: list[Fill],
    equity_curve: list[EquityPoint],
    total_return: float,
    costs: CostModel,
    position_size: float,
    conn: sqlite3.Connection,
    clock: Clock,
) -> int:
    """Write a ``backtest.run.recorded`` event for a baseline run."""
    payload: dict[str, Any] = {
        "spec_symbol": spec.symbol,
        "training_window": [
            spec.training_window[0].isoformat(),
            spec.training_window[1].isoformat(),
        ],
        "decision_times": [e.decision_time.isoformat() for e in equity_curve],
        "trades": [
            {**asdict(t), "side": t.side.value,
             "bar_event_time": t.bar_event_time.isoformat(),
             "decision_time": t.decision_time.isoformat()}
            for t in trades
        ],
        "equity_curve": [
            {**asdict(e), "decision_time": e.decision_time.isoformat()}
            for e in equity_curve
        ],
        "total_return": total_return,
        "num_trades": len(trades),
        "costs": {
            "commission_per_share": costs.commission_per_share,
            "slippage_bps": costs.slippage_bps,
        },
        "position_size": position_size,
    }
    return append_event(
        conn=conn,
        event_type="backtest.run.recorded",
        entity_id=f"backtest:{spec.symbol}",
        actor="code:baselines",
        payload=payload,
        clock=clock,
    )


def _fill_at_open(  # noqa: PLR0913 -- internal helper, kw-only-args
    *,
    bar: dict[str, Any],
    side: Side,
    position_size: float,
    costs: CostModel,
    cash: float,
    shares: float,
    decision_time: datetime,
) -> tuple[Fill, float, float]:
    """Fill at this bar's open, applying costs. Return (Fill, new_cash, new_shares)."""
    fill_price = float(bar["open"])
    event_time = bar["event_time"]
    if side == Side.LONG:
        fill_shares = position_size / fill_price
        cost = costs.apply(fill_price, fill_shares)
        new_cash = cash - fill_price * fill_shares - cost
        new_shares = fill_shares
    else:  # FLAT -- close the open long
        fill_shares = shares
        cost = costs.apply(fill_price, fill_shares)
        new_cash = cash + fill_price * fill_shares - cost
        new_shares = 0.0
    return (
        Fill(
            side=side,
            bar_event_time=event_time,
            decision_time=decision_time,
            price=fill_price,
            shares=fill_shares,
            cost=cost,
            cash_after=new_cash,
            shares_after=new_shares,
        ),
        new_cash,
        new_shares,
    )


def _empty_result(spec: BacktestSpec) -> BacktestResult:
    return BacktestResult(
        spec=spec, trades=[], equity_curve=[],
        total_return=0.0, num_trades=0, run_event_id=0,
    )


def buy_and_hold(
    conn: sqlite3.Connection,
    *,
    spec: BacktestSpec,
    clock: Clock,
    costs: CostModel | None = None,
    position_size: float = 10_000.0,
) -> BacktestResult:
    """Buy at the first bar's open, close at the last bar's close.

    The simplest baseline: enter at the first available bar's open,
    hold to the last bar's close, and exit. Two fills (entry + exit).
    The return is the gross price-change minus the cost of two fills.
    """
    costs = costs or CostModel()
    bars = _bars_in_window(conn, spec)
    if bars.height < 1:
        return _empty_result(spec)
    first = bars.row(0, named=True)
    last = bars.row(-1, named=True)

    entry_trade, cash, shares = _fill_at_open(
        bar=first, side=Side.LONG, position_size=position_size, costs=costs,
        cash=0.0, shares=0.0, decision_time=spec.training_window[0],
    )
    exit_price = float(last["close"])
    exit_cost = costs.apply(exit_price, shares)
    cash += exit_price * shares - exit_cost
    exit_trade = Fill(
        side=Side.FLAT,
        bar_event_time=last["event_time"],
        decision_time=spec.training_window[1],
        price=exit_price,
        shares=shares,
        cost=exit_cost,
        cash_after=cash,
        shares_after=0.0,
    )
    trades = [entry_trade, exit_trade]
    equity_curve = [
        EquityPoint(
            decision_time=spec.training_window[0],
            cash=cash,
            position_value=0.0,
            total_equity=cash,
        ),
    ]
    total_return = cash / position_size
    run_event_id = _write_run_event(
        spec=spec, trades=trades, equity_curve=equity_curve,
        total_return=total_return, costs=costs, position_size=position_size,
        conn=conn, clock=clock,
    )
    return BacktestResult(
        spec=spec, trades=trades, equity_curve=equity_curve,
        total_return=total_return, num_trades=len(trades),
        run_event_id=run_event_id,
    )


def random_entry_null(  # noqa: PLR0913 -- spec, clock, conn, then 4 strategy params
    conn: sqlite3.Connection,
    *,
    spec: BacktestSpec,
    clock: Clock,
    trade_count: int,
    avg_holding_period: timedelta,
    seed: int,
    costs: CostModel | None = None,
    position_size: float = 10_000.0,
) -> BacktestResult:
    """Seeded random null matched on trade count and average holding period.

    Generates ``trade_count`` round-trips, each starting at a random
    time within the spec's training window (offset by the average
    holding period so the exit fits), held for an exponentially-
    distributed random duration with mean ``avg_holding_period``.
    Fully reproducible: same seed -> same result.

    Each round-trip has TWO fills: a long entry at the entry bar's
    open, and a flat exit at the next bar's open (the same next-bar
    semantics as the engine).
    """
    costs = costs or CostModel()
    rng = random.Random(seed)  # noqa: S311 -- not cryptographic; this is a baseline
    bars = _bars_in_window(conn, spec)
    if bars.height < _MIN_BARS_FOR_RUN:
        return _empty_result(spec)

    bar_rows: list[dict[str, Any]] = list(bars.to_dicts())
    n_bars = len(bar_rows)

    window_start, window_end = spec.training_window
    window_seconds = (window_end - window_start).total_seconds()
    avg_hold_seconds = max(1.0, avg_holding_period.total_seconds())
    latest_entry_offset = max(0.0, window_seconds - avg_hold_seconds)

    entry_offsets = [rng.uniform(0.0, latest_entry_offset) for _ in range(trade_count)]
    hold_seconds = [rng.expovariate(1.0 / avg_hold_seconds) for _ in range(trade_count)]
    pairs = sorted(zip(entry_offsets, hold_seconds, strict=True), key=lambda p: p[0])

    trades: list[Fill] = []
    cash = 0.0
    shares = 0.0
    for entry_offset, hold in pairs:
        entry_time = window_start + timedelta(seconds=entry_offset)
        exit_time = min(entry_time + timedelta(seconds=hold), window_end)

        # The "entry bar" is the first bar whose event_time >= entry_time.
        entry_idx = next(
            (i for i, row in enumerate(bar_rows) if row["event_time"] >= entry_time),
            None,
        )
        if entry_idx is None:
            continue
        entry_bar = bar_rows[entry_idx]
        entry_trade, cash, shares = _fill_at_open(
            bar=entry_bar, side=Side.LONG, position_size=position_size, costs=costs,
            cash=cash, shares=shares, decision_time=entry_time,
        )
        trades.append(entry_trade)

        # Next-bar fill: exit at the bar AFTER the entry bar.
        exit_idx = entry_idx + 1
        if exit_idx >= n_bars:
            # No more bars; force-close at the last bar's close.
            exit_price = float(bar_rows[-1]["close"])
            exit_cost = costs.apply(exit_price, shares)
            cash += exit_price * shares - exit_cost
            trades.append(Fill(
                side=Side.FLAT,
                bar_event_time=bar_rows[-1]["event_time"],
                decision_time=exit_time,
                price=exit_price,
                shares=shares,
                cost=exit_cost,
                cash_after=cash,
                shares_after=0.0,
            ))
            shares = 0.0
            continue
        exit_bar = bar_rows[exit_idx]
        exit_trade, cash, shares = _fill_at_open(
            bar=exit_bar, side=Side.FLAT, position_size=position_size, costs=costs,
            cash=cash, shares=shares, decision_time=exit_time,
        )
        trades.append(exit_trade)

    total_return = cash / position_size
    equity_curve = [
        EquityPoint(
            decision_time=window_start,
            cash=cash,
            position_value=0.0,
            total_equity=cash,
        ),
    ]
    run_event_id = _write_run_event(
        spec=spec, trades=trades, equity_curve=equity_curve,
        total_return=total_return, costs=costs, position_size=position_size,
        conn=conn, clock=clock,
    )
    return BacktestResult(
        spec=spec, trades=trades, equity_curve=equity_curve,
        total_return=total_return, num_trades=len(trades),
        run_event_id=run_event_id,
    )


__all__ = ["buy_and_hold", "random_entry_null"]
