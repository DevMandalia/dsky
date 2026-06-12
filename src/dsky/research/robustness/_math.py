"""Shared trade-PnL helpers for robustness checks."""

from datetime import datetime

from dsky.research.backtest.engine import Fill, Side


def round_trip_contributions(
    trades: list[Fill],
    *,
    position_size: float,
) -> list[tuple[datetime, float]]:
    """Return ``(entry_decision_time, pnl_fraction)`` per long/flat round trip.

    ``pnl_fraction`` is realized PnL divided by ``position_size`` (same
    scale as :attr:`~dsky.research.backtest.engine.BacktestResult.total_return`).
    """
    if position_size == 0.0:
        return []
    out: list[tuple[datetime, float]] = []
    i = 0
    while i < len(trades):
        if trades[i].side == Side.LONG:
            entry = trades[i]
            j = i + 1
            while j < len(trades) and trades[j].side != Side.FLAT:
                j += 1
            if j < len(trades):
                exit_fill = trades[j]
                pnl = (
                    (exit_fill.price - entry.price) * entry.shares
                    - entry.cost
                    - exit_fill.cost
                )
                out.append((entry.decision_time, pnl / position_size))
            i = j + 1
        else:
            i += 1
    return out


def metric_from_contributions(contributions: list[tuple[datetime, float]]) -> float:
    """Sum of per-round-trip PnL fractions."""
    return sum(pnl for _, pnl in contributions)
