"""Replays the event stream, applies strategies, and produces fills, marks, and PnL.

The engine enforces the time-discipline gate: at every decision, the
engine has access to data with ``available_time <= decision_time``. If
a bar with ``available_time > decision_time`` is observed, the engine
raises :class:`LookAheadViolation`.

This is the structural backstop behind ``data.bars.load_bars()``'s
gate: even if a future bug let future data through the data layer,
the engine's own check would catch it and fail loud.
"""
from datetime import datetime


class LookAheadViolation(Exception):  # noqa: N818 -- named per the test spec, not the ruff default
    """Raised when the engine accesses data with available_time > decision_time.

    The backtest is point-in-time: at every decision, the engine has
    access to data with ``available_time <= decision_time``. If a bar
    whose ``available_time`` is strictly greater than the
    ``decision_time`` is observed, the engine's time discipline has
    been violated.

    The test in ``tests/test_lookahead_invariant.py`` enforces this
    invariant.
    """

    def __init__(
        self,
        *,
        decision_time: datetime,
        available_time: datetime,
        symbol: str,
    ) -> None:
        """Build a LookAheadViolation with the offending decision/available times."""
        self.decision_time = decision_time
        self.available_time = available_time
        self.symbol = symbol
        super().__init__(
            f"look-ahead: bar for {symbol!r} has available_time="
            f"{available_time.isoformat()} which is strictly after "
            f"decision_time={decision_time.isoformat()}",
        )


__all__ = ["LookAheadViolation"]
