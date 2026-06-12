"""The look-ahead invariant: the engine must never observe a bar
whose ``available_time`` exceeds the engine's current ``decision_time``.

This test exercises the engine's data-access contract: at every
decision point the engine may only see bars via
``load_bars(symbol, decision_time)``.

The invariant lives in two layers:

  1. The data layer (``data.bars.load_bars``) is the primary
     defence -- it only returns rows with
     ``available_time <= decision_time``.

  2. The engine's own defensive check is the backstop -- after every
     ``load_bars`` call, it scans the result and raises
     :class:`LookAheadViolation` if any row has
     ``available_time > decision_time``. This test exercises that
     backstop by running the engine in a normal scenario and asserting
     it completes without raising.
"""
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import polars as pl

from dsky.clock import FrozenClock
from dsky.data.bars import write_bars
from dsky.db.engine import open_db
from dsky.db.projections import rebuild_projections
from dsky.research.backtest.engine import BacktestEngine, BacktestSpec, LookAheadViolation

# A frozen "now" well after the 2026-01-13 14:30 UTC open (the first
# decision time below) so the clock is sane and the manifest event
# has a stable ts.
_T0 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)

# A trivial strategy type alias -- the spec is the SHAPE of the
# callable, not its logic. The engine is the unit under test; the
# strategy is just whatever the engine is running.
Strategy = Callable[[dict[str, Any], datetime, pl.DataFrame], Any]


def _synthetic_spy_bars() -> pl.DataFrame:
    """3 daily bars for SPY, Mon/Tue/Wed 2026-01-12/13/14.

    Using the default ``next_nyse_open`` rule, the bars become
    "available" at:

      Mon bar -> Tue  9:30 ET (14:30 UTC) = 2026-01-13 14:30 UTC
      Tue bar -> Wed  9:30 ET (14:30 UTC) = 2026-01-14 14:30 UTC
      Wed bar -> Thu  9:30 ET (14:30 UTC) = 2026-01-15 14:30 UTC
    """
    return pl.DataFrame({
        "event_time": [
            datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 13, 0, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 14, 0, 0, 0, tzinfo=UTC),
        ],
        "open":   [450.0, 452.0, 451.0],
        "high":   [455.0, 456.0, 454.0],
        "low":    [449.0, 451.0, 450.0],
        "close":  [454.0, 455.0, 453.0],
        "volume": [100_000_000, 110_000_000, 105_000_000],
    })


def _setup_synthetic_data(conn: object, parquet_path: object) -> None:
    """Write 3 synthetic SPY bars to a fresh in-memory DB + Parquet file."""
    bars = _synthetic_spy_bars()
    write_bars(
        conn=conn,  # type: ignore[arg-type]
        symbol="SPY",
        bars=bars,
        parquet_path=parquet_path,  # type: ignore[arg-type]
        vendor="synthetic",
        date_start="2026-01-12",
        date_end="2026-01-14",
        clock=FrozenClock(at=_T0),
    )
    rebuild_projections(conn)  # type: ignore[arg-type]


def _trivial_strategy(state: dict[str, Any], decision_time: datetime, bars: pl.DataFrame) -> str:
    """An innocent strategy: always hold.

    The strategy is intentionally trivial in its action -- the test
    is about the engine's data access, not the strategy's logic.
    The strategy is here to exercise the engine's iteration loop.
    """
    return "hold"


class TestEngineLookAheadInvariant:
    """The engine's time discipline.

    At every decision, the engine has access to data with
    ``available_time <= decision_time``. If a bar with
    ``available_time > decision_time`` is observed, the engine
    raises :class:`LookAheadViolation`.

    """

    def test_engine_never_requests_a_bar_with_future_available_time(
        self, tmp_path: object,
    ) -> None:
        """The engine must not violate the time discipline in a normal run.

        The engine iterates 3 decision times (one per market open). At
        each decision, the engine calls ``load_bars(symbol, decision_time)``
        and the data layer returns only rows whose
        ``available_time <= decision_time``. The engine's own defensive
        check confirms this and would raise :class:`LookAheadViolation`
        if a violation slipped through.

        If the engine is implemented correctly, this test runs to
        completion with no exception. If the engine ever asks for
        future data, the test catches the violation.
        """
        # The test runs the engine. The engine enforces the
        # available_time <= decision_time gate both via the data
        # layer (load_bars) and its own defensive backstop
        # (_check_no_look_ahead). A correct engine runs to
        # completion with no exception. If the engine ever asks
        # for future data, LookAheadViolation is raised.
        # Synthetic data: 3 daily bars with available_time = next market open.
        conn = open_db(":memory:")
        _setup_synthetic_data(conn, tmp_path / "SPY.parquet")  # type: ignore[operator]
        clock = FrozenClock(at=_T0)

        engine = BacktestEngine(conn=conn, clock=clock)

        # 3 decision times -- one per market open. At each decision,
        # the engine should see only the bars whose available_time is
        # <= the decision_time:
        #   decision 1 (Tue 14:30)  -> Mon bar visible (avail Tue 14:30, == OK)
        #   decision 2 (Wed 14:30)  -> Mon + Tue bars visible
        #   decision 3 (Thu 14:30)  -> Mon + Tue + Wed bars visible
        decision_times = [
            datetime(2026, 1, 13, 14, 30, 0, tzinfo=UTC),
            datetime(2026, 1, 14, 14, 30, 0, tzinfo=UTC),
            datetime(2026, 1, 15, 14, 30, 0, tzinfo=UTC),
        ]

        spec = BacktestSpec(
            symbol="SPY",
            training_window=(decision_times[0], decision_times[-1]),
        )

        result = engine.run(
            strategy=_trivial_strategy,
            spec=spec,
            decision_times=decision_times,
        )

        # If the engine raised LookAheadViolation, we'd never reach
        # here. The assertion is intentionally minimal: the test's
        # value is in NOT raising. The strategy is trivial; the
        # invariant is in the engine.
        assert result is not None

    def test_lookahead_violation_carries_the_offending_times(self) -> None:
        """The exception must carry the decision_time, available_time, and symbol.

        Part of the engine's contract: when a violation is raised, the
        diagnostic message must include the exact decision_time and
        the offending available_time, so the bug is debuggable from a
        stack trace.
        """
        decision_time = datetime(2026, 1, 14, 14, 30, 0, tzinfo=UTC)
        future = datetime(2026, 1, 15, 14, 30, 0, tzinfo=UTC)

        violation = LookAheadViolation(
            decision_time=decision_time,
            available_time=future,
            symbol="SPY",
        )
        assert violation.decision_time == decision_time
        assert violation.available_time == future
        assert violation.symbol == "SPY"
        assert "SPY" in str(violation)
        assert decision_time.isoformat() in str(violation)
        assert future.isoformat() in str(violation)
