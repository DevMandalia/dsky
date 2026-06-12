"""Tests for the backtest engine.

The headline tests are the two known-answer assertions:

* :class:`TestEngineKnownAnswer` -- a synthetic series with a planted
  drift. The strategy captures the drift. The engine's return must
  match the analytical answer to within rounding.

* :class:`TestEngineFlatSeries` -- a flat (zero-drift) series. The
  strategy captures nothing. The engine's return must equal exactly
  the cost (the strategy loses precisely the cost of two fills).

If the engine's fill semantics or cost model is wrong, these tests
will fail with a specific numerical mismatch -- they're not
redundantly testing the cost model (that's ``test_costs``) or
the data gate (that's ``test_lookahead_invariant``); they're
testing the engine's arithmetic end-to-end.
"""
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

from dsky.clock import FrozenClock
from dsky.data.bars import write_bars
from dsky.db.engine import open_db
from dsky.db.projections import rebuild_projections
from dsky.research.backtest import engine as engine_module
from dsky.research.backtest.costs import CostModel
from dsky.research.backtest.engine import (
    BacktestEngine,
    BacktestSpec,
    LookAheadViolation,
    Side,
)

# A frozen "now" anchored well above the test's decision times so the
# clock is sane and the manifest events have a stable ts.
_T0 = datetime(2026, 1, 22, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_db() -> Iterator[Any]:
    """Fresh in-memory DB with the dsky schema applied."""
    conn = open_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(at=_T0)


def _make_spy_bars(prices: list[float]) -> pl.DataFrame:
    """Build N daily SPY bars from a list of close prices.

    All bars use OHLC = close, volume = 1M (costs and equity math
    only need the open and close; OHLC = close is fine).
    """
    n = len(prices)
    return pl.DataFrame({
        "event_time": [
            datetime(2026, 1, 12 + i, 0, 0, 0, tzinfo=UTC) for i in range(n)
        ],
        "open":   list(prices),
        "high":   list(prices),
        "low":    list(prices),
        "close":  list(prices),
        "volume": [1_000_000] * n,
    })


def _write_spy(conn: Any, parquet_path: Any, prices: list[float]) -> None:
    """Write SPY bars to the DB and rebuild projections."""
    bars = _make_spy_bars(prices)
    write_bars(
        conn=conn, symbol="SPY", bars=bars, parquet_path=parquet_path,
        vendor="synthetic",
        date_start="2026-01-12", date_end=f"2026-01-{11 + len(prices)}",
        clock=FrozenClock(at=_T0),
    )
    rebuild_projections(conn)


def _decision_times_for(n: int) -> list[datetime]:
    """Build N decision times, one per weekday market open starting at day 2.

    Skips weekends so the last decision is on a weekday and sees
    the last bar. Without the skip, a decision on Saturday 14:30
    would NOT see a Friday bar (its ``available_time`` is the next
    Monday 14:30 -- the weekend-skip in :func:`next_nyse_open`).
    """
    out: list[datetime] = []
    day = 13
    while len(out) < n:
        dt = datetime(2026, 1, day, 14, 30, 0, tzinfo=UTC)
        if dt.weekday() < 5:  # Mon=0..Fri=4 are trading days
            out.append(dt)
        day += 1
    return out


def _spec(decision_times: list[datetime]) -> BacktestSpec:
    return BacktestSpec(
        symbol="SPY",
        training_window=(decision_times[0], decision_times[-1]),
    )


# ---------------------------------------------------------------------------
# Known-answer test: planted drift
# ---------------------------------------------------------------------------

class TestEngineKnownAnswer:
    """A planted edge: prices rise linearly. The strategy captures it."""

    def test_engine_recovers_planted_uptrend_at_roughly_the_correct_magnitude(
        self, mem_db, tmp_path, clock,
    ) -> None:
        # 5 daily bars priced 100, 102, 104, 106, 108 -- roughly 7.7% drift.
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        decision_times = _decision_times_for(5)

        # Strategy: go long on the first decision, go flat on the
        # fourth (so the next-bar fill at the fifth decision closes
        # the position at day 5's open).
        def planted_strategy(state: dict[str, Any], _dt: datetime, _bars: pl.DataFrame) -> str:
            state["count"] = state.get("count", 0) + 1
            if state["count"] == 1:
                return "long"
            if state["count"] == 4:
                return "flat"
            return "hold"

        engine = BacktestEngine(
            conn=mem_db, clock=clock,
            costs=CostModel(commission_per_share=0.005, slippage_bps=5.0),
            position_size=10_000.0,
        )
        result = engine.run(
            spec=_spec(decision_times),
            strategy=planted_strategy,
            decision_times=decision_times,
        )

        # Analytical expected return. Buy at day 2 open, sell at day 5
        # open; shares are position_size divided by buy price. Costs
        # are commission_per_share times shares plus slippage_bps
        # times notional divided by 10_000, applied on both fills.
        buy_price = 102.0
        sell_price = 108.0
        shares = 10_000.0 / buy_price
        buy_cost = 0.005 * shares + 5.0 * buy_price * shares / 10_000.0
        sell_cost = 0.005 * shares + 5.0 * sell_price * shares / 10_000.0
        gross_pnl = (sell_price - buy_price) * shares
        expected_return = (gross_pnl - buy_cost - sell_cost) / 10_000.0

        # The engine should land within a fraction of a basis point of
        # the analytical answer. Any larger gap means a bug in the
        # fill, cost, or mark-to-market arithmetic.
        assert result.total_return == pytest.approx(expected_return, abs=1e-6), (
            f"planted-edge return {result.total_return:.6f} "
            f"differs from analytical {expected_return:.6f}"
        )
        # Sanity: gross return is ~5.88%, net of costs ~5.77%, so
        # the result should be in that ballpark -- the test would be
        # useless if it accepted "0% minus costs" or "10% minus costs".
        assert 0.05 < result.total_return < 0.07

    def test_engine_writes_a_backtest_run_event(
        self, mem_db, tmp_path, clock,
    ) -> None:
        """The run must produce a ``backtest.run.recorded`` event in the log."""
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        decision_times = _decision_times_for(5)

        def strategy(state: dict[str, Any], _dt: datetime, _bars: pl.DataFrame) -> str:
            state["count"] = state.get("count", 0) + 1
            if state["count"] == 1:
                return "long"
            if state["count"] == 4:
                return "flat"
            return "hold"

        engine = BacktestEngine(conn=mem_db, clock=clock)
        result = engine.run(
            spec=_spec(decision_times),
            strategy=strategy,
            decision_times=decision_times,
        )

        # The result's run_event_id must point to a real event in the log.
        row = mem_db.execute(
            "SELECT event_type, payload FROM events WHERE id = ?",
            (result.run_event_id,),
        ).fetchone()
        assert row["event_type"] == "backtest.run.recorded"
        payload = json.loads(row["payload"])
        assert payload["spec_symbol"] == "SPY"
        assert payload["num_trades"] == len(result.trades) == 2
        assert payload["trades"][0]["side"] == Side.LONG.value
        assert payload["trades"][1]["side"] == Side.FLAT.value
        assert len(payload["equity_curve"]) == 5  # one point per decision


# ---------------------------------------------------------------------------
# Flat-series test: zero drift, return should be -2 * cost / position_size
# ---------------------------------------------------------------------------

class TestEngineFlatSeries:
    """A flat series: gross PnL is zero; net PnL is minus costs."""

    def test_engine_on_flat_series_returns_approximately_minus_costs(
        self, mem_db, tmp_path, clock,
    ) -> None:
        # 5 daily bars, all priced 100. No drift.
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        decision_times = _decision_times_for(5)

        def strategy(state: dict[str, Any], _dt: datetime, _bars: pl.DataFrame) -> str:
            state["count"] = state.get("count", 0) + 1
            if state["count"] == 1:
                return "long"
            if state["count"] == 4:
                return "flat"
            return "hold"

        engine = BacktestEngine(
            conn=mem_db, clock=clock,
            costs=CostModel(commission_per_share=0.005, slippage_bps=5.0),
            position_size=10_000.0,
        )
        result = engine.run(
            spec=_spec(decision_times),
            strategy=strategy,
            decision_times=decision_times,
        )

        # Gross PnL is zero, so net PnL is exactly minus the cost of
        # two fills. The return is that divided by position size.
        shares = 10_000.0 / 100.0
        cost_per_fill = 0.005 * shares + 5.0 * 100.0 * shares / 10_000.0
        expected_return = -2.0 * cost_per_fill / 10_000.0

        assert result.total_return == pytest.approx(expected_return, abs=1e-6), (
            f"flat-series return {result.total_return:.6f} "
            f"differs from minus costs {expected_return:.6f}; "
            f"costs are not biting as expected"
        )
        # Sanity: should be a small negative number, not zero.
        # If this is zero or positive, costs aren't being applied.
        assert result.total_return < 0.0
        # And it should be in the right ballpark: ~ minus 0.11%.
        assert -0.002 < result.total_return < -0.0001


# ---------------------------------------------------------------------------
# Next-bar fill semantics (no same-bar peeking)
# ---------------------------------------------------------------------------

class TestEngineNextBarFills:
    """Orders fill at the NEXT bar's open, not the bar visible at decision time."""

    def test_long_fill_uses_next_bar_open_not_current_bar_open(
        self, mem_db, tmp_path, clock,
    ) -> None:
        # Distinct opens make same-bar vs next-bar fills obvious.
        prices = [100.0, 200.0, 300.0, 400.0, 500.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        decision_times = _decision_times_for(5)

        def go_long_once(state: dict[str, Any], _dt: datetime, _bars: pl.DataFrame) -> str:
            state["done"] = state.get("done", False)
            if not state["done"]:
                state["done"] = True
                return "long"
            return "hold"

        result = BacktestEngine(conn=mem_db, clock=clock, position_size=10_000.0).run(
            spec=_spec(decision_times),
            strategy=go_long_once,
            decision_times=decision_times,
        )

        assert len(result.trades) == 1
        entry = result.trades[0]
        assert entry.side == Side.LONG
        # Decision 1 sees bar 1 (open=100). Fill must be bar 2 open=200.
        assert entry.price == 200.0
        assert entry.price != prices[0]


# ---------------------------------------------------------------------------
# Spec enforcement
# ---------------------------------------------------------------------------

class TestEngineSpecEnforcement:
    """The engine refuses to run if decision times fall outside the spec's window."""

    def test_decision_time_before_spec_window_raises(self, mem_db, clock) -> None:
        spec = BacktestSpec(
            symbol="SPY",
            training_window=(datetime(2026, 1, 14, tzinfo=UTC), datetime(2026, 1, 20, tzinfo=UTC)),
        )
        engine = BacktestEngine(conn=mem_db, clock=clock)
        with pytest.raises(ValueError, match="outside the spec"):
            engine.run(
                spec=spec,
                strategy=lambda s, _t, _b: "hold",
                decision_times=[datetime(2026, 1, 13, tzinfo=UTC)],  # before the window
            )

    def test_decision_time_after_spec_window_raises(self, mem_db, clock) -> None:
        spec = BacktestSpec(
            symbol="SPY",
            training_window=(datetime(2026, 1, 14, tzinfo=UTC), datetime(2026, 1, 20, tzinfo=UTC)),
        )
        engine = BacktestEngine(conn=mem_db, clock=clock)
        with pytest.raises(ValueError, match="outside the spec"):
            engine.run(
                spec=spec,
                strategy=lambda s, _t, _b: "hold",
                decision_times=[datetime(2026, 1, 21, tzinfo=UTC)],  # after the window
            )


# ---------------------------------------------------------------------------
# Engine's defensive backstop
# ---------------------------------------------------------------------------

class TestEngineLookAheadBackstop:
    """The engine scans every bars return and raises if the data layer's gate ever breaks."""

    def test_engine_raises_look_ahead_violation_on_future_data(
        self, mem_db, clock,
    ) -> None:
        """Simulate a buggy data layer by patching load_bars to return a future row."""
        spec = BacktestSpec(
            symbol="SPY",
            training_window=(datetime(2026, 1, 13, tzinfo=UTC), datetime(2026, 1, 15, tzinfo=UTC)),
        )
        engine = engine_module.BacktestEngine(conn=mem_db, clock=clock)

        # Patch the engine's local import (not the bars module's
        # reference) -- the engine imported ``load_bars`` at the top
        # of its own module, so we patch that bound name.
        decision_time = datetime(2026, 1, 14, 14, 30, 0, tzinfo=UTC)
        future_avail = datetime(2026, 1, 15, 14, 30, 0, tzinfo=UTC)  # 1 day later
        bad_bars = pl.DataFrame({
            "event_time": [datetime(2026, 1, 13, 0, 0, 0, tzinfo=UTC)],
            "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0],
            "volume": [1_000_000],
            "available_time": [future_avail],
        })

        original = engine_module.load_bars
        engine_module.load_bars = lambda *_a, **_kw: bad_bars  # type: ignore[assignment]
        try:
            with pytest.raises(LookAheadViolation) as excinfo:
                engine.run(
                    spec=spec,
                    strategy=lambda s, _t, _b: "hold",
                    decision_times=[decision_time],
                )
        finally:
            engine_module.load_bars = original  # type: ignore[assignment]

        assert excinfo.value.decision_time == decision_time
        assert excinfo.value.available_time == future_avail
        assert excinfo.value.symbol == "SPY"
