"""Tests for the backtest baselines.

* :func:`buy_and_hold` is a known formula: gross price-change minus
  two fills of cost. The test asserts the analytical answer.

* :func:`random_entry_null` is a seeded random walk. The test asserts
  (a) the same seed yields the same result (reproducibility), and
  (b) the trade count and average holding period match the request
  -- the "apples-to-apples" claim.
"""
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest

from dsky.clock import FrozenClock
from dsky.data.bars import write_bars
from dsky.db.engine import open_db
from dsky.db.projections import rebuild_projections
from dsky.research.backtest.baselines import buy_and_hold, random_entry_null
from dsky.research.backtest.costs import CostModel
from dsky.research.backtest.engine import BacktestSpec, Side

_T0 = datetime(2026, 1, 22, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def mem_db() -> Iterator[Any]:
    conn = open_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(at=_T0)


def _make_uptrend_bars(prices: list[float]) -> pl.DataFrame:
    """N daily bars with OHLC=close and a deterministic volume.

    Bars start at 2026-01-12 and step one day at a time -- ``timedelta``
    handles month overflow so this works for ``len(prices) > 19``
    (where simple ``+ i`` on the day would overflow January).
    """
    n = len(prices)
    start = datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC)
    return pl.DataFrame({
        "event_time": [start + timedelta(days=i) for i in range(n)],
        "open":   list(prices),
        "high":   list(prices),
        "low":    list(prices),
        "close":  list(prices),
        "volume": [1_000_000] * n,
    })


def _write_spy(conn: Any, parquet_path: Any, prices: list[float]) -> None:
    bars = _make_uptrend_bars(prices)
    write_bars(
        conn=conn, symbol="SPY", bars=bars, parquet_path=parquet_path,
        vendor="synthetic",
        date_start="2026-01-12", date_end=f"2026-01-{11 + len(prices)}",
        clock=FrozenClock(at=_T0),
    )
    rebuild_projections(conn)


def _spec(prices: list[float]) -> BacktestSpec:
    """Build a spec covering the bars in the price list."""
    n = len(prices)
    return BacktestSpec(
        symbol="SPY",
        training_window=(
            datetime(2026, 1, 12, tzinfo=UTC),  # first bar's event_time
            datetime(2026, 1, 11 + n, tzinfo=UTC),  # last bar's event_time
        ),
    )


# ---------------------------------------------------------------------------
# buy-and-hold
# ---------------------------------------------------------------------------

class TestBuyAndHold:
    """Two fills: enter at first bar's open, exit at last bar's close."""

    def test_buy_and_hold_on_uptrend_yields_gross_minus_costs(
        self, mem_db, tmp_path, clock,
    ) -> None:
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        spec = _spec(prices)

        costs = CostModel(commission_per_share=0.005, slippage_bps=5.0)
        result = buy_and_hold(
            mem_db, spec=spec, clock=clock, costs=costs, position_size=10_000.0,
        )

        # Analytical: buy at first bar's open (100), close at last bar's
        # close (108). shares = 10_000/100 = 100. Each fill's cost
        # uses its own fill price (exit is at 108, not 100).
        buy_price = 100.0
        sell_price = 108.0
        shares = 10_000.0 / buy_price
        buy_cost = costs.apply(buy_price, shares)
        sell_cost = costs.apply(sell_price, shares)
        expected_return = (
            (sell_price - buy_price) * shares - buy_cost - sell_cost
        ) / 10_000.0

        assert result.total_return == pytest.approx(expected_return, abs=1e-6)
        assert result.num_trades == 2
        # 7.86% gross - 0.11% costs = ~7.75% net
        assert 0.07 < result.total_return < 0.08

    def test_buy_and_hold_on_flat_series_yields_approximately_minus_costs(
        self, mem_db, tmp_path, clock,
    ) -> None:
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        spec = _spec(prices)

        costs = CostModel(commission_per_share=0.005, slippage_bps=5.0)
        result = buy_and_hold(
            mem_db, spec=spec, clock=clock, costs=costs, position_size=10_000.0,
        )

        # Gross PnL = 0. Net PnL = -2 * cost.
        shares = 10_000.0 / 100.0
        cost_per_fill = costs.apply(100.0, shares)
        expected_return = -2.0 * cost_per_fill / 10_000.0

        assert result.total_return == pytest.approx(expected_return, abs=1e-6)
        assert result.total_return < 0.0

    def test_buy_and_hold_writes_a_run_event(
        self, mem_db, tmp_path, clock,
    ) -> None:
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        spec = _spec(prices)

        result = buy_and_hold(mem_db, spec=spec, clock=clock)
        assert result.run_event_id > 0


# ---------------------------------------------------------------------------
# random_entry_null
# ---------------------------------------------------------------------------

class TestRandomEntryNull:
    """Seeded random: reproducible, trade count and holding period matched."""

    def test_random_null_is_reproducible_from_seed(
        self, mem_db, tmp_path, clock,
    ) -> None:
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        spec = _spec(prices)

        result_a = random_entry_null(
            mem_db, spec=spec, clock=clock,
            trade_count=3, avg_holding_period=timedelta(days=2), seed=42,
        )
        result_b = random_entry_null(
            mem_db, spec=spec, clock=clock,
            trade_count=3, avg_holding_period=timedelta(days=2), seed=42,
        )
        assert result_a.total_return == result_b.total_return
        assert len(result_a.trades) == len(result_b.trades)
        # And the same seed means the actual trades match.
        for ta, tb in zip(result_a.trades, result_b.trades, strict=True):
            assert ta.price == tb.price
            assert ta.decision_time == tb.decision_time

    def test_random_null_with_different_seeds_differs(
        self, mem_db, tmp_path, clock,
    ) -> None:
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        spec = _spec(prices)

        result_a = random_entry_null(
            mem_db, spec=spec, clock=clock,
            trade_count=3, avg_holding_period=timedelta(days=2), seed=42,
        )
        result_b = random_entry_null(
            mem_db, spec=spec, clock=clock,
            trade_count=3, avg_holding_period=timedelta(days=2), seed=43,
        )
        # Different seeds: at least one of the trade prices must differ.
        prices_a = [t.price for t in result_a.trades]
        prices_b = [t.price for t in result_b.trades]
        assert prices_a != prices_b

    def test_random_null_trade_count_matches_request(
        self, mem_db, tmp_path, clock,
    ) -> None:
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        spec = _spec(prices)

        # 5 round-trips -> 10 fills (5 entries + 5 exits, with a
        # possible 11th if the last round-trip force-closes).
        result = random_entry_null(
            mem_db, spec=spec, clock=clock,
            trade_count=5, avg_holding_period=timedelta(days=2), seed=42,
        )
        # We allow 10 or 11 (the last round-trip might force-close at
        # the last bar's close if the next-bar index is out of range).
        assert len(result.trades) in (10, 11)
        # And the trade_count in the event is the same.
        assert result.num_trades == len(result.trades)

    def test_random_null_avg_holding_period_matches_request(
        self, mem_db, tmp_path, clock,
    ) -> None:
        """The realized average holding period (over enough samples) is close to the request.

        For a small sample (5 round-trips) with exponential
        distribution, the standard deviation of the sample mean is
        large. We use a generous tolerance: ±50% of the request.
        """
        prices = [100.0] * 30
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        # Build a spec covering 30 days (~6 weeks of trading days).
        spec = BacktestSpec(
            symbol="SPY",
            training_window=(
                datetime(2026, 1, 12, tzinfo=UTC),
                datetime(2026, 2, 10, tzinfo=UTC),
            ),
        )
        # Use a fixed seed for reproducibility (matches the seed
        # passed to random_entry_null below).
        request_hold = timedelta(days=2)
        result = random_entry_null(
            mem_db, spec=spec, clock=clock,
            trade_count=10, avg_holding_period=request_hold, seed=7,
        )
        # Compute holding periods from the trade pairs (entry/exit).
        rounds: list[tuple[datetime, datetime]] = [
            (result.trades[i].decision_time, result.trades[i + 1].decision_time)
            for i in range(0, len(result.trades), 2)
        ]
        actual_avg_seconds = sum(
            (ex - en).total_seconds() for en, ex in rounds
        ) / max(1, len(rounds))
        # Generous tolerance: plus/minus 50% of the request, since
        # the sample mean of 10 exponentials is noisy.
        assert (
            abs(actual_avg_seconds - request_hold.total_seconds())
            < 0.5 * request_hold.total_seconds()
        )

    def test_random_null_uses_position_size_consistently(
        self, mem_db, tmp_path, clock,
    ) -> None:
        """Each entry buys ``position_size / entry_price`` shares."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
        _write_spy(mem_db, tmp_path / "SPY.parquet", prices)
        spec = _spec(prices)

        result = random_entry_null(
            mem_db, spec=spec, clock=clock,
            trade_count=2, avg_holding_period=timedelta(days=2), seed=42,
            position_size=10_000.0,
        )
        entry_trades = [t for t in result.trades if t.side == Side.LONG]
        for trade in entry_trades:
            # Entry trade: shares * fill_price should equal position_size.
            assert trade.shares * trade.price == pytest.approx(10_000.0, abs=1.0)
