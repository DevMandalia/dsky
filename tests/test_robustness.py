"""Tests for the robustness suite (subsample, outlier, regime, nulls)."""
import random
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dsky.clock import FrozenClock
from dsky.data.bars import write_bars
from dsky.db.engine import open_db
from dsky.db.projections import rebuild_projections
from dsky.research.backtest.engine import (
    BacktestEngine,
    BacktestSpec,
    Fill,
    Side,
)
from dsky.research.robustness import run_robustness_suite
from dsky.research.robustness.nulls import analyze_nulls
from dsky.research.robustness.outlier_drop import analyze_outlier_drop
from dsky.research.robustness.subsample import analyze_subsample
from dsky.research.spec import (
    ComputableRule,
    HypothesisSpec,
    SuccessCriteria,
    TimeWindow,
)

_T0 = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
_HOLDOUT_START = datetime(2026, 2, 1, 0, 0, 0, tzinfo=UTC)
_TRADE_COUNT = 6
_AVG_HOLD = timedelta(days=3)


@pytest.fixture
def mem_db() -> Iterator[sqlite3.Connection]:
    conn = open_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(at=_T0)


def _hypothesis_spec() -> HypothesisSpec:
    return HypothesisSpec(
        asset="SPY",
        signal_definition=ComputableRule(
            kind="momentum",
            parameters=(("lookback_days", 20),),
        ),
        entry_rule=ComputableRule(kind="threshold_cross", parameters=()),
        exit_rule=ComputableRule(kind="holding_period", parameters=(("max_days", 5),)),
        prediction="momentum predicts positive holdout return",
        success_criteria=SuccessCriteria(
            metric="total_return", threshold=0.02, comparator="gt",
        ),
        train_window=TimeWindow(start="2020-01-01", end="2023-12-31"),
        holdout_window=TimeWindow(start="2026-02-01", end="2026-02-28"),
        required_null_models=("random_entry_null",),
    )


def _bars_df(prices: list[float], start: datetime = _HOLDOUT_START) -> pl.DataFrame:
    return pl.DataFrame({
        "event_time": [start + timedelta(days=i) for i in range(len(prices))],
        "open": list(prices),
        "high": list(prices),
        "low": list(prices),
        "close": list(prices),
        "volume": [1_000_000] * len(prices),
    })


def _write_bars(
    conn: sqlite3.Connection,
    path: Path,
    prices: list[float],
    *,
    start: datetime = _HOLDOUT_START,
) -> None:
    bars = _bars_df(prices, start=start)
    write_bars(
        conn=conn,
        symbol="SPY",
        bars=bars,
        parquet_path=path,
        vendor="synthetic",
        date_start=start.date().isoformat(),
        date_end=(start + timedelta(days=len(prices) - 1)).date().isoformat(),
        clock=FrozenClock(at=_T0),
    )
    rebuild_projections(conn)


def _holdout_decision_times(n: int) -> list[datetime]:
    out: list[datetime] = []
    cursor = _HOLDOUT_START
    while len(out) < n:
        dt = datetime(
            cursor.year, cursor.month, cursor.day, 14, 30, 0, tzinfo=UTC,
        )
        if dt.weekday() < 5:
            out.append(dt)
        cursor += timedelta(days=1)
    return out


def _run_holdout_backtest(
    conn: sqlite3.Connection,
    clock: FrozenClock,
    prices: list[float],
    path: Path,
    strategy: Any,
) -> Any:
    _write_bars(conn, path, prices)
    dts = _holdout_decision_times(min(len(prices) - 1, 25))
    spec = BacktestSpec(symbol="SPY", training_window=(dts[0], dts[-1]))
    return BacktestEngine(conn=conn, clock=clock, position_size=10_000.0).run(
        spec=spec, strategy=strategy, decision_times=dts,
    )


def _trend_strategy(state: dict[str, Any], _dt: datetime, _bars: pl.DataFrame) -> str:
    """Single hold across the holdout uptrend."""
    state["n"] = state.get("n", 0) + 1
    if state["n"] == 1:
        return "long"
    if state["n"] == 20:
        return "flat"
    return "hold"


def _noise_strategy(state: dict[str, Any], _dt: datetime, _bars: pl.DataFrame) -> str:
    """Alternate in/out on a random-walk series -- no real edge."""
    state["n"] = state.get("n", 0) + 1
    if state["n"] % 7 == 1:
        return "long"
    if state["n"] % 7 == 4:
        return "flat"
    return "hold"


def _round_trip_fill(
    entry_t: datetime,
    exit_t: datetime,
    entry_price: float,
    exit_price: float,
) -> list[Fill]:
    shares = 100.0
    return [
        Fill(
            side=Side.LONG, bar_event_time=entry_t, decision_time=entry_t,
            price=entry_price, shares=shares, cost=0.0,
            cash_after=-entry_price * shares, shares_after=shares,
        ),
        Fill(
            side=Side.FLAT, bar_event_time=exit_t, decision_time=exit_t,
            price=exit_price, shares=shares, cost=0.0,
            cash_after=exit_price * shares, shares_after=0.0,
        ),
    ]


def _craft_outlier_trades() -> list[Fill]:
    """Six small wins + one dominant outlier round trip."""
    base = datetime(2026, 2, 3, 14, 30, 0, tzinfo=UTC)
    trades: list[Fill] = []
    for i in range(6):
        et = base + timedelta(days=i * 2)
        xt = et + timedelta(days=1)
        trades.extend(_round_trip_fill(et, xt, 100.0, 100.02))
    trades.extend(_round_trip_fill(
        base + timedelta(days=20), base + timedelta(days=21), 100.0, 109.0,
    ))
    return trades


class TestHoldoutDiscipline:
    def test_holdout_window_only_read_in_nulls_module(self) -> None:
        """The holdout slice is consumed exactly once, in nulls.py."""
        root = Path(__file__).resolve().parents[1] / "src/dsky/research/robustness"
        for path in root.glob("*.py"):
            if path.name in {"nulls.py", "_math.py"}:
                continue
            text = path.read_text()
            # Attribute access only -- docstring mentions are fine.
            assert "hypothesis_spec.holdout_window" not in text, (
                f"{path.name} must not read holdout_window"
            )


class TestPlantedEdgeSurvives:
    def test_planted_uptrend_passes_robustness_suite(
        self, mem_db, tmp_path, clock,
    ) -> None:
        # Steady holdout drift: ~0.5% per day over 28 bars.
        prices = [100.0 + 0.5 * i for i in range(28)]
        bt = _run_holdout_backtest(
            mem_db, clock, prices, tmp_path / "SPY.parquet", _trend_strategy,
        )
        assert bt.total_return > 0.03

        result = run_robustness_suite(
            mem_db,
            entity_id="idea-edge",
            hypothesis_spec=_hypothesis_spec(),
            clock=clock,
            strategy_metric=bt.total_return,
            trades=bt.trades,
            bars=_bars_df(prices),
            position_size=10_000.0,
            trade_count=_TRADE_COUNT,
            avg_holding_period=_AVG_HOLD,
            n_null_seeds=30,
        )

        assert not result.subsample.unstable
        assert not result.outlier_drop.edge_lost
        assert result.nulls.significant
        assert result.nulls.p_value <= 0.05
        assert result.passed is True

        row = mem_db.execute(
            "SELECT event_type FROM events WHERE id = ?",
            (result.run_event_id,),
        ).fetchone()
        assert row["event_type"] == "robustness.run.recorded"


class TestRandomWalkRejected:
    def test_pure_noise_fails_null_comparison(
        self, mem_db, tmp_path, clock,
    ) -> None:
        rng = random.Random(0)  # noqa: S311 -- deterministic test noise
        prices = [100.0]
        for _ in range(27):
            prices.append(prices[-1] + rng.choice([-0.2, 0.2]))

        bt = _run_holdout_backtest(
            mem_db, clock, prices, tmp_path / "SPY.parquet", _noise_strategy,
        )
        nulls = analyze_nulls(
            mem_db,
            _hypothesis_spec(),
            clock,
            strategy_metric=bt.total_return,
            trade_count=_TRADE_COUNT,
            avg_holding_period=_AVG_HOLD,
            n_seeds=40,
        )
        assert not nulls.significant
        assert nulls.p_value > 0.05

        suite = run_robustness_suite(
            mem_db,
            entity_id="idea-noise",
            hypothesis_spec=_hypothesis_spec(),
            clock=clock,
            strategy_metric=bt.total_return,
            trades=bt.trades,
            bars=_bars_df(prices),
            position_size=10_000.0,
            trade_count=_TRADE_COUNT,
            avg_holding_period=_AVG_HOLD,
            n_null_seeds=40,
        )
        assert suite.passed is False


class TestOutlierDrop:
    def test_outlier_dependent_edge_is_flagged(self) -> None:
        trades = _craft_outlier_trades()
        # Headline ~9.1%; the 100->109 round trip dominates.
        key_metric = 0.0912
        result = analyze_outlier_drop(
            key_metric=key_metric,
            trades=trades,
            position_size=10_000.0,
        )
        assert result.edge_lost is True
        assert result.metric_after_drop < key_metric * 0.25


class TestSubsampleStability:
    def test_evenly_spread_edge_is_stable(self) -> None:
        """Three equal positive thirds should not flag instability."""
        base = datetime(2026, 2, 1, 14, 30, 0, tzinfo=UTC)
        trades: list[Fill] = []
        for i in range(9):
            et = base + timedelta(days=i * 3)
            xt = et + timedelta(days=1)
            trades.extend(_round_trip_fill(et, xt, 100.0, 101.0))
        result = analyze_subsample(
            key_metric=0.009,
            trades=trades,
            position_size=10_000.0,
        )
        assert result.first_third_metric > 0
        assert result.middle_third_metric > 0
        assert result.last_third_metric > 0
        assert not result.unstable


class TestPHackingSmoke:
    """Feed pure-noise series; essentially all fail the null gate."""

    def test_noise_series_rejected_at_least_eighty_percent(
        self, mem_db, tmp_path, clock,
    ) -> None:
        rejected = 0
        n_series = 20
        for series_seed in range(n_series):
            rng = random.Random(series_seed + 100)  # noqa: S311 -- test noise
            prices = [100.0]
            for _ in range(27):
                prices.append(prices[-1] + rng.uniform(-0.25, 0.25))

            path = tmp_path / f"SPY_{series_seed}.parquet"
            bt = _run_holdout_backtest(mem_db, clock, prices, path, _noise_strategy)
            nulls = analyze_nulls(
                mem_db,
                _hypothesis_spec(),
                clock,
                strategy_metric=bt.total_return,
                trade_count=_TRADE_COUNT,
                avg_holding_period=_AVG_HOLD,
                n_seeds=80,
            )
            if not nulls.significant:
                rejected += 1
        need = int(n_series * 0.8)
        assert rejected >= need, (
            f"only {rejected}/{n_series} noise series rejected by null gate "
            f"(need >={need})"
        )
