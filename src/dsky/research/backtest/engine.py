"""Event-driven backtest engine.

The engine is a forward-stepping trade simulation, not a vectorized
event study. It iterates over a sequence of ``decision_times`` in
strict chronological order, and at each decision point it may ONLY
see data via :func:`dsky.data.bars.load_bars` with that
``decision_time`` as the gate. This is the structural enforcement of
AGENTS.md's time-discipline invariant: "the backtest may only read
points where available_time <= decision_time".

Fills
-----
When the strategy emits a target that differs from the engine's
current position, the engine submits a pending order. The order is
filled on the **next bar's open** -- specifically, at the first
decision after the order is placed where the data exposes a "new"
bar (one whose ``event_time`` is strictly greater than the bar
visible at the time the order was placed). This is the realistic
fill model: a decision made at 9:30 ET on day T sees day T-1's bar
and the next fill happens at the open of day T+1.

The engine's defensive backstop
-------------------------------
Even though ``data.bars.load_bars`` enforces the
``available_time <= decision_time`` gate, the engine scans every
returned row and raises :class:`LookAheadViolation` if a bar
violates the gate. This is belt-and-suspenders: if a future bug
let future data through the data layer, the engine's own check
would catch it and fail loud.

Output
------
The engine writes a single ``backtest.run.recorded`` event to the
log per run, with the spec, decision times, trade blotter, equity
curve, and summary stats embedded in the payload. No separate
trade-fill events; the trade blotter is the run's payload.
"""
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import polars as pl

from dsky.clock import Clock
from dsky.data.bars import load_bars
from dsky.db.events import append_event
from dsky.research.backtest.costs import CostModel

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Side(StrEnum):
    """The target position the engine tracks."""

    LONG = "long"
    FLAT = "flat"


@dataclass(frozen=True)
class Signal:
    """A target-position signal emitted by the strategy.

    The engine compares the new ``target`` to its current target
    state; if they differ, it submits a pending order. ``hold`` is
    not a target -- the engine treats any unrecognized return value
    as "no change" (see :meth:`BacktestEngine._parse_signal`).
    """

    target: Side


@dataclass(frozen=True)
class BacktestSpec:
    """A pre-registered backtest specification.

    Frozen -- the engine refuses to run if a decision time falls
    outside the spec's training window. The spec is the contract
    that makes "you may only test on this slice" enforceable.
    """

    symbol: str
    training_window: tuple[datetime, datetime]

    def contains(self, t: datetime) -> bool:
        """True iff ``t`` is within the spec's training window (inclusive)."""
        start, end = self.training_window
        return start <= t <= end


@dataclass(frozen=True)
class Fill:
    """A single trade fill at a known bar's open price."""

    side: Side
    bar_event_time: datetime
    decision_time: datetime
    price: float
    shares: float
    cost: float
    cash_after: float
    shares_after: float


@dataclass(frozen=True)
class EquityPoint:
    """A point on the equity curve, taken at a decision time."""

    decision_time: datetime
    cash: float
    position_value: float
    total_equity: float


@dataclass(frozen=True)
class BacktestResult:
    """The output of a single backtest run.

    ``trades`` is the blotter; ``equity_curve`` is one point per
    decision time; ``total_return`` is the cumulative PnL divided by
    the initial position size; ``run_event_id`` is the id of the
    ``backtest.run.recorded`` event written to the log.
    """

    spec: BacktestSpec
    trades: list[Fill]
    equity_curve: list[EquityPoint]
    total_return: float
    num_trades: int
    run_event_id: int


# A strategy is a callable that receives (state, decision_time, bars)
# and returns either a Signal or a string shorthand ("long" / "flat" /
# "hold" / anything else = no change). The state dict is owned by the
# engine and persists across decisions; the strategy can mutate it.
Strategy = Callable[[dict[str, Any], datetime, pl.DataFrame], "Signal | str"]


# Internal: an order waiting for the next bar's open to fill.
@dataclass
class _PendingOrder:
    target: Side
    pending_bar_event_time: datetime  # the bar known when the order was placed
    decision_time: datetime


# Internal: mutable per-run state. Kept off the public API; the engine
# owns it and threads it through the per-decision steps.
@dataclass
class _RunState:
    current_target: Side = Side.FLAT
    pending: _PendingOrder | None = None
    cash: float = 0.0
    shares: float = 0.0
    trades: list[Fill] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Forward-stepping trade simulation. See module docstring for the contract."""

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        clock: Clock,
        costs: CostModel | None = None,
        position_size: float = 10_000.0,
    ) -> None:
        """Store connection, clock, and per-trade cost model.

        Position size is USD notional per fill.
        """
        self._conn = conn
        self._clock = clock
        self._costs = costs if costs is not None else CostModel()
        self._position_size = position_size

    def run(
        self,
        *,
        spec: BacktestSpec,
        strategy: Strategy,
        decision_times: list[datetime],
    ) -> BacktestResult:
        """Run the backtest. See module docstring for the fill semantics.

        Parameters
        ----------
        spec:
            The pre-registered spec. The engine raises ``ValueError``
            if any decision time falls outside ``spec.training_window``.
        strategy:
            A callable ``(state, decision_time, bars) -> Signal | str``.
            ``state`` is a dict the engine owns and persists across
            decisions; the strategy can mutate it for its own bookkeeping.
        decision_times:
            Chronologically-ordered list of decision times. The engine
            iterates in this order and does not sort.

        Returns
        -------
        BacktestResult
            The trade blotter, equity curve, and a reference to the
            ``backtest.run.recorded`` event written to the log.

        """
        self._validate_decision_times(spec, decision_times)

        run_state = _RunState()
        strategy_state: dict[str, Any] = {}

        for decision_time in decision_times:
            self._step(
                decision_time=decision_time,
                spec=spec,
                strategy=strategy,
                strategy_state=strategy_state,
                run_state=run_state,
            )

        self._force_close_if_pending(
            spec=spec,
            last_decision_time=decision_times[-1] if decision_times else None,
            run_state=run_state,
        )

        total_return = self._compute_total_return(run_state)
        run_event_id = self._write_run_event(
            spec=spec,
            decision_times=decision_times,
            run_state=run_state,
            total_return=total_return,
        )

        return BacktestResult(
            spec=spec,
            trades=run_state.trades,
            equity_curve=run_state.equity_curve,
            total_return=total_return,
            num_trades=len(run_state.trades),
            run_event_id=run_event_id,
        )

    # -----------------------------------------------------------------------
    # Run-level steps
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_decision_times(
        spec: BacktestSpec, decision_times: list[datetime],
    ) -> None:
        """All decision times must be inside the spec's training window."""
        for t in decision_times:
            if not spec.contains(t):
                msg = (
                    f"decision_time {t.isoformat()} is outside the spec's "
                    f"training window [{spec.training_window[0].isoformat()}, "
                    f"{spec.training_window[1].isoformat()}]"
                )
                raise ValueError(msg)

    def _step(
        self,
        *,
        decision_time: datetime,
        spec: BacktestSpec,
        strategy: Strategy,
        strategy_state: dict[str, Any],
        run_state: _RunState,
    ) -> None:
        """One iteration: fill pending, call strategy, mark to market."""
        # The only public read path -- the data layer's gate applies here.
        bars = load_bars(self._conn, spec.symbol, decision_time)

        # Engine's defensive backstop: if the data layer's gate ever breaks,
        # this raises LookAheadViolation with a clear message.
        self._check_no_look_ahead(spec.symbol, decision_time, bars)

        current_bar = self._latest_bar(bars)

        if pending_fill := self._try_fill_pending(current_bar, run_state):
            run_state.trades.append(pending_fill)
            run_state.cash = pending_fill.cash_after
            run_state.shares = pending_fill.shares_after
            run_state.pending = None

        new_target = self._parse_signal(strategy(strategy_state, decision_time, bars))
        if (
            new_target is not None
            and new_target != run_state.current_target
            and current_bar is not None
        ):
            run_state.pending = _PendingOrder(
                target=new_target,
                pending_bar_event_time=current_bar["event_time"],
                decision_time=decision_time,
            )
            run_state.current_target = new_target

        self._mark_to_market(decision_time, current_bar, run_state)

    def _try_fill_pending(
        self, current_bar: dict[str, Any] | None, run_state: _RunState,
    ) -> Fill | None:
        """Fill a pending order at the current bar's open iff it's a new bar."""
        if run_state.pending is None or current_bar is None:
            return None
        current_event_t = current_bar["event_time"]
        if _to_utc(current_event_t) <= _to_utc(run_state.pending.pending_bar_event_time):
            return None  # not a new bar; wait
        fill_price = float(current_bar["open"])
        if run_state.pending.target == Side.LONG:
            # Entering long: spend ``position_size`` notional, receive shares.
            fill_shares = self._position_size / fill_price
            cost = self._costs.apply(fill_price, fill_shares)
            cash_after = run_state.cash - fill_price * fill_shares - cost
            shares_after = fill_shares
        else:  # Side.FLAT -- close the entire open long position
            fill_shares = run_state.shares
            cost = self._costs.apply(fill_price, fill_shares)
            cash_after = run_state.cash + fill_price * fill_shares - cost
            shares_after = 0.0
        return Fill(
            side=run_state.pending.target,
            bar_event_time=current_event_t,
            decision_time=run_state.pending.decision_time,
            price=fill_price,
            shares=fill_shares,
            cost=cost,
            cash_after=cash_after,
            shares_after=shares_after,
        )

    def _mark_to_market(
        self,
        decision_time: datetime,
        current_bar: dict[str, Any] | None,
        run_state: _RunState,
    ) -> None:
        """Append an equity point at the current decision's close."""
        if current_bar is not None:
            position_value = run_state.shares * float(current_bar["close"])
        else:
            position_value = 0.0
        run_state.equity_curve.append(EquityPoint(
            decision_time=decision_time,
            cash=run_state.cash,
            position_value=position_value,
            total_equity=run_state.cash + position_value,
        ))

    def _force_close_if_pending(
        self,
        *,
        spec: BacktestSpec,
        last_decision_time: datetime | None,
        run_state: _RunState,
    ) -> None:
        """If a position is still open at the end of the run, close it at the last bar's close.

        This produces a final realized PnL even if the strategy never
        explicitly went flat. The close uses the last decision's
        ``close`` price -- the only honest mark within the engine's
        point-in-time view.
        """
        if run_state.pending is None or last_decision_time is None:
            return
        if not run_state.equity_curve:
            return
        last_bar = self._latest_bar(load_bars(self._conn, spec.symbol, last_decision_time))
        if last_bar is None:
            return
        fill_price = float(last_bar["close"])
        cost = self._costs.apply(fill_price, run_state.shares)
        if run_state.pending.target == Side.LONG:
            # Pending buy that never filled: enter at the last bar's close.
            fill_shares = self._position_size / fill_price
            cash_after = run_state.cash - fill_price * fill_shares - cost
        else:  # Side.FLAT -- close any open long position
            fill_shares = run_state.shares
            cash_after = run_state.cash + fill_price * fill_shares - cost
        run_state.trades.append(Fill(
            side=run_state.pending.target,
            bar_event_time=last_bar["event_time"],
            decision_time=last_decision_time,
            price=fill_price,
            shares=fill_shares,
            cost=cost,
            cash_after=cash_after,
            shares_after=0.0,
        ))
        run_state.cash = cash_after
        run_state.shares = 0.0
        run_state.pending = None
        # Replace the last equity point with the realized close.
        run_state.equity_curve[-1] = EquityPoint(
            decision_time=last_decision_time,
            cash=run_state.cash,
            position_value=0.0,
            total_equity=run_state.cash,
        )

    def _compute_total_return(self, run_state: _RunState) -> float:
        """Cumulative PnL divided by the initial position size."""
        if not run_state.equity_curve:
            return 0.0
        # The initial position size is the cost basis; the "initial
        # equity" of a $10,000 trade is the cash outlay at entry,
        # which equals position_size for an all-in entry. For the
        # mark-to-market path, the engine never had a "starting cash"
        # of position_size -- the cash started at 0 and went negative
        # on entry. The realized close at the end of the run gives a
        # final cash value whose magnitude reflects the PnL.
        final_cash = run_state.equity_curve[-1].total_equity
        # Return is final equity divided by the notional deployed per fill.
        return final_cash / self._position_size

    def _write_run_event(
        self,
        *,
        spec: BacktestSpec,
        decision_times: list[datetime],
        run_state: _RunState,
        total_return: float,
    ) -> int:
        """Write the BACKTEST_RUN event to the log and return its id."""
        payload = {
            "spec_symbol": spec.symbol,
            "training_window": [
                spec.training_window[0].isoformat(),
                spec.training_window[1].isoformat(),
            ],
            "decision_times": [t.isoformat() for t in decision_times],
            "trades": [
                {
                    **asdict(t),
                    "side": t.side.value,
                    "bar_event_time": t.bar_event_time.isoformat(),
                    "decision_time": t.decision_time.isoformat(),
                }
                for t in run_state.trades
            ],
            "equity_curve": [
                {**asdict(e), "decision_time": e.decision_time.isoformat()}
                for e in run_state.equity_curve
            ],
            "total_return": total_return,
            "num_trades": len(run_state.trades),
            "costs": {
                "commission_per_share": self._costs.commission_per_share,
                "slippage_bps": self._costs.slippage_bps,
            },
            "position_size": self._position_size,
        }
        return append_event(
            conn=self._conn,
            event_type="backtest.run.recorded",
            entity_id=f"backtest:{spec.symbol}",
            actor="code:engine",
            payload=payload,
            clock=self._clock,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_no_look_ahead(
        symbol: str, decision_time: datetime, bars: pl.DataFrame,
    ) -> None:
        """Engine's defensive backstop behind the data layer's gate.

        If any row in ``bars`` has ``available_time > decision_time``,
        raise :class:`LookAheadViolation` with the offending timestamps.
        This is structural: it cannot be disabled, and it runs on every
        decision.
        """
        if bars.height == 0 or "available_time" not in bars.columns:
            return
        for t in bars["available_time"].to_list():
            t_utc = _to_utc(t)
            if t_utc > _to_utc(decision_time):
                raise LookAheadViolation(
                    decision_time=decision_time,
                    available_time=t_utc,
                    symbol=symbol,
                )

    @staticmethod
    def _latest_bar(bars: pl.DataFrame) -> dict[str, Any] | None:
        """Return the bar with the latest ``event_time``, or None if empty."""
        if bars.height == 0:
            return None
        return bars.sort("event_time").row(-1, named=True)

    @staticmethod
    def _parse_signal(signal: "Signal | str | None") -> Side | None:
        """Interpret the strategy's return value as a target Side.

        - ``Signal(target=...)`` -> that target.
        - ``"long"`` / ``"LONG"`` -> ``Side.LONG``.
        - ``"flat"`` / ``"FLAT"`` -> ``Side.FLAT``.
        - ``"hold"`` / ``""`` / ``None`` / anything else -> ``None``
          (engine treats as "no change", keeps its current target).
        """
        if signal is None:
            return None
        if isinstance(signal, Signal):
            return signal.target
        if isinstance(signal, str):
            s = signal.strip().upper()
            if s == "LONG":
                return Side.LONG
            if s == "FLAT":
                return Side.FLAT
        return None


# ---------------------------------------------------------------------------
# LookAheadViolation -- also imported by tests/test_lookahead_invariant.py
# ---------------------------------------------------------------------------

class LookAheadViolation(Exception):  # noqa: N818 -- named per the test spec
    """Raised when the engine observes a bar with available_time > decision_time.

    The backtest is point-in-time: at every decision, the engine has
    access to data with ``available_time <= decision_time``. If a bar
    whose ``available_time`` is strictly greater than the
    ``decision_time`` is observed, the engine's time discipline has
    been violated. The data layer's gate is the primary defence; this
    exception is the engine's defensive backstop.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_utc(t: datetime) -> datetime:
    """Normalize a datetime to UTC for tz-aware comparison."""
    if t.tzinfo is None:
        return t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestSpec",
    "EquityPoint",
    "Fill",
    "LookAheadViolation",
    "Side",
    "Signal",
    "Strategy",
]
