"""Deterministic operations invoked by the CLI (no Typer here)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003 — used at runtime for Parquet paths
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import sqlite3

    from dsky.clock import Clock
from dsky.config import data_dir
from dsky.data.bars import next_nyse_open, write_bars
from dsky.db.events import append_event, verify_chain
from dsky.db.projections import rebuild_projections
from dsky.lifecycle.machine import transition
from dsky.lifecycle.registry import load_pre_registered_spec, pre_register
from dsky.lifecycle.states import ApprovalType, State
from dsky.research.backtest.engine import BacktestEngine, BacktestSpec
from dsky.research.ledger import check_duplicate, fingerprint, record_rejection
from dsky.research.robustness import run_robustness_suite
from dsky.research.spec import (
    ComputableRule,
    HypothesisSpec,
    SuccessCriteria,
    TimeWindow,
)

# Canonical running-example hypothesis (Step 11 smoke).
BOUNCE_HYPOTHESIS = (
    "When SPY closes down more than 2% on a day, the next day tends to bounce."
)

_TRAIN_WINDOW = TimeWindow(start="2026-01-01", end="2026-01-31")
_HOLDOUT_WINDOW = TimeWindow(start="2026-02-01", end="2026-02-28")
_TRAIN_BAR_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_HOLDOUT_BAR_START = datetime(2026, 2, 1, 0, 0, 0, tzinfo=UTC)
_POSITION_SIZE = 10_000.0
_NULL_TRADE_COUNT = 6
_NULL_AVG_HOLD = timedelta(days=3)
_DOWN_DAY_THRESHOLD_PCT = -2.0
_MIN_BARS_FOR_RETURN = 2
_WEEKDAY_FRIDAY = 4
_SUMMARY_SNIPPET_LEN = 72


@dataclass(frozen=True)
class OpResult:
    """Human-facing outcome of a CLI operation."""

    summary: str
    event_id: int
    details: dict[str, Any]


def bounce_strategy(
    state: dict[str, Any], _decision_time: datetime, bars: pl.DataFrame,
) -> str:
    """Enter long after a >2% down day; exit on the following decision."""
    if state.get("pending_exit"):
        state["pending_exit"] = False
        state["in_position"] = False
        return "flat"
    if bars.height < _MIN_BARS_FOR_RETURN:
        return "hold"
    sorted_bars = bars.sort("event_time")
    last = sorted_bars.row(-1, named=True)
    prev = sorted_bars.row(-2, named=True)
    ret_pct = (float(last["close"]) - float(prev["close"])) / float(prev["close"]) * 100.0
    if ret_pct <= _DOWN_DAY_THRESHOLD_PCT and not state.get("in_position"):
        state["in_position"] = True
        state["pending_exit"] = True
        return "long"
    return "hold"


def build_draft_spec(*, asset: str, hypothesis: str) -> dict[str, Any]:
    """Deterministic structured spec for the bounce hypothesis."""
    return {
        "asset": asset,
        "hypothesis": hypothesis,
        "signal_definition": {
            "kind": "down_day_reversal",
            "parameters": [["threshold_pct", -2.0]],
        },
        "entry_rule": {"kind": "next_open_long", "parameters": []},
        "exit_rule": {"kind": "next_close_flat", "parameters": []},
        "prediction": hypothesis,
        "success_criteria": {
            "metric": "total_return",
            "threshold": 0.01,
            "comparator": "gt",
        },
        "train_window": {"start": _TRAIN_WINDOW.start, "end": _TRAIN_WINDOW.end},
        "holdout_window": {"start": _HOLDOUT_WINDOW.start, "end": _HOLDOUT_WINDOW.end},
        "required_null_models": ["random_entry_null"],
    }


def draft_to_hypothesis_spec(draft: dict[str, Any]) -> HypothesisSpec:
    """Convert an ``idea.specified`` draft dict to a :class:`HypothesisSpec`."""
    sig = draft["signal_definition"]
    entry = draft["entry_rule"]
    exit_r = draft["exit_rule"]
    sc = draft["success_criteria"]
    tw = draft["train_window"]
    hw = draft["holdout_window"]
    return HypothesisSpec(
        asset=str(draft["asset"]),
        signal_definition=ComputableRule(
            kind=str(sig["kind"]),
            parameters=tuple((str(p[0]), p[1]) for p in sig["parameters"]),
        ),
        entry_rule=ComputableRule(
            kind=str(entry["kind"]),
            parameters=tuple((str(p[0]), p[1]) for p in entry.get("parameters", [])),
        ),
        exit_rule=ComputableRule(
            kind=str(exit_r["kind"]),
            parameters=tuple((str(p[0]), p[1]) for p in exit_r.get("parameters", [])),
        ),
        prediction=str(draft["prediction"]),
        success_criteria=SuccessCriteria(
            metric=str(sc["metric"]),
            threshold=float(sc["threshold"]),
            comparator=str(sc["comparator"]),
        ),
        train_window=TimeWindow(start=str(tw["start"]), end=str(tw["end"])),
        holdout_window=TimeWindow(start=str(hw["start"]), end=str(hw["end"])),
        required_null_models=tuple(draft["required_null_models"]),
    )


def _load_specified_draft(conn: sqlite3.Connection, entity_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT payload FROM events
        WHERE entity_id = ? AND event_type = 'idea.specified'
        ORDER BY id DESC LIMIT 1
        """,
        (entity_id,),
    ).fetchone()
    if row is None:
        msg = f"no idea.specified event for {entity_id!r}; run specify first"
        raise LookupError(msg)
    payload: dict[str, Any] = json.loads(row["payload"])
    draft: dict[str, Any] = payload["draft"]
    return draft


def _decision_times_in_range(start: datetime, end: datetime, n: int) -> list[datetime]:
    out: list[datetime] = []
    cursor = start
    while len(out) < n and cursor <= end + timedelta(days=60):
        dt = datetime(cursor.year, cursor.month, cursor.day, 14, 30, 0, tzinfo=UTC)
        if start <= dt <= end + timedelta(days=1) and dt.weekday() <= _WEEKDAY_FRIDAY:
            out.append(dt)
        cursor += timedelta(days=1)
    return out[:n]


def _backtest_spec_for_window(asset: str, window: TimeWindow) -> BacktestSpec:
    start_day = datetime.fromisoformat(window.start).replace(tzinfo=UTC)
    end_day = datetime.fromisoformat(window.end).replace(tzinfo=UTC)
    return BacktestSpec(
        symbol=asset,
        training_window=(next_nyse_open(start_day), next_nyse_open(end_day)),
    )


def seed_synthetic_bars(  # noqa: PLR0913
    conn: sqlite3.Connection,
    *,
    asset: str,
    clock: Clock,
    parquet_path: Path,
    train_prices: list[float] | None = None,
    holdout_prices: list[float] | None = None,
) -> None:
    """Write deterministic synthetic train+holdout bars to Parquet."""
    if train_prices is None:
        train_prices = [100.0 + 0.1 * i for i in range(22)]
    if holdout_prices is None:
        rng = random.Random(11)  # noqa: S311 -- deterministic demo noise
        holdout_prices = [100.0]
        for _ in range(21):
            holdout_prices.append(holdout_prices[-1] + rng.uniform(-0.4, 0.4))

    train_times = [_TRAIN_BAR_START + timedelta(days=i) for i in range(len(train_prices))]
    holdout_times = [
        _HOLDOUT_BAR_START + timedelta(days=i) for i in range(len(holdout_prices))
    ]
    bars = pl.DataFrame({
        "event_time": train_times + holdout_times,
        "open": train_prices + holdout_prices,
        "high": train_prices + holdout_prices,
        "low": train_prices + holdout_prices,
        "close": train_prices + holdout_prices,
        "volume": [1_000_000] * (len(train_prices) + len(holdout_prices)),
    })
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    write_bars(
        conn=conn,
        symbol=asset,
        bars=bars,
        parquet_path=parquet_path,
        vendor="synthetic",
        date_start=_TRAIN_WINDOW.start,
        date_end=_HOLDOUT_WINDOW.end,
        clock=clock,
    )
    rebuild_projections(conn)


def op_capture(  # noqa: PLR0913
    conn: sqlite3.Connection,
    *,
    entity_id: str,
    hypothesis: str,
    asset: str,
    actor: str,
    clock: Clock,
) -> OpResult:
    """``idea.registered`` — entity enters CAPTURED."""
    event_id = append_event(
        conn=conn,
        event_type="idea.registered",
        entity_id=entity_id,
        actor=actor,
        payload={"name": entity_id, "asset": asset, "hypothesis": hypothesis},
        clock=clock,
    )
    snippet = (
        hypothesis
        if len(hypothesis) <= _SUMMARY_SNIPPET_LEN
        else f"{hypothesis[:_SUMMARY_SNIPPET_LEN]}..."
    )
    return OpResult(
        summary=f"Captured idea {entity_id!r} for {asset}: {snippet}",
        event_id=event_id,
        details={"asset": asset},
    )


def op_specify(  # noqa: PLR0913
    conn: sqlite3.Connection,
    *,
    entity_id: str,
    hypothesis: str,
    asset: str,
    actor: str,
    clock: Clock,
) -> OpResult:
    """``idea.specified`` + transition to SPECIFIED."""
    draft = build_draft_spec(asset=asset, hypothesis=hypothesis)
    spec_event_id = append_event(
        conn=conn,
        event_type="idea.specified",
        entity_id=entity_id,
        actor=actor,
        payload={"draft": draft},
        clock=clock,
    )
    state_event_id = transition(
        conn, entity_id, State.SPECIFIED, actor, clock,
    )
    return OpResult(
        summary=(
            f"Specified {entity_id!r}: signal={draft['signal_definition']['kind']}, "
            f"train={draft['train_window']['start']}..{draft['train_window']['end']}"
        ),
        event_id=state_event_id,
        details={"spec_event_id": spec_event_id},
    )


def op_preregister(
    conn: sqlite3.Connection,
    *,
    entity_id: str,
    actor: str,
    clock: Clock,
) -> OpResult:
    """``spec.pre_registered`` + transition to PRE_REGISTERED."""
    draft = _load_specified_draft(conn, entity_id)
    spec = draft_to_hypothesis_spec(draft)
    spec_event_id = pre_register(conn, entity_id, spec, actor, clock)
    state_event_id = transition(
        conn, entity_id, State.PRE_REGISTERED, actor, clock,
    )
    return OpResult(
        summary=(
            f"Pre-registered {entity_id!r}; spec_hash={spec.content_hash()[:12]}..."
        ),
        event_id=state_event_id,
        details={"spec_event_id": spec_event_id, "spec_hash": spec.content_hash()},
    )


def op_backtest(
    conn: sqlite3.Connection,
    *,
    entity_id: str,
    actor: str,
    clock: Clock,
    parquet_path: Path,
) -> OpResult:
    """Train-window backtest + ``backtest.run.recorded`` + BACKTESTED."""
    spec = load_pre_registered_spec(conn, entity_id)
    seed_synthetic_bars(conn, asset=spec.asset, clock=clock, parquet_path=parquet_path)
    bt_spec = _backtest_spec_for_window(spec.asset, spec.train_window)
    start = bt_spec.training_window[0]
    end = bt_spec.training_window[1]
    decision_times = _decision_times_in_range(start, end, 18)
    result = BacktestEngine(
        conn=conn, clock=clock, position_size=_POSITION_SIZE,
    ).run(
        spec=bt_spec,
        strategy=bounce_strategy,
        decision_times=decision_times,
    )
    state_event_id = transition(conn, entity_id, State.BACKTESTED, actor, clock)
    return OpResult(
        summary=(
            f"Backtest (train) total_return={result.total_return:.4f}, "
            f"trades={result.num_trades}"
        ),
        event_id=state_event_id,
        details={"backtest_event_id": result.run_event_id, "total_return": result.total_return},
    )


def op_robustness(
    conn: sqlite3.Connection,
    *,
    entity_id: str,
    actor: str,
    clock: Clock,
    parquet_path: Path,
) -> OpResult:
    """Holdout robustness battery + ROBUSTNESS_CHECKED."""
    hypothesis = load_pre_registered_spec(conn, entity_id)
    if not parquet_path.exists():
        seed_synthetic_bars(
            conn, asset=hypothesis.asset, clock=clock, parquet_path=parquet_path,
        )
    holdout_spec = _backtest_spec_for_window(hypothesis.asset, hypothesis.holdout_window)
    start, end = holdout_spec.training_window
    decision_times = _decision_times_in_range(start, end, 18)
    bt = BacktestEngine(
        conn=conn, clock=clock, position_size=_POSITION_SIZE,
    ).run(
        spec=holdout_spec,
        strategy=bounce_strategy,
        decision_times=decision_times,
    )
    holdout_bars = pl.read_parquet(parquet_path).filter(
        pl.col("event_time").dt.date() >= datetime.fromisoformat(
            hypothesis.holdout_window.start,
        ).date(),
    )
    suite = run_robustness_suite(
        conn,
        entity_id,
        hypothesis,
        clock,
        strategy_metric=bt.total_return,
        trades=bt.trades,
        bars=holdout_bars,
        position_size=_POSITION_SIZE,
        trade_count=_NULL_TRADE_COUNT,
        avg_holding_period=_NULL_AVG_HOLD,
        n_null_seeds=40,
        actor=actor,
    )
    state_event_id = transition(
        conn, entity_id, State.ROBUSTNESS_CHECKED, actor, clock,
    )
    verdict = "PASSED" if suite.passed else "FAILED"
    return OpResult(
        summary=(
            f"Robustness {verdict}: p={suite.nulls.p_value:.3f}, "
            f"unstable={suite.subsample.unstable}, "
            f"outlier_lost={suite.outlier_drop.edge_lost}"
        ),
        event_id=state_event_id,
        details={
            "robustness_event_id": suite.run_event_id,
            "passed": suite.passed,
            "p_value": suite.nulls.p_value,
        },
    )


def op_review(  # noqa: PLR0913
    conn: sqlite3.Connection,
    *,
    entity_id: str,
    verdict: str,
    rationale: str,
    actor: str,
    clock: Clock,
) -> OpResult:
    """``review.decided``; reject path uses machine + ledger."""
    spec = load_pre_registered_spec(conn, entity_id)
    review_id = append_event(
        conn=conn,
        event_type="review.decided",
        entity_id=entity_id,
        actor=actor,
        payload={"verdict": verdict, "rationale": rationale},
        clock=clock,
    )
    v = verdict.strip().lower()
    if v == "reject":
        reject_id = transition(conn, entity_id, State.REJECTED, actor, clock)
        ledger_id = record_rejection(
            conn, entity_id, spec, rationale, actor, clock,
        )
        return OpResult(
            summary=f"Review REJECT: {rationale}",
            event_id=reject_id,
            details={"review_event_id": review_id, "ledger_event_id": ledger_id},
        )
    if v == "approve":
        state_id = transition(
            conn, entity_id, State.ADVERSARIAL_REVIEW, actor, clock,
        )
        return OpResult(
            summary=f"Review APPROVE: {rationale}",
            event_id=state_id,
            details={"review_event_id": review_id},
        )
    msg = f"verdict must be 'approve' or 'reject', got {verdict!r}"
    raise ValueError(msg)


def op_promote(
    conn: sqlite3.Connection,
    *,
    entity_id: str,
    approval_type: str,
    actor: str,
    clock: Clock,
) -> OpResult:
    """``idea.promoted`` + transition to APPROVED."""
    at = ApprovalType(approval_type.strip().lower())
    promo_id = append_event(
        conn=conn,
        event_type="idea.promoted",
        entity_id=entity_id,
        actor=actor,
        payload={"approval_type": at.value},
        clock=clock,
    )
    state_id = transition(
        conn, entity_id, State.APPROVED, actor, clock, approval_type=at,
    )
    return OpResult(
        summary=f"Promoted {entity_id!r} with approval_type={at.value}",
        event_id=state_id,
        details={"promoted_event_id": promo_id},
    )


def op_status(conn: sqlite3.Connection, entity_id: str) -> str:
    """Ordered event history for ``entity_id``."""
    rows = conn.execute(
        """
        SELECT id, ts, event_type, actor, payload
        FROM events WHERE entity_id = ?
        ORDER BY id
        """,
        (entity_id,),
    ).fetchall()
    if not rows:
        return f"No events for entity {entity_id!r}"
    lines = [f"History for {entity_id!r} ({len(rows)} events):", ""]
    lines.extend(
        f"  [{row['id']}] {row['ts']}  {row['event_type']}  actor={row['actor']}"
        for row in rows
    )
    return "\n".join(lines)


def op_verify_chain(conn: sqlite3.Connection) -> OpResult:
    """Verify the append-only hash chain."""
    broken = verify_chain(conn)
    if broken is None:
        return OpResult(summary="Hash chain intact", event_id=0, details={})
    return OpResult(
        summary=f"Hash chain BROKEN at event id {broken}",
        event_id=broken,
        details={"broken_id": broken},
    )


def op_ledger_check(conn: sqlite3.Connection, entity_id: str) -> OpResult:
    """Check rejection ledger for a duplicate fingerprint."""
    spec = load_pre_registered_spec(conn, entity_id)
    fp = fingerprint(spec)
    warning = check_duplicate(conn, spec)
    if warning is None:
        return OpResult(
            summary=f"No duplicate in ledger for fingerprint {fp[:12]}...",
            event_id=0,
            details={"fingerprint": fp},
        )
    return OpResult(
        summary=(
            f"DUPLICATE: prior rejection at {warning.rejected_ts} "
            f"— {warning.reason}"
        ),
        event_id=0,
        details={
            "fingerprint": fp,
            "prior_reason": warning.reason,
            "prior_asset": warning.asset,
        },
    )


def default_parquet_path(asset: str) -> Path:
    """Parquet path for ``asset`` under :func:`dsky.config.data_dir`."""
    return data_dir() / f"{asset}.parquet"
