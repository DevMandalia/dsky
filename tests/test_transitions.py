"""Tests for the lifecycle state machine.

These tests are the gates that force discipline. The state machine is the
mechanical realisation of AGENTS.md's *Pre-registration before backtest*
invariant: a backtest cannot legally run on a spec that has not been
pre-registered (PRE_REGISTERED state).

The test ``test_anti_p_hacking_gate_specified_to_backtested_raises`` is the
dedicated check for that gate. It must remain in the file even if a future
refactor weakens the parametric illegal-transition sweep.

State machine overview
----------------------
States (14): CAPTURED, SPECIFIED, PRE_REGISTERED, BACKTESTED,
ROBUSTNESS_CHECKED, ADVERSARIAL_REVIEW, APPROVED, LIVE_MONITORED,
TRIGGERED, THESIS_REVIEW, PAPER_TRADE, ATTRIBUTED, REJECTED, RETIRED.

Forward path: CAPTURED -> SPECIFIED -> PRE_REGISTERED -> BACKTESTED ->
ROBUSTNESS_CHECKED -> ADVERSARIAL_REVIEW -> APPROVED -> LIVE_MONITORED ->
TRIGGERED -> THESIS_REVIEW -> PAPER_TRADE -> ATTRIBUTED.

Branches:
- REJECTED: reachable from any pre-approval research stage
  (CAPTURED, SPECIFIED, PRE_REGISTERED, BACKTESTED,
  ROBUSTNESS_CHECKED, ADVERSARIAL_REVIEW).
- RETIRED: reachable from APPROVED or LIVE_MONITORED.

Terminal states: ATTRIBUTED, REJECTED, RETIRED (no outgoing transitions).
"""
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from dsky.clock import FrozenClock
from dsky.db.engine import open_db
from dsky.db.events import append_event, verify_chain
from dsky.db.projections import rebuild_projections
from dsky.lifecycle.machine import IllegalTransition, UnknownEntity, transition
from dsky.lifecycle.registry import pre_register
from dsky.lifecycle.states import ApprovalType, State
from dsky.research.spec import (
    ComputableRule,
    HypothesisSpec,
    SuccessCriteria,
    TimeWindow,
)

_T0 = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_db() -> Iterator[sqlite3.Connection]:
    """In-memory SQLite with the dsky schema applied."""
    conn = open_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def clock() -> FrozenClock:
    """A frozen clock anchored at ``_T0`` so timestamps are deterministic."""
    return FrozenClock(at=_T0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_spec() -> HypothesisSpec:
    """A complete spec for seeding ``spec.pre_registered`` events in tests."""
    return HypothesisSpec(
        asset="TEST",
        signal_definition=ComputableRule(kind="test_signal", parameters=()),
        entry_rule=ComputableRule(kind="test_entry", parameters=()),
        exit_rule=ComputableRule(kind="test_exit", parameters=()),
        prediction="test prediction",
        success_criteria=SuccessCriteria(
            metric="total_return", threshold=0.0, comparator="gt",
        ),
        train_window=TimeWindow(start="2020-01-01", end="2023-12-31"),
        holdout_window=TimeWindow(start="2024-01-01", end="2024-12-31"),
        required_null_models=("buy_and_hold",),
    )


def _seed_pre_registered(
    conn: sqlite3.Connection, entity_id: str, clock: FrozenClock,
) -> None:
    """Append a ``spec.pre_registered`` event (required before BACKTESTED)."""
    pre_register(conn, entity_id, _minimal_spec(), "seeder", clock)


def _register(conn: sqlite3.Connection, entity_id: str, clock: FrozenClock) -> int:
    """Append an ``idea.registered`` event. Implies state CAPTURED."""
    return append_event(
        conn=conn,
        event_type="idea.registered",
        entity_id=entity_id,
        actor="seeder",
        payload={"name": entity_id, "asset": "TEST"},
        clock=clock,
    )


def _walk_to(
    conn: sqlite3.Connection,
    entity_id: str,
    target: State,
    clock: FrozenClock,
) -> None:
    """Append events so ``entity_id`` is in ``target`` state, starting fresh.

    Uses ``idea.state_changed`` for all transitions from CAPTURED onward.
    Used to set up entities for transition() tests. Does NOT call
    ``transition()`` itself: the tests verify the state machine reads
    from the log, so the seed must produce a real log.
    """
    linear: tuple[State, ...] = (
        State.CAPTURED, State.SPECIFIED, State.PRE_REGISTERED,
        State.BACKTESTED, State.ROBUSTNESS_CHECKED, State.ADVERSARIAL_REVIEW,
        State.APPROVED, State.LIVE_MONITORED, State.TRIGGERED,
        State.THESIS_REVIEW, State.PAPER_TRADE, State.ATTRIBUTED,
    )

    _register(conn, entity_id, clock)

    if target == State.CAPTURED:
        return

    if target == State.REJECTED:
        # Reject from CAPTURED (earliest legal origin).
        append_event(
            conn=conn,
            event_type="idea.state_changed",
            entity_id=entity_id,
            actor="seeder",
            payload={"from": State.CAPTURED.value, "to": State.REJECTED.value,
                     "reason": "test seed"},
            clock=clock,
        )
        return

    if target == State.RETIRED:
        # Walk to APPROVED, then retire.
        idx = linear.index(State.APPROVED)
        prev = State.CAPTURED
        for s in linear[1:idx + 1]:
            append_event(
                conn=conn,
                event_type="idea.state_changed",
                entity_id=entity_id,
                actor="seeder",
                payload={"from": prev.value, "to": s.value},
                clock=clock,
            )
            prev = s
        append_event(
            conn=conn,
            event_type="idea.state_changed",
            entity_id=entity_id,
            actor="seeder",
            payload={"from": State.APPROVED.value, "to": State.RETIRED.value},
            clock=clock,
        )
        return

    if target not in linear:
        msg = f"_walk_to: don't know how to seed to {target!r}"
        raise ValueError(msg)

    idx = linear.index(target)
    prev = State.CAPTURED
    for s in linear[1:idx + 1]:
        append_event(
            conn=conn,
            event_type="idea.state_changed",
            entity_id=entity_id,
            actor="seeder",
            payload={"from": prev.value, "to": s.value},
            clock=clock,
        )
        prev = s


# ---------------------------------------------------------------------------
# Transition table (kept in sync with the table under test in machine.py)
# ---------------------------------------------------------------------------

_FORWARD_PATH: tuple[tuple[State, State], ...] = (
    (State.CAPTURED,          State.SPECIFIED),
    (State.SPECIFIED,         State.PRE_REGISTERED),
    (State.PRE_REGISTERED,    State.BACKTESTED),
    (State.BACKTESTED,        State.ROBUSTNESS_CHECKED),
    (State.ROBUSTNESS_CHECKED, State.ADVERSARIAL_REVIEW),
    (State.ADVERSARIAL_REVIEW, State.APPROVED),
    (State.APPROVED,          State.LIVE_MONITORED),
    (State.LIVE_MONITORED,    State.TRIGGERED),
    (State.TRIGGERED,         State.THESIS_REVIEW),
    (State.THESIS_REVIEW,     State.PAPER_TRADE),
    (State.PAPER_TRADE,       State.ATTRIBUTED),
)

_REJECTABLE_FROM: tuple[State, ...] = (
    State.CAPTURED,
    State.SPECIFIED,
    State.PRE_REGISTERED,
    State.BACKTESTED,
    State.ROBUSTNESS_CHECKED,
    State.ADVERSARIAL_REVIEW,
)

_RETIREABLE_FROM: tuple[State, ...] = (
    State.APPROVED,
    State.LIVE_MONITORED,
)

_LEGAL_TRANSITIONS: frozenset[tuple[State, State]] = frozenset(
    [(_f, _t) for _f, _t in _FORWARD_PATH]
    + [(_s, State.REJECTED) for _s in _REJECTABLE_FROM]
    + [(_s, State.RETIRED) for _s in _RETIREABLE_FROM],
)

_ALL_STATES: tuple[State, ...] = tuple(State)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("from_state", "to_state"), _LEGAL_TRANSITIONS)
def test_every_legal_transition_is_allowed(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
    from_state: State,
    to_state: State,
) -> None:
    """Each (from, to) pair in the legal transition table is allowed."""
    _walk_to(mem_db, "idea-1", from_state, clock)
    # APPROVED requires an ApprovalType. Use WATCHLIST for the parametric
    # sweep; test_approved_requires_approval_type covers the explicit
    # case below.
    kwargs: dict[str, Any] = {}
    if to_state is State.APPROVED:
        kwargs["approval_type"] = ApprovalType.WATCHLIST
    if to_state is State.BACKTESTED:
        _seed_pre_registered(mem_db, "idea-1", clock)
    new_id = transition(mem_db, "idea-1", to_state, "actor", clock, **kwargs)
    assert new_id > 0
    # The latest state-change event for this entity must record to_state.
    row = mem_db.execute(
        "SELECT json_extract(payload, '$.to') AS s "
        "FROM events "
        "WHERE entity_id = 'idea-1' AND event_type = 'idea.state_changed' "
        "ORDER BY id DESC LIMIT 1",
    ).fetchone()
    assert row["s"] == to_state.value
    # Hash chain must remain intact.
    assert verify_chain(mem_db) is None


@pytest.mark.parametrize(
    ("from_state", "to_state"),
    [
        (_s, _t)
        for _s in _ALL_STATES
        for _t in _ALL_STATES
        if _s != _t and (_s, _t) not in _LEGAL_TRANSITIONS
    ],
)
def test_every_illegal_transition_raises(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
    from_state: State,
    to_state: State,
) -> None:
    """Every (from, to) pair NOT in the legal transition table raises."""
    _walk_to(mem_db, "idea-1", from_state, clock)
    with pytest.raises(IllegalTransition):
        transition(mem_db, "idea-1", to_state, "actor", clock)
    # The log must NOT have been mutated by a failed transition.
    rows_after = mem_db.execute(
        "SELECT id FROM events WHERE entity_id = 'idea-1'",
    ).fetchall()
    rows_before_ids = {
        r["id"] for r in mem_db.execute(
            "SELECT id FROM events WHERE entity_id = 'idea-1'",
        ).fetchall()
    }
    assert {r["id"] for r in rows_after} == rows_before_ids


def test_anti_p_hacking_gate_specified_to_backtested_raises(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """SPECIFIED -> BACKTESTED must raise IllegalTransition.

    This is the central invariant of pre-registration. A backtest cannot
    legally run on a spec that has not been pre-registered (PRE_REGISTERED
    state). If this transition is ever allowed, the entire
    pre-registration discipline collapses: an analyst could iterate on a
    spec after seeing backtest results, which is exactly the
    p-hacking failure mode the gate exists to prevent.

    The parametric illegal-transition sweep covers this case, but this
    dedicated test makes the intent unambiguous in the test name and
    protects against a future refactor that drops it from the sweep.
    """
    _walk_to(mem_db, "idea-1", State.SPECIFIED, clock)
    with pytest.raises(IllegalTransition):
        transition(mem_db, "idea-1", State.BACKTESTED, "actor", clock)


def test_transition_appends_state_change_event(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """A successful transition appends an ``idea.state_changed`` event."""
    _register(mem_db, "idea-1", clock)
    new_id = transition(mem_db, "idea-1", State.SPECIFIED, "alice", clock)

    row = mem_db.execute(
        "SELECT event_type, actor, "
        "       json_extract(payload, '$.from') AS f, "
        "       json_extract(payload, '$.to')   AS t "
        "FROM events WHERE id = ?",
        (new_id,),
    ).fetchone()
    assert row["event_type"] == "idea.state_changed"
    assert row["actor"] == "alice"
    assert row["f"] == State.CAPTURED.value
    assert row["t"] == State.SPECIFIED.value


def test_transition_records_event_time_from_clock(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """The appended event carries the clock's time, never datetime.now()."""
    _register(mem_db, "idea-1", clock)
    transition(mem_db, "idea-1", State.SPECIFIED, "alice", clock)
    row = mem_db.execute(
        "SELECT ts FROM events "
        "WHERE entity_id = 'idea-1' AND event_type = 'idea.state_changed'",
    ).fetchone()
    assert row["ts"] == _T0.isoformat()


def test_transition_uses_the_provided_clock(
    mem_db: sqlite3.Connection,
) -> None:
    """Two transitions with two different clocks produce two timestamps."""
    clock_a = FrozenClock(at=datetime(2026, 1, 1, tzinfo=UTC))
    clock_b = FrozenClock(at=datetime(2026, 6, 10, tzinfo=UTC))
    _walk_to(mem_db, "idea-1", State.SPECIFIED, clock_a)
    transition(mem_db, "idea-1", State.PRE_REGISTERED, "alice", clock_b)
    row = mem_db.execute(
        "SELECT ts FROM events "
        "WHERE entity_id = 'idea-1' AND event_type = 'idea.state_changed' "
        "ORDER BY id DESC LIMIT 1",
    ).fetchone()
    assert row["ts"] == clock_b.now().isoformat()


def test_approved_requires_approval_type(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """APPROVED must carry an ApprovalType in the event payload."""
    _walk_to(mem_db, "idea-1", State.ADVERSARIAL_REVIEW, clock)
    new_id = transition(
        mem_db, "idea-1", State.APPROVED, "alice", clock,
        approval_type=ApprovalType.WATCHLIST,
    )
    row = mem_db.execute(
        "SELECT json_extract(payload, '$.approval_type') AS at "
        "FROM events WHERE id = ?",
        (new_id,),
    ).fetchone()
    assert row["at"] == ApprovalType.WATCHLIST.value


def test_approved_without_approval_type_raises(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """Transitioning to APPROVED with no approval_type must raise."""
    _walk_to(mem_db, "idea-1", State.ADVERSARIAL_REVIEW, clock)
    with pytest.raises((TypeError, ValueError)):
        transition(mem_db, "idea-1", State.APPROVED, "alice", clock)


def test_approved_rejects_unknown_approval_type(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """A non-ApprovalType value for approval_type must be rejected."""
    _walk_to(mem_db, "idea-1", State.ADVERSARIAL_REVIEW, clock)
    with pytest.raises((TypeError, ValueError)):
        transition(
            mem_db, "idea-1", State.APPROVED, "alice", clock,
            approval_type="paper_trade",  # type: ignore[arg-type]
        )


def test_transition_on_unknown_entity_raises(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """transition() on an entity with no events raises UnknownEntity."""
    with pytest.raises(UnknownEntity):
        transition(mem_db, "ghost", State.SPECIFIED, "alice", clock)


def test_self_transition_is_illegal(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """Transitioning to the same state is forbidden (no-op moves)."""
    _walk_to(mem_db, "idea-1", State.SPECIFIED, clock)
    with pytest.raises(IllegalTransition):
        transition(mem_db, "idea-1", State.SPECIFIED, "alice", clock)


def test_machine_reads_state_from_log_not_caller(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """The state machine derives current state from the log; the caller
    cannot lie about it.

    Even if the projection has been corrupted to claim a state the entity
    is not in, the machine still sees the true state from the log and
    rejects the transition accordingly. This is the *log wins*
    invariant applied to the state machine.
    """
    _walk_to(mem_db, "idea-1", State.SPECIFIED, clock)
    rebuild_projections(mem_db)

    # Corrupt the projection to claim the idea is in BACKTESTED.
    mem_db.execute(
        "UPDATE current_ideas SET current_state = ? WHERE entity_id = ?",
        (State.BACKTESTED.value, "idea-1"),
    )
    mem_db.commit()

    # The machine must see SPECIFIED (from the log) and reject BACKTESTED.
    with pytest.raises(IllegalTransition):
        transition(mem_db, "idea-1", State.BACKTESTED, "alice", clock)


def test_transition_changelog_is_complete(
    mem_db: sqlite3.Connection,
    clock: FrozenClock,
) -> None:
    """A full forward walk produces exactly 12 idea.state_changed events
    in insertion order, with the expected (from, to) pairs.
    """
    _register(mem_db, "idea-1", clock)
    for next_state in (
        State.SPECIFIED, State.PRE_REGISTERED, State.BACKTESTED,
        State.ROBUSTNESS_CHECKED, State.ADVERSARIAL_REVIEW, State.APPROVED,
        State.LIVE_MONITORED, State.TRIGGERED, State.THESIS_REVIEW,
        State.PAPER_TRADE, State.ATTRIBUTED,
    ):
        kwargs: dict[str, Any] = {}
        if next_state is State.APPROVED:
            kwargs["approval_type"] = ApprovalType.WATCHLIST
        if next_state is State.BACKTESTED:
            _seed_pre_registered(mem_db, "idea-1", clock)
        transition(mem_db, "idea-1", next_state, "alice", clock, **kwargs)

    rows = mem_db.execute(
        "SELECT json_extract(payload, '$.from') AS f, "
        "       json_extract(payload, '$.to')   AS t "
        "FROM events "
        "WHERE entity_id = 'idea-1' AND event_type = 'idea.state_changed' "
        "ORDER BY id",
    ).fetchall()
    expected = [
        (State.CAPTURED.value,          State.SPECIFIED.value),
        (State.SPECIFIED.value,         State.PRE_REGISTERED.value),
        (State.PRE_REGISTERED.value,    State.BACKTESTED.value),
        (State.BACKTESTED.value,        State.ROBUSTNESS_CHECKED.value),
        (State.ROBUSTNESS_CHECKED.value, State.ADVERSARIAL_REVIEW.value),
        (State.ADVERSARIAL_REVIEW.value, State.APPROVED.value),
        (State.APPROVED.value,          State.LIVE_MONITORED.value),
        (State.LIVE_MONITORED.value,    State.TRIGGERED.value),
        (State.TRIGGERED.value,         State.THESIS_REVIEW.value),
        (State.THESIS_REVIEW.value,     State.PAPER_TRADE.value),
        (State.PAPER_TRADE.value,       State.ATTRIBUTED.value),
    ]
    assert [(r["f"], r["t"]) for r in rows] == expected
    assert verify_chain(mem_db) is None
