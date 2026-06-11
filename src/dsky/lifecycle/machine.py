"""Transition function that enforces the legal state graph.

The state machine is the mechanical realisation of AGENTS.md's
*Pre-registration before backtest* invariant. The ``transition()``
function reads the current state from the event log (not from the
caller), validates the move against ``TRANSITIONS``, and on success
appends a single ``idea.state_changed`` event via ``append_event``.

Because state changes go through ``append_event``, every transition is
hash-chained, auditable, and survives the *log wins* discipline: the
projection can be corrupted, but a ``rebuild_projections()`` will
reconstruct the same state from the log.
"""
import json
import sqlite3
from typing import Any

from dsky.clock import Clock
from dsky.db.events import append_event
from dsky.lifecycle.states import ApprovalType, State

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

# Exception names are the public spec (see tests/test_transitions.py);
# the ruff N818 "Error suffix" rule is suppressed with an inline ignore
# on the class lines below.  Renaming would break the test imports and
# the contract that callers rely on.
class IllegalTransition(ValueError):  # noqa: N818
    """Raised when ``transition()`` is asked for a move not in TRANSITIONS.

    The illegal move is recorded in the message so the failure mode is
    easy to diagnose from a stack trace.
    """

    def __init__(
        self, entity_id: str, from_state: State | None, to_state: State,
    ) -> None:
        """Build an exception that names the entity and the illegal move."""
        self.entity_id = entity_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"illegal transition for {entity_id!r}: "
            f"{from_state.value if from_state is not None else None!r} -> "
            f"{to_state.value!r}",
        )


class UnknownEntity(LookupError):  # noqa: N818
    """Raised when ``transition()`` is called for an entity that has no events.

    The state machine cannot derive a current state from an empty log, so
    the only safe behaviour is to refuse the call. Callers must register
    the entity first (via an ``idea.registered`` event).
    """

    def __init__(self, entity_id: str) -> None:
        """Build an exception that names the missing entity."""
        self.entity_id = entity_id
        super().__init__(f"no events for entity {entity_id!r}")


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

# The legal next-state for each state. A state with an empty frozenset
# is terminal (no outgoing transitions). Frozen so accidental mutation
# is caught at runtime.
#
# The shape of this table is the binding spec. The tests in
# tests/test_transitions.py enumerate every legal pair, every illegal
# pair, and (separately) the SPECIFIED -> BACKTESTED anti-p-hacking gate.
TRANSITIONS: dict[State, frozenset[State]] = {
    State.CAPTURED:           frozenset({State.SPECIFIED, State.REJECTED}),
    State.SPECIFIED:          frozenset({State.PRE_REGISTERED, State.REJECTED}),
    State.PRE_REGISTERED:     frozenset({State.BACKTESTED, State.REJECTED}),
    State.BACKTESTED:         frozenset({State.ROBUSTNESS_CHECKED, State.REJECTED}),
    State.ROBUSTNESS_CHECKED: frozenset({State.ADVERSARIAL_REVIEW, State.REJECTED}),
    State.ADVERSARIAL_REVIEW: frozenset({State.APPROVED, State.REJECTED}),
    State.APPROVED:           frozenset({State.LIVE_MONITORED, State.RETIRED}),
    State.LIVE_MONITORED:     frozenset({State.TRIGGERED, State.RETIRED}),
    State.TRIGGERED:          frozenset({State.THESIS_REVIEW}),
    State.THESIS_REVIEW:      frozenset({State.PAPER_TRADE}),
    State.PAPER_TRADE:        frozenset({State.ATTRIBUTED}),
    State.ATTRIBUTED:         frozenset(),
    State.REJECTED:           frozenset(),
    State.RETIRED:            frozenset(),
}


# ---------------------------------------------------------------------------
# Reading the current state from the log
# ---------------------------------------------------------------------------

def _current_state_from_log(
    conn: sqlite3.Connection, entity_id: str,
) -> State | None:
    """Read the current state of ``entity_id`` from the event log.

    Walks the events for this entity in reverse id order and returns the
    state implied by the most recent relevant event. The projection is
    NOT consulted: the log is the source of truth (AGENTS.md: "Current
    state is a projection, never edited in place.").

    Returns ``None`` if the entity has no events at all. Callers must
    raise ``UnknownEntity`` in that case.
    """
    row = conn.execute(
        """
        SELECT event_type, payload FROM events
        WHERE entity_id = ?
          AND event_type IN (
              'idea.registered', 'idea.state_changed', 'idea.rejected'
          )
        ORDER BY id DESC
        LIMIT 1
        """,
        (entity_id,),
    ).fetchone()
    if row is None:
        return None

    event_type = str(row["event_type"])
    payload: dict[str, Any] = json.loads(row["payload"])

    if event_type == "idea.registered":
        # The registration event implies CAPTURED. We don't trust a
        # caller-supplied state field here -- CAPTURED is the only
        # legal starting state in the lifecycle.
        return State.CAPTURED
    if event_type == "idea.state_changed":
        return State(str(payload["to"]))
    if event_type == "idea.rejected":
        return State.REJECTED
    # Defensive: an event_type we don't recognise fell through the IN
    # filter (impossible given the SQL, but cheap to guard).
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transition(  # noqa: PLR0913
    conn: sqlite3.Connection,
    entity_id: str,
    to_state: State,
    actor: str,
    clock: Clock,
    *,
    approval_type: ApprovalType | None = None,
) -> int:
    """Validate the move, then append a single state-change event.

    The current state is read from the event log -- the caller cannot
    lie about it. If the move is not in ``TRANSITIONS``,
    ``IllegalTransition`` is raised and no event is written. If the
    entity has no events, ``UnknownEntity`` is raised.

    On success, an ``idea.state_changed`` event is appended via
    ``append_event`` (the single write path) and the new event's id
    is returned.

    Parameters
    ----------
    conn:
        An open sqlite3 connection (caller owns lifecycle).
    entity_id:
        The opaque identifier of the idea being transitioned.
    to_state:
        The target state.
    actor:
        Who/what caused the transition (e.g. ``"human:D"``,
        ``"code:backtest"``).
    clock:
        Source of the current timestamp.
    approval_type:
        Required iff ``to_state is State.APPROVED``. Must be a member
        of ``ApprovalType`` (``CONTEXT`` / ``WATCHLIST`` /
        ``PAPER_TRADEABLE``). Ignored for non-APPROVED transitions.

    Returns
    -------
    int
        The id of the newly-appended ``idea.state_changed`` event.

    """
    current = _current_state_from_log(conn, entity_id)
    if current is None:
        raise UnknownEntity(entity_id)

    if to_state not in TRANSITIONS[current]:
        raise IllegalTransition(entity_id, current, to_state)

    payload: dict[str, Any] = {"from": current.value, "to": to_state.value}

    if to_state is State.APPROVED:
        if not isinstance(approval_type, ApprovalType):
            msg = (
                f"transition to APPROVED requires an ApprovalType, "
                f"got {type(approval_type).__name__}: {approval_type!r}"
            )
            raise TypeError(msg)
        payload["approval_type"] = approval_type.value

    return append_event(
        conn=conn,
        event_type="idea.state_changed",
        entity_id=entity_id,
        actor=actor,
        payload=payload,
        clock=clock,
    )


__all__ = [
    "TRANSITIONS",
    "ApprovalType",
    "IllegalTransition",
    "State",
    "UnknownEntity",
    "transition",
]
