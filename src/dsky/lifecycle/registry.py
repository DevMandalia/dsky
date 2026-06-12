"""Pre-registration registry: freeze, hash, and record hypothesis specs.

``pre_register()`` is the single entry point for locking a
:class:`~dsky.research.spec.HypothesisSpec` before any backtest may
run. It validates completeness, canonical-serialises the spec, hashes
it, and appends a ``spec.pre_registered`` event via ``append_event``
(the only write path).
"""
import json
import sqlite3

from dsky.clock import Clock
from dsky.db.events import append_event
from dsky.research.spec import HypothesisSpec


class AlreadyPreRegistered(LookupError):  # noqa: N818 -- named per test spec
    """Raised when ``pre_register()`` is called twice for the same entity."""

    def __init__(self, entity_id: str) -> None:
        """Record the entity that was already pre-registered."""
        self.entity_id = entity_id
        super().__init__(
            f"entity {entity_id!r} already has a spec.pre_registered event",
        )


def has_pre_registered(conn: sqlite3.Connection, entity_id: str) -> bool:
    """True iff a ``spec.pre_registered`` event exists for ``entity_id``."""
    row = conn.execute(
        "SELECT 1 FROM events WHERE entity_id = ? AND event_type = ? LIMIT 1",
        (entity_id, "spec.pre_registered"),
    ).fetchone()
    return row is not None


def pre_register(
    conn: sqlite3.Connection,
    entity_id: str,
    spec: HypothesisSpec,
    actor: str,
    clock: Clock,
) -> int:
    """Validate, hash, and record a frozen hypothesis spec.

    The spec is validated for completeness, serialised to canonical
    JSON, and fingerprinted with SHA-256. A ``spec.pre_registered``
    event is appended to the log. After this call the spec in the
    event log is immutable; corrections are new events (re-registration
    is forbidden for the same entity).

    Parameters
    ----------
    conn:
        Open sqlite3 connection (caller owns lifecycle).
    entity_id:
        The idea this spec belongs to.
    spec:
        A :class:`HypothesisSpec`. Must pass
        :meth:`HypothesisSpec.validate_completeness`.
    actor:
        Who/what caused the pre-registration.
    clock:
        Source of the current timestamp.

    Returns
    -------
    int
        The id of the newly-appended ``spec.pre_registered`` event.

    Raises
    ------
    IncompleteSpec
        If the spec fails completeness validation.
    AlreadyPreRegistered
        If a prior ``spec.pre_registered`` event exists for this entity.

    """
    if has_pre_registered(conn, entity_id):
        raise AlreadyPreRegistered(entity_id)

    spec.validate_completeness()
    spec_hash = spec.content_hash()
    canonical = spec.canonical_json()

    payload = {
        "spec": spec.to_canonical_dict(),
        "spec_hash": spec_hash,
        "canonical": canonical,
    }
    return append_event(
        conn=conn,
        event_type="spec.pre_registered",
        entity_id=entity_id,
        actor=actor,
        payload=payload,
        clock=clock,
    )


def load_pre_registered_spec(
    conn: sqlite3.Connection, entity_id: str,
) -> HypothesisSpec:
    """Load the frozen spec from the latest ``spec.pre_registered`` event."""
    row = conn.execute(
        """
        SELECT payload FROM events
        WHERE entity_id = ? AND event_type = 'spec.pre_registered'
        ORDER BY id DESC LIMIT 1
        """,
        (entity_id,),
    ).fetchone()
    if row is None:
        msg = f"no spec.pre_registered event for entity {entity_id!r}"
        raise LookupError(msg)
    payload = json.loads(row["payload"])
    return HypothesisSpec.from_event_payload(payload)


def verify_stored_spec_hash(conn: sqlite3.Connection, entity_id: str) -> bool:
    """True iff the stored ``spec_hash`` matches a recomputation from the spec."""
    row = conn.execute(
        """
        SELECT payload FROM events
        WHERE entity_id = ? AND event_type = 'spec.pre_registered'
        ORDER BY id DESC LIMIT 1
        """,
        (entity_id,),
    ).fetchone()
    if row is None:
        return False
    payload = json.loads(row["payload"])
    spec = HypothesisSpec.from_event_payload(payload)
    return spec.content_hash() == str(payload["spec_hash"])


__all__ = [
    "AlreadyPreRegistered",
    "has_pre_registered",
    "load_pre_registered_spec",
    "pre_register",
    "verify_stored_spec_hash",
]
