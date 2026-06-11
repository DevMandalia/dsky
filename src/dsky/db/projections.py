"""Read-side projections: derived state, rebuilt by replaying the event log.

Rule (AGENTS.md): "Current state is a projection, never edited in place."
This module is the *only* code path that may write to the projection tables.
The whole point is that anyone holding the events log (and verifying its
hash chain) can reconstruct the same projections — forever, on any machine.
"""
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Tables that derive from the event stream. Each is wiped and rebuilt by
# rebuild_projections(). The tuple order is also the deletion order.
PROJECTION_TABLES: tuple[str, ...] = (
    "current_ideas",
    "rejection_ledger",
    "playbooks",
    "paper_positions",
    "attribution_history",
    "data_manifest",
)


@dataclass(frozen=True, slots=True)
class ReplayedEvent:
    """An event row with its canonical-JSON payload already deserialised.

    Handlers receive this rather than a raw sqlite3.Row so each field is
    typed and `frozen=True` rules out accidental mutation.
    """

    id: int
    ts: str
    event_type: str
    entity_id: str
    actor: str
    payload: dict[str, Any]


ProjectionHandler = Callable[[sqlite3.Connection, ReplayedEvent], None]


# ---------- handlers -------------------------------------------------------

def _handle_idea_registered(conn: sqlite3.Connection, e: ReplayedEvent) -> None:
    _upsert_current_idea(conn, e, "registered")


def _handle_idea_state_changed(conn: sqlite3.Connection, e: ReplayedEvent) -> None:
    _upsert_current_idea(conn, e, str(e.payload["to"]))


def _handle_idea_rejected(conn: sqlite3.Connection, e: ReplayedEvent) -> None:
    _upsert_current_idea(conn, e, "rejected")
    conn.execute(
        """
        INSERT INTO rejection_ledger (fingerprint, asset, reason, rejected_ts)
        VALUES (?, ?, ?, ?)
        """,
        (
            str(e.payload["fingerprint"]),
            str(e.payload["asset"]),
            str(e.payload["reason"]),
            e.ts,
        ),
    )


def _handle_playbook_approved(conn: sqlite3.Connection, e: ReplayedEvent) -> None:
    conn.execute(
        """
        INSERT INTO playbooks
            (entity_id, approval_type, monitoring_status, last_event_id)
        VALUES (?, ?, ?, ?)
        """,
        (
            e.entity_id,
            str(e.payload["approval_type"]),
            str(e.payload["monitoring_status"]),
            e.id,
        ),
    )


def _handle_playbook_monitoring_changed(
    conn: sqlite3.Connection, e: ReplayedEvent,
) -> None:
    conn.execute(
        """
        UPDATE playbooks
           SET monitoring_status = ?, last_event_id = ?
         WHERE entity_id = ?
        """,
        (str(e.payload["monitoring_status"]), e.id, e.entity_id),
    )


def _handle_data_manifest_recorded(
    conn: sqlite3.Connection, e: ReplayedEvent,
) -> None:
    """Upsert the current (vendor, symbol) manifest entry.

    The data_manifest table is keyed on (vendor, symbol) so a refetch
    replaces the previous row in place. Full history of all fetches is
    preserved in the events log; the projection is the *current* state.
    """
    conn.execute(
        """
        INSERT INTO data_manifest
            (vendor, symbol, fetch_ts, date_start, date_end,
             row_count, content_hash, parquet_path, last_event_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(vendor, symbol) DO UPDATE SET
            fetch_ts      = excluded.fetch_ts,
            date_start    = excluded.date_start,
            date_end      = excluded.date_end,
            row_count     = excluded.row_count,
            content_hash  = excluded.content_hash,
            parquet_path  = excluded.parquet_path,
            last_event_id = excluded.last_event_id
        """,
        (
            str(e.payload["vendor"]),
            str(e.payload["symbol"]),
            str(e.payload["fetch_ts"]),
            str(e.payload["date_start"]),
            str(e.payload["date_end"]),
            int(e.payload["row_count"]),
            str(e.payload["content_hash"]),
            str(e.payload["parquet_path"]),
            e.id,
        ),
    )


# ---------- shared helpers -------------------------------------------------

def _upsert_current_idea(
    conn: sqlite3.Connection, e: ReplayedEvent, state: str,
) -> None:
    """Set the current state of an idea. Later events override earlier ones."""
    conn.execute(
        """
        INSERT INTO current_ideas
            (entity_id, current_state, last_event_id, updated_ts)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
            current_state = excluded.current_state,
            last_event_id = excluded.last_event_id,
            updated_ts    = excluded.updated_ts
        """,
        (e.entity_id, state, e.id, e.ts),
    )


# Map of event_type -> handler. Add a new projection by adding a row here.
_HANDLERS: dict[str, ProjectionHandler] = {
    "idea.registered":             _handle_idea_registered,
    "idea.state_changed":          _handle_idea_state_changed,
    "idea.rejected":               _handle_idea_rejected,
    "playbook.approved":           _handle_playbook_approved,
    "playbook.monitoring_changed": _handle_playbook_monitoring_changed,
    "data.manifest_recorded":      _handle_data_manifest_recorded,
}


# ---------- public API -----------------------------------------------------

def rebuild_projections(conn: sqlite3.Connection) -> None:
    """Truncate every projection table and reconstruct it from the event log.

    Properties:
    * Idempotent:    calling N times yields the same state as calling once.
    * Deterministic: events are replayed in id order; no wall-clock is used.
    * Atomic:        all writes happen inside a single implicit transaction,
                     which is committed at the end.  If any handler raises,
                     the implicit rollback restores the pre-rebuild state.

    Event types not registered in ``_HANDLERS`` are recorded in the log but
    do not affect any projection.  This is the expected path for events
    whose projection handlers have not been written yet.
    """
    # All operations below run inside one implicit transaction.  We commit
    # only after every handler has run successfully.  If anything raises,
    # we roll back so the connection is left in either fully-committed or
    # fully-rolled-back state — never half-built.
    try:
        for table in PROJECTION_TABLES:
            conn.execute(f"DELETE FROM {table}")  # noqa: S608

        rows = conn.execute(
            "SELECT id, ts, event_type, entity_id, actor, payload "
            "FROM events ORDER BY id"
        ).fetchall()

        for row in rows:
            raw_payload: Any = json.loads(row["payload"])  # we wrote it ourselves
            if not isinstance(raw_payload, dict):
                # Defensive: a non-dict payload would be a violation of the
                # append_event() contract.  Skip it rather than crash the whole
                # rebuild.
                continue
            event = ReplayedEvent(
                id=int(row["id"]),
                ts=str(row["ts"]),
                event_type=str(row["event_type"]),
                entity_id=str(row["entity_id"]),
                actor=str(row["actor"]),
                payload=raw_payload,
            )
            handler = _HANDLERS.get(event.event_type)
            if handler is not None:
                handler(conn, event)
    except BaseException:
        conn.rollback()
        raise

    conn.commit()
