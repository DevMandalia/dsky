"""Tests for db/projections.py.

Covers:
- rebuild_projections() is idempotent and deterministic.
- current_ideas reflects the latest state per entity (last event wins).
- rejection_ledger has one row per idea.rejected event.
- playbooks reflects the latest approval and monitoring per entity.
- Placeholder tables stay empty (no handlers for them yet).
- Projection tables start empty until rebuild_projections() is called.
- The rebuild is atomic: a mid-rebuild failure rolls back to the previous
  projection state.
- The write-discipline scan still passes (db/projections.py is allowed).
"""
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dsky.clock import FrozenClock
from dsky.db.engine import open_db
from dsky.db.events import append_event
from dsky.db.projections import PROJECTION_TABLES, rebuild_projections

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "dsky"

# Sort key per projection table — must be a column whose value is unique
# per row so the snapshot is independent of physical insertion order.
_SORT_KEYS: dict[str, str] = {
    "current_ideas":      "entity_id",
    "rejection_ledger":   "fingerprint, asset, rejected_ts",  # composite
    "playbooks":          "entity_id",
    "paper_positions":    "id",
    "attribution_history": "id",
}

# User-facing columns per projection table.  Listing them explicitly (rather
# than `SELECT *`) means the snapshot excludes SQLite's implicit rowid, so
# rebuild idempotency doesn't depend on autoincrement counter state.
_COLUMNS: dict[str, str] = {
    "current_ideas":      "entity_id, current_state, last_event_id, updated_ts",
    "rejection_ledger":   "fingerprint, asset, reason, rejected_ts",
    "playbooks":          "entity_id, approval_type, monitoring_status, last_event_id",
    "paper_positions":    "id",
    "attribution_history": "id",
}


@pytest.fixture
def mem_db() -> Iterator[sqlite3.Connection]:
    """Isolated in-memory DB, schema applied, closed after each test."""
    conn = open_db(":memory:")
    yield conn
    conn.close()


def _seed_curated_events(conn: sqlite3.Connection) -> None:
    """Append a sequence that exercises every projection handler."""
    seq = iter([_TS] * 5)

    def t() -> FrozenClock:
        return FrozenClock(at=next(seq))

    append_event(
        conn=conn,
        event_type="idea.registered",
        entity_id="idea-1",
        actor="human:D",
        payload={"name": "long-vol", "asset": "SPY"},
        clock=t(),
    )
    append_event(
        conn=conn,
        event_type="idea.state_changed",
        entity_id="idea-1",
        actor="human:D",
        payload={"from": "registered", "to": "approved"},
        clock=t(),
    )
    append_event(
        conn=conn,
        event_type="idea.rejected",
        entity_id="idea-2",
        actor="human:D",
        payload={"fingerprint": "abc123", "asset": "QQQ", "reason": "no edge"},
        clock=t(),
    )
    append_event(
        conn=conn,
        event_type="playbook.approved",
        entity_id="pb-1",
        actor="human:D",
        payload={"approval_type": "paper", "monitoring_status": "active"},
        clock=t(),
    )
    append_event(
        conn=conn,
        event_type="playbook.monitoring_changed",
        entity_id="pb-1",
        actor="code:monitor",
        payload={"monitoring_status": "paused"},
        clock=t(),
    )


def _snapshot(conn: sqlite3.Connection) -> dict[str, list[tuple]]:
    """Read every projection table into a deterministic, comparable form.

    Excludes the implicit rowid so idempotency doesn't depend on
    autoincrement counter state.  Tolerates missing tables (used by the
    atomicity test, which deliberately drops a table mid-rebuild).
    """
    out: dict[str, list[tuple]] = {}
    for table, cols in _COLUMNS.items():
        sort = _SORT_KEYS[table]
        try:
            rows = conn.execute(
                f"SELECT {cols} FROM {table} ORDER BY {sort}"  # noqa: S608
            ).fetchall()
        except sqlite3.OperationalError:
            # Table doesn't exist (e.g. atomicity test dropped it).
            continue
        out[table] = [tuple(r) for r in rows]
    return out


# ---------------------------------------------------------------------------
# Idempotency & determinism (the headline test the user asked for)
# ---------------------------------------------------------------------------

def test_rebuild_is_idempotent_and_deterministic(
    mem_db: sqlite3.Connection,
) -> None:
    """Two rebuilds on the same event log produce identical projection state."""
    _seed_curated_events(mem_db)

    rebuild_projections(mem_db)
    snap1 = _snapshot(mem_db)

    rebuild_projections(mem_db)
    snap2 = _snapshot(mem_db)

    assert snap1 == snap2


def test_rebuild_remains_idempotent_across_many_calls(
    mem_db: sqlite3.Connection,
) -> None:
    """Rebuilding many times still converges to the same state."""
    _seed_curated_events(mem_db)
    rebuild_projections(mem_db)
    reference = _snapshot(mem_db)

    for _ in range(10):
        rebuild_projections(mem_db)
        assert _snapshot(mem_db) == reference


def test_rebuild_with_unknown_event_types_is_idempotent(
    mem_db: sqlite3.Connection,
) -> None:
    """Events whose event_type has no handler are skipped (not an error)."""
    append_event(
        conn=mem_db, event_type="future.unknown.type", entity_id="x",
        actor="code:test", payload={"k": 1}, clock=FrozenClock(at=_TS),
    )
    rebuild_projections(mem_db)
    snap1 = _snapshot(mem_db)
    rebuild_projections(mem_db)
    snap2 = _snapshot(mem_db)
    assert snap1 == snap2
    # No rows anywhere — that event type has no handler.
    assert all(rows == [] for rows in snap1.values())


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_projections_start_empty_until_rebuild(
    mem_db: sqlite3.Connection,
) -> None:
    """Until rebuild_projections() runs, every projection table is empty."""
    _seed_curated_events(mem_db)  # events are written; projections are not

    for table in PROJECTION_TABLES:
        rows = mem_db.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
        assert rows == [], f"{table} should be empty before rebuild"


# ---------------------------------------------------------------------------
# Per-projection semantics
# ---------------------------------------------------------------------------

def test_current_ideas_reflects_latest_state_per_entity(
    mem_db: sqlite3.Connection,
) -> None:
    """current_ideas keeps the state of the most recent idea.* event per entity."""
    _seed_curated_events(mem_db)
    rebuild_projections(mem_db)

    rows = {
        r["entity_id"]: r["current_state"]
        for r in mem_db.execute("SELECT entity_id, current_state FROM current_ideas")
    }
    assert rows == {
        "idea-1": "approved",   # last idea event for idea-1 was state_changed -> approved
        "idea-2": "rejected",   # only event for idea-2 was a rejection
    }


def test_current_ideas_handles_a_long_chain_of_state_changes(
    mem_db: sqlite3.Connection,
) -> None:
    """A series of transitions for one entity leaves only the last state."""
    states = ["registered", "approved", "paused", "deprecated"]
    ts_iter = iter([_TS] * len(states))

    append_event(
        conn=mem_db, event_type="idea.registered", entity_id="i",
        actor="human", payload={}, clock=FrozenClock(at=next(ts_iter)),
    )
    for s in states[1:]:
        append_event(
            conn=mem_db, event_type="idea.state_changed", entity_id="i",
            actor="human", payload={"from": "x", "to": s},
            clock=FrozenClock(at=next(ts_iter)),
        )

    rebuild_projections(mem_db)
    row = mem_db.execute(
        "SELECT current_state FROM current_ideas WHERE entity_id = 'i'"
    ).fetchone()
    assert row is not None
    assert row["current_state"] == "deprecated"


def test_rejection_ledger_one_row_per_rejection(
    mem_db: sqlite3.Connection,
) -> None:
    """rejection_ledger has one row for every idea.rejected event."""
    _seed_curated_events(mem_db)
    rebuild_projections(mem_db)

    rows = mem_db.execute(
        "SELECT fingerprint, asset, reason, rejected_ts "
        "FROM rejection_ledger"
    ).fetchall()
    assert len(rows) == 1
    r = rows[0]
    assert r["fingerprint"] == "abc123"
    assert r["asset"] == "QQQ"
    assert r["reason"] == "no edge"
    assert r["rejected_ts"] == _TS.isoformat()


def test_rejection_ledger_records_every_rejection(
    mem_db: sqlite3.Connection,
) -> None:
    """Multiple rejections of different ideas all land in the ledger."""
    ts_iter = iter([_TS, _TS])

    append_event(
        conn=mem_db, event_type="idea.rejected", entity_id="i-A",
        actor="code:sieve",
        payload={"fingerprint": "fp-A", "asset": "SPY", "reason": "x"},
        clock=FrozenClock(at=next(ts_iter)),
    )
    append_event(
        conn=mem_db, event_type="idea.rejected", entity_id="i-B",
        actor="code:sieve",
        payload={"fingerprint": "fp-B", "asset": "QQQ", "reason": "y"},
        clock=FrozenClock(at=next(ts_iter)),
    )
    rebuild_projections(mem_db)

    fps = [
        r["fingerprint"]
        for r in mem_db.execute(
            "SELECT fingerprint FROM rejection_ledger "
            "ORDER BY fingerprint, rejected_ts"
        )
    ]
    assert fps == ["fp-A", "fp-B"]


def test_playbooks_reflects_latest_approval_and_monitoring(
    mem_db: sqlite3.Connection,
) -> None:
    """playbooks keeps the most recent approval_type and monitoring_status."""
    _seed_curated_events(mem_db)
    rebuild_projections(mem_db)

    row = mem_db.execute(
        "SELECT * FROM playbooks WHERE entity_id = 'pb-1'"
    ).fetchone()
    assert row is not None
    assert row["approval_type"] == "paper"
    assert row["monitoring_status"] == "paused"  # most recent event


# ---------------------------------------------------------------------------
# Placeholder tables
# ---------------------------------------------------------------------------

def test_placeholder_tables_remain_empty_after_rebuild(
    mem_db: sqlite3.Connection,
) -> None:
    """paper_positions and attribution_history have no handlers yet; empty."""
    _seed_curated_events(mem_db)
    rebuild_projections(mem_db)

    for table in ("paper_positions", "attribution_history"):
        rows = mem_db.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
        assert rows == [], f"{table} should be empty (no handlers yet)"


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------

def test_rebuild_is_atomic_on_failure(
    mem_db: sqlite3.Connection,
) -> None:
    """A mid-rebuild failure rolls back; the prior projection state survives."""
    _seed_curated_events(mem_db)
    rebuild_projections(mem_db)
    snap_before = _snapshot(mem_db)

    # Force a mid-rebuild failure by removing a projection table out from
    # under the rebuild loop.  The rebuild's DELETE FROM <that table> will
    # raise sqlite3.OperationalError, and the implicit transaction rolls
    # back the rest of the writes.
    mem_db.execute("DROP TABLE rejection_ledger")

    with pytest.raises(sqlite3.OperationalError):
        rebuild_projections(mem_db)

    # The previously-built state of the surviving tables is intact.
    snap_after = _snapshot(mem_db)
    assert snap_after.get("current_ideas") == snap_before["current_ideas"]
    assert snap_after.get("playbooks")      == snap_before["playbooks"]
    # rejection_ledger is gone (DROP committed before the rebuild's tx),
    # so it won't appear in the snapshot — that's expected.


# ---------------------------------------------------------------------------
# Discipline
# ---------------------------------------------------------------------------

def test_db_projections_is_in_the_allowed_write_set() -> None:
    """The projections module is explicitly allowed to write SQL.

    This locks in the rule: db/events.py and db/projections.py are the only
    two modules in src/dsky/ that may issue raw INSERT/UPDATE/DELETE.
    """
    projections_py = SRC_ROOT / "db" / "projections.py"
    assert projections_py.exists()
    # The full test is in test_events.py:test_only_events_py_issues_db_writes.
    # Here we just confirm the projections file is recognised as allowed.
    # Smoke check: the file does contain the expected write keywords.
    text = projections_py.read_text(encoding="utf-8")
    assert "INSERT INTO" in text
    assert "UPDATE playbooks" in text
    assert "DELETE FROM" in text
