"""Append-only event log: the single write path for all decision-relevant facts.

Rule (AGENTS.md): append_event() is the ONLY function in the codebase that
writes to the database.  Nothing in advisory/ may import this module.
"""
import hashlib
import json
import sqlite3
from typing import Any

from dsky.clock import Clock

# Seed hash used for the very first event (no predecessor).
_GENESIS_SEED: str = "dsky-genesis-v1"


def _canonical(payload: dict[str, Any]) -> str:
    """Serialise *payload* to canonical JSON.

    Determinism guarantees:
    - ``sort_keys=True``  — key order is language/insertion independent.
    - ``separators=(',', ':')`` — no optional whitespace anywhere.
    - ``ensure_ascii=True`` — byte-stable across locale variants (default).

    Same dict value → identical byte sequence on every Python version/platform.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _compute_hash(prev_hash: str, canonical_payload: str, ts: str) -> str:
    """SHA-256 over ``prev_hash + canonical_payload + ts`` (UTF-8, no separator).

    The concatenation order is fixed: predecessor → content → timestamp.
    Changing any field in any row invalidates all subsequent hashes.
    """
    data = (prev_hash + canonical_payload + ts).encode()
    return hashlib.sha256(data).hexdigest()


def append_event(  # noqa: PLR0913
    *,
    conn: sqlite3.Connection,
    event_type: str,
    entity_id: str,
    actor: str,
    payload: dict[str, Any],
    clock: Clock,
) -> int:
    """Insert one immutable event and return its new ``id``.

    This is the ONLY function in the codebase permitted to write to the DB.

    Parameters
    ----------
    conn:
        An open sqlite3 connection (caller owns lifecycle).
    event_type:
        Dot-namespaced string, e.g. ``"spec.registered"``.
    entity_id:
        Opaque identifier for the entity this event belongs to.
    actor:
        Who/what caused the event, e.g. ``"human:D"``,
        ``"code:backtest"``, ``"llm:critic"``.
    payload:
        Arbitrary dict that will be canonical-JSON-serialised.
    clock:
        Source of the current timestamp (never datetime.now() directly).

    """
    ts = clock.now().isoformat()
    canonical_payload = _canonical(payload)

    # Fetch the hash of the most-recently inserted row (or the genesis seed).
    row = conn.execute("SELECT hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
    prev_hash: str = row["hash"] if row else _GENESIS_SEED

    event_hash = _compute_hash(prev_hash, canonical_payload, ts)

    cursor = conn.execute(
        """
        INSERT INTO events (ts, event_type, entity_id, actor, payload, prev_hash, hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, event_type, entity_id, actor, canonical_payload, prev_hash, event_hash),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def verify_chain(conn: sqlite3.Connection) -> int | None:
    """Recompute every hash in insertion order and return the first broken id.

    Returns ``None`` if the chain is intact from genesis to the latest row.
    A ``None`` result certifies that no row has been silently edited since it
    was written.
    """
    rows = conn.execute(
        "SELECT id, ts, payload, prev_hash, hash FROM events ORDER BY id"
    ).fetchall()

    expected_prev = _GENESIS_SEED
    for row in rows:
        if row["prev_hash"] != expected_prev:
            return int(row["id"])
        recomputed = _compute_hash(row["prev_hash"], row["payload"], row["ts"])
        if recomputed != row["hash"]:
            return int(row["id"])
        expected_prev = row["hash"]

    return None
