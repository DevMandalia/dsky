"""Tests for db/engine.py and db/events.py.

Covers:
- Three sequential appends produce a valid, fully-connected hash chain.
- verify_chain() returns None on an intact log.
- Manually tampering with payload in the DB causes verify_chain() to flag
  exactly that row.
- Tampering with prev_hash is also detected.
- Canonical JSON is deterministic: same dict → same bytes, regardless of
  insertion order of keys.
- append_event() accepts a Clock parameter; it never calls datetime.now().
- Write-discipline: no module other than db/events.py issues a raw
  INSERT, UPDATE, or DELETE statement.
"""
import ast
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dsky.clock import FrozenClock
from dsky.db.engine import open_db
from dsky.db.events import _GENESIS_SEED, _canonical, _compute_hash, append_event, verify_chain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "dsky"
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", ".uv", ".pytest_cache", ".mypy_cache",
     ".ruff_cache", "__pycache__", "build", "dist", ".eggs"}
)

_T0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
_T1 = datetime(2024, 1, 15, 12, 0, 1, tzinfo=UTC)
_T2 = datetime(2024, 1, 15, 12, 0, 2, tzinfo=UTC)


@pytest.fixture
def mem_db() -> Iterator[sqlite3.Connection]:
    """Isolated in-memory DB, schema applied, closed after each test."""
    conn = open_db(":memory:")
    yield conn
    conn.close()


def _append(conn: sqlite3.Connection, seq: int, ts: datetime) -> int:
    """Convenience wrapper: append a numbered test event."""
    return append_event(
        conn=conn,
        event_type="test.event",
        entity_id="entity-1",
        actor="code:test",
        payload={"seq": seq, "note": "test"},
        clock=FrozenClock(at=ts),
    )


def _rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM events ORDER BY id").fetchall()


# ---------------------------------------------------------------------------
# Chain integrity
# ---------------------------------------------------------------------------

def test_three_events_form_valid_chain(mem_db: sqlite3.Connection) -> None:
    """Appending three events produces a correctly linked hash chain."""
    id1 = _append(mem_db, 1, _T0)
    id2 = _append(mem_db, 2, _T1)
    id3 = _append(mem_db, 3, _T2)

    assert [id1, id2, id3] == [1, 2, 3]

    rows = _rows(mem_db)
    assert len(rows) == 3

    # Row 1: genesis seed is the predecessor.
    assert rows[0]["prev_hash"] == _GENESIS_SEED
    ts0 = rows[0]["ts"]
    expected_h0 = _compute_hash(_GENESIS_SEED, rows[0]["payload"], ts0)
    assert rows[0]["hash"] == expected_h0

    # Row 2: row-1's hash is the predecessor.
    assert rows[1]["prev_hash"] == rows[0]["hash"]
    expected_h1 = _compute_hash(rows[0]["hash"], rows[1]["payload"], rows[1]["ts"])
    assert rows[1]["hash"] == expected_h1

    # Row 3: row-2's hash is the predecessor.
    assert rows[2]["prev_hash"] == rows[1]["hash"]
    expected_h2 = _compute_hash(rows[1]["hash"], rows[2]["payload"], rows[2]["ts"])
    assert rows[2]["hash"] == expected_h2


def test_verify_chain_returns_none_on_intact_log(mem_db: sqlite3.Connection) -> None:
    """verify_chain() returns None when no row has been tampered with."""
    _append(mem_db, 1, _T0)
    _append(mem_db, 2, _T1)
    _append(mem_db, 3, _T2)

    assert verify_chain(mem_db) is None


def test_verify_chain_empty_db_returns_none(mem_db: sqlite3.Connection) -> None:
    """An empty log is trivially intact."""
    assert verify_chain(mem_db) is None


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------

def test_tamper_payload_flags_correct_row(mem_db: sqlite3.Connection) -> None:
    """Editing a row's payload in the DB causes verify_chain() to return that row's id."""
    _append(mem_db, 1, _T0)
    _append(mem_db, 2, _T1)
    _append(mem_db, 3, _T2)

    # Tamper: silently mutate the payload of row 2.
    tampered = _canonical({"seq": 999, "note": "tampered"})
    mem_db.execute("UPDATE events SET payload = ? WHERE id = 2", (tampered,))
    mem_db.commit()

    assert verify_chain(mem_db) == 2


def test_tamper_first_row_flags_id_1(mem_db: sqlite3.Connection) -> None:
    """Tampering with the genesis row is caught at id 1."""
    _append(mem_db, 1, _T0)
    _append(mem_db, 2, _T1)

    tampered = _canonical({"seq": 0, "note": "genesis-tamper"})
    mem_db.execute("UPDATE events SET payload = ? WHERE id = 1", (tampered,))
    mem_db.commit()

    assert verify_chain(mem_db) == 1


def test_tamper_prev_hash_flags_correct_row(mem_db: sqlite3.Connection) -> None:
    """Editing prev_hash directly is also caught."""
    _append(mem_db, 1, _T0)
    _append(mem_db, 2, _T1)
    _append(mem_db, 3, _T2)

    mem_db.execute("UPDATE events SET prev_hash = 'forged' WHERE id = 3")
    mem_db.commit()

    assert verify_chain(mem_db) == 3


def test_tamper_last_row_flags_correctly(mem_db: sqlite3.Connection) -> None:
    """Tampering with the most-recent row is caught even though no successor exists."""
    _append(mem_db, 1, _T0)
    _append(mem_db, 2, _T1)
    _append(mem_db, 3, _T2)

    tampered = _canonical({"seq": 99})
    mem_db.execute("UPDATE events SET payload = ? WHERE id = 3", (tampered,))
    mem_db.commit()

    assert verify_chain(mem_db) == 3


# ---------------------------------------------------------------------------
# Canonical JSON determinism
# ---------------------------------------------------------------------------

def test_canonical_json_is_deterministic() -> None:
    """Same dict value always produces the same byte sequence."""
    d = {"z": 1, "a": 2, "m": [3, 4], "b": {"y": False, "x": None}}
    first  = _canonical(d)
    second = _canonical({"m": [3, 4], "z": 1, "b": {"x": None, "y": False}, "a": 2})
    assert first == second
    # And confirm the form: sorted keys, no spaces.
    assert first == '{"a":2,"b":{"x":null,"y":false},"m":[3,4],"z":1}'


def test_canonical_json_nested_sort() -> None:
    """Nested dicts are also key-sorted."""
    result = _canonical({"outer": {"z": 1, "a": 2}})
    assert result == '{"outer":{"a":2,"z":1}}'


# ---------------------------------------------------------------------------
# Clock discipline: append_event takes clock, never calls datetime.now() itself
# ---------------------------------------------------------------------------

def test_append_event_uses_clock_not_datetime_now(mem_db: sqlite3.Connection) -> None:
    """The timestamp stored in the DB must equal FrozenClock.now().isoformat()."""
    fixed = datetime(2024, 6, 1, 9, 0, 0, tzinfo=UTC)
    append_event(
        conn=mem_db,
        event_type="check.ts",
        entity_id="ent",
        actor="code:test",
        payload={},
        clock=FrozenClock(at=fixed),
    )
    row = mem_db.execute("SELECT ts FROM events").fetchone()
    assert row["ts"] == fixed.isoformat()


# ---------------------------------------------------------------------------
# Write-discipline: only db/events.py may issue raw INSERT/UPDATE/DELETE
# ---------------------------------------------------------------------------

_WRITE_STATEMENTS = ("INSERT ", "UPDATE ", "DELETE ")

# The scan covers only production source (src/dsky/).  Test code is
# intentionally excluded: tamper-detection tests must reach past the normal
# API to corrupt the DB directly — that is their purpose.  The write-path
# rule in AGENTS.md targets production modules, not the test suite.
_WRITE_SCAN_ROOT = SRC_ROOT
_ALLOWED_WRITE_FILE = SRC_ROOT / "db" / "events.py"


def _iter_src_python_files() -> Iterator[Path]:
    for path in sorted(_WRITE_SCAN_ROOT.rglob("*.py")):
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        yield path


def _has_raw_write_string(text: str) -> list[str]:
    """Return SQL write keywords found in string literals in the AST."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            upper = node.value.upper()
            found.extend(kw.strip() for kw in _WRITE_STATEMENTS if kw in upper)
    return found


def test_only_events_py_issues_db_writes() -> None:
    """No production module other than db/events.py may contain raw
    INSERT/UPDATE/DELETE strings embedded in SQL execute() calls.

    Scope: src/dsky/ only.  Test code is excluded by design (tamper tests
    must corrupt the DB directly; that is how they prove verify_chain works).
    """
    allowed = _ALLOWED_WRITE_FILE.resolve()
    offenders: list[str] = []
    for path in _iter_src_python_files():
        if path.resolve() == allowed:
            continue
        text = path.read_text(encoding="utf-8")
        keywords = _has_raw_write_string(text)
        if keywords:
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"  {rel}: contains {keywords}")
    assert not offenders, (
        "Raw INSERT/UPDATE/DELETE found outside db/events.py in production source:\n"
        + "\n".join(offenders)
    )
