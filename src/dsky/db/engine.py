"""SQLite connection helper: WAL mode, foreign keys, schema bootstrap."""
import sqlite3
from importlib.resources import files
from pathlib import Path


def _schema_sql() -> str:
    """Read schema.sql from the package, whether installed or from source."""
    try:
        return (files("dsky.db") / "schema.sql").read_text(encoding="utf-8")
    except (TypeError, FileNotFoundError):
        # Fallback for editable installs that haven't materialised resources yet.
        return (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database at *path*.

    Applies one-time setup on every new connection:
    * WAL journal mode — allows readers while a writer is active.
    * Foreign-key enforcement — SQLite skips FK checks by default.
    * Schema bootstrap via schema.sql — idempotent (CREATE IF NOT EXISTS).

    The caller owns the returned connection and is responsible for closing it.
    Pass ``path=":memory:"`` for an isolated in-memory database in tests.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_schema_sql())
    conn.commit()
    return conn
