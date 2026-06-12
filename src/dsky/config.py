"""Project configuration loaded from environment variables."""

import os
from pathlib import Path


def db_path() -> Path:
    """SQLite event-log path (``DSKY_DB``, default ``dsky.db`` in cwd)."""
    return Path(os.environ.get("DSKY_DB", "dsky.db"))


def data_dir() -> Path:
    """Directory for Parquet bar files (``DSKY_DATA``, default ``data``)."""
    return Path(os.environ.get("DSKY_DATA", "data"))


def default_actor() -> str:
    """CLI actor string (``DSKY_ACTOR``, default ``human:cli``)."""
    return os.environ.get("DSKY_ACTOR", "human:cli")
