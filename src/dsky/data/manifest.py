"""Append-only provenance for data fetches.

Every data fetch records a single ``data.manifest_recorded`` event via
``append_event`` (the single write path). The events log is the source
of truth; the ``data_manifest`` projection table is derived from it by
``db.projections.rebuild_projections()``.

The ``parquet_path`` field is the absolute path to the Parquet file
containing the bar data. Without a manifest entry, the data is *not*
usable: ``data.bars.load_bars()`` refuses to return rows for a
``parquet_path`` that has no provenance.

The only public function is ``record_manifest()``. There is no public
way to "edit" a manifest entry -- the events log is append-only, and a
"correction" is itself a new event (re-running the fetch).
"""
import sqlite3
from typing import Any

from dsky.clock import Clock
from dsky.db.events import append_event


def record_manifest(  # noqa: PLR0913
    conn: sqlite3.Connection,
    *,
    vendor: str,
    symbol: str,
    fetch_ts: str,
    date_start: str,
    date_end: str,
    row_count: int,
    content_hash: str,
    parquet_path: str,
    clock: Clock,
    actor: str = "code:data",
) -> int:
    """Record provenance for a data fetch. Returns the new event id.

    The event is appended BEFORE the Parquet file is written by
    ``data.bars.write_bars()``. This is the AGENTS.md guarantee:
    provenance is recorded before the data is usable. If the Parquet
    write subsequently fails, the manifest points to a non-existent
    file; ``load_bars()`` will refuse to load it.

    Parameters
    ----------
    conn:
        An open sqlite3 connection (caller owns lifecycle).
    vendor:
        Source identifier, e.g. ``"polygon"``, ``"alpha_vantage"``.
    symbol:
        Ticker symbol, e.g. ``"SPY"``.
    fetch_ts:
        ISO-8601 UTC timestamp of when the fetch happened.
    date_start, date_end:
        ISO-8601 dates (``YYYY-MM-DD``) of the data range fetched.
    row_count:
        Number of rows in the Parquet file.
    content_hash:
        Hex digest of the Parquet file contents.
    parquet_path:
        Absolute path to the Parquet file.
    clock:
        Source of the current timestamp for the event record.
    actor:
        Who/what caused the fetch. Defaults to ``"code:data"``.

    Returns
    -------
    int
        The id of the newly-appended ``data.manifest_recorded`` event.

    """
    payload: dict[str, Any] = {
        "vendor": vendor,
        "symbol": symbol,
        "fetch_ts": fetch_ts,
        "date_start": date_start,
        "date_end": date_end,
        "row_count": row_count,
        "content_hash": content_hash,
        "parquet_path": parquet_path,
    }
    entity_id = f"{vendor}:{symbol}"
    return append_event(
        conn=conn,
        event_type="data.manifest_recorded",
        entity_id=entity_id,
        actor=actor,
        payload=payload,
        clock=clock,
    )


__all__ = ["record_manifest"]
