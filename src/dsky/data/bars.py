"""Bar data with the structural time-discipline gate.

Bars are stored as Parquet files keyed by symbol. Every row carries:
- ``event_time`` -- the bar's own timestamp (e.g. the trading day).
- ``available_time`` -- when that bar would realistically be known live.

The AGENTS.md invariant -- "the backtest may only read points where
``available_time <= decision_time``" -- is enforced *structurally* by
``load_bars()``, which is the only public read path. The caller cannot
bypass the gate.

Available-time rule
-------------------
By default, the available_time of a daily bar is the next NYSE market
open (9:30 AM ET, Monday-Friday). The rule is configurable: pass a
custom callable to ``compute_available_time()`` or to ``write_bars()``.

Manifest
--------
Every Parquet file has a corresponding ``data.manifest_recorded`` event
in the events log, written *before* the Parquet file by
``write_bars()``. ``load_bars()`` refuses to return data whose
``parquet_path`` has no manifest entry.
"""
import hashlib
import io
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from dsky.clock import Clock
from dsky.data.manifest import record_manifest

# ---------------------------------------------------------------------------
# Available-time rule
# ---------------------------------------------------------------------------

# NYSE opens at 9:30 AM ET. We treat ET as a fixed UTC-5 offset (EST);
# DST-aware handling is a future refinement. The available-time rule
# treats the bar's date in ET, not UTC, so a daily bar stamped
# 2026-01-12 00:00:00 UTC belongs to Monday 2026-01-12 in ET.
_ET_OFFSET = timedelta(hours=-5)
ET = timezone(_ET_OFFSET, name="ET")
_NYSE_OPEN_HOUR = 9
_NYSE_OPEN_MINUTE = 30
_WEEKEND = frozenset({5, 6})  # 5 = Saturday, 6 = Sunday


def next_nyse_open(event_time: datetime) -> datetime:
    """Return the next NYSE market open after ``event_time``.

    A daily bar's ``available_time`` is the NEXT market open, not the
    current one: a Monday bar (covering 9:30 ET to 16:00 ET) is
    published on Tuesday at 9:30 ET, because the bar cannot be known
    until the trading day is over. The rule skips weekends but not
    holidays (the holiday calendar is a future refinement).
    """
    if event_time.tzinfo is None:
        utc_dt = event_time.replace(tzinfo=UTC)
    else:
        utc_dt = event_time.astimezone(UTC)

    # Advance one calendar day in UTC, then anchor at 9:30 AM ET.
    # This guarantees the next market open is strictly after the bar.
    next_day_utc = utc_dt.date() + timedelta(days=1)
    candidate_et = datetime(
        next_day_utc.year, next_day_utc.month, next_day_utc.day,
        _NYSE_OPEN_HOUR, _NYSE_OPEN_MINUTE,
        tzinfo=ET,
    )
    while candidate_et.weekday() in _WEEKEND:
        candidate_et += timedelta(days=1)
    return candidate_et.astimezone(UTC)


AvailableTimeRule = Callable[[datetime], datetime]


def compute_available_time(
    event_time: datetime,
    *,
    rule: AvailableTimeRule | None = None,
) -> datetime:
    """Compute the available_time for a bar at ``event_time``.

    By default, uses :func:`next_nyse_open`. If a custom ``rule`` is
    supplied, it is used instead -- this is the spec's "configurable"
    hook. The rule must return a tz-aware datetime (in any timezone;
    it is converted to UTC before comparison).
    """
    if rule is None:
        return next_nyse_open(event_time)
    return rule(event_time)


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

def _content_hash(bars: pl.DataFrame) -> str:
    """Stable SHA-256 over the bar contents.

    Independent of row insertion order (sorted by ``event_time``) and
    independent of the Parquet writer's byte-level nondeterminism. The
    hash is a fingerprint of *the data*, not of the file bytes: a
    re-fetch of identical data produces an identical hash.
    """
    sorted_bars = bars.sort("event_time") if "event_time" in bars.columns else bars
    buf = io.BytesIO()
    sorted_bars.write_parquet(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ---------------------------------------------------------------------------
# write_bars
# ---------------------------------------------------------------------------

def write_bars(  # noqa: PLR0913
    conn: sqlite3.Connection,
    symbol: str,
    bars: pl.DataFrame,
    *,
    parquet_path: str | Path,
    vendor: str,
    date_start: str,
    date_end: str,
    clock: Clock,
    actor: str = "code:data",
    available_time_rule: AvailableTimeRule | None = None,
) -> int:
    """Write bars to Parquet and record provenance in the events log.

    The manifest event is recorded *before* the Parquet file is written.
    This is the AGENTS.md guarantee: provenance first, data second. If
    the Parquet write fails after the manifest is recorded, the manifest
    points to a non-existent file; ``load_bars()`` will refuse to load
    it.

    Parameters
    ----------
    conn:
        Open sqlite3 connection (caller owns lifecycle).
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    bars:
        Polars DataFrame. Must include an ``event_time`` column of
        dtype ``Datetime``. If ``available_time`` is missing, it is
        computed from ``event_time`` using the default or supplied
        rule.
    parquet_path:
        Destination path for the Parquet file. Parent dirs are
        created.
    vendor:
        Source identifier (e.g. ``"polygon"``).
    date_start, date_end:
        ISO-8601 dates (``YYYY-MM-DD``) of the data range.
    clock:
        Source of the current timestamp for the manifest event.
    actor:
        Who/what caused the fetch. Defaults to ``"code:data"``.
    available_time_rule:
        Optional custom function to compute ``available_time``. If
        supplied, used to fill in missing ``available_time`` values.

    Returns
    -------
    int
        The id of the newly-appended ``data.manifest_recorded`` event.

    Raises
    ------
    ValueError
        If ``bars`` is missing the required ``event_time`` column.

    """
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if "event_time" not in bars.columns:
        msg = f"bars must have an 'event_time' column; got {bars.columns!r}"
        raise ValueError(msg)

    # Fill in available_time if missing.
    if "available_time" not in bars.columns:
        event_times = bars["event_time"].to_list()
        available_times = [
            compute_available_time(et, rule=available_time_rule)
            for et in event_times
        ]
        bars = bars.with_columns(pl.Series("available_time", available_times))

    # Compute the content hash BEFORE writing the file so the manifest
    # carries a verifiable fingerprint of the data.
    content_hash_value = _content_hash(bars)
    row_count = bars.height
    fetch_ts = clock.now().isoformat()
    resolved_path = str(parquet_path.resolve())

    # Step 1: record provenance FIRST. If the Parquet write fails after
    # this, the manifest points to a missing file; load_bars() will
    # refuse to use it.
    manifest_event_id = record_manifest(
        conn=conn,
        vendor=vendor,
        symbol=symbol,
        fetch_ts=fetch_ts,
        date_start=date_start,
        date_end=date_end,
        row_count=row_count,
        content_hash=content_hash_value,
        parquet_path=resolved_path,
        clock=clock,
        actor=actor,
    )

    # Step 2: write the Parquet file.
    bars.write_parquet(parquet_path)

    return manifest_event_id


# ---------------------------------------------------------------------------
# Lookup + read
# ---------------------------------------------------------------------------

def current_manifest(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    vendor: str | None = None,
) -> dict[str, Any] | None:
    """Look up the current manifest entry for ``symbol``.

    The ``data_manifest`` projection is the source of truth for the
    *current* (vendor, symbol) pair. If ``vendor`` is given, narrows
    to that vendor. Otherwise, returns the latest across all vendors.

    Returns ``None`` if no manifest exists.
    """
    if vendor is None:
        row = conn.execute(
            "SELECT * FROM data_manifest WHERE symbol = ? "
            "ORDER BY fetch_ts DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM data_manifest WHERE symbol = ? AND vendor = ? "
            "ORDER BY fetch_ts DESC LIMIT 1",
            (symbol, vendor),
        ).fetchone()

    if row is None:
        return None
    return dict(row)


def load_bars(
    conn: sqlite3.Connection,
    symbol: str,
    decision_time: datetime,
) -> pl.DataFrame:
    """Load bars for ``symbol`` at or before ``decision_time``.

    The structural time-discipline gate. The caller passes
    ``decision_time`` and only rows whose ``available_time`` is at or
    before that time are returned. ``load_bars`` is the only public
    read path: the caller cannot bypass the gate.

    Provenance is required: if no ``data.manifest_recorded`` event
    exists for ``symbol`` (i.e. the ``data_manifest`` projection is
    empty for it), ``LookupError`` is raised. If the manifest points
    to a non-existent file, ``FileNotFoundError`` is raised.

    Parameters
    ----------
    conn:
        Open sqlite3 connection (caller owns lifecycle). The
        ``data_manifest`` projection must be up to date; call
        ``db.projections.rebuild_projections(conn)`` after writes.
    symbol:
        Ticker symbol to load.
    decision_time:
        The decision instant. Naive datetimes are treated as UTC.

    Returns
    -------
    pl.DataFrame
        Bars whose ``available_time <= decision_time``. All other
        rows are filtered out.

    Raises
    ------
    LookupError
        If no manifest entry exists for ``symbol``.
    FileNotFoundError
        If the manifest points to a parquet file that does not exist.

    """
    manifest = current_manifest(conn, symbol)
    if manifest is None:
        msg = f"no manifest for symbol {symbol!r}; data is not usable"
        raise LookupError(msg)

    parquet_path = Path(manifest["parquet_path"])
    if not parquet_path.exists():
        msg = (
            f"manifest for {symbol!r} points to non-existent file: "
            f"{parquet_path}"
        )
        raise FileNotFoundError(msg)

    bars = pl.read_parquet(parquet_path)

    # Normalise decision_time to UTC for comparison.
    if decision_time.tzinfo is None:
        decision_time_utc = decision_time.replace(tzinfo=UTC)
    else:
        decision_time_utc = decision_time.astimezone(UTC)

    return bars.filter(pl.col("available_time") <= decision_time_utc)


__all__ = [
    "AvailableTimeRule",
    "compute_available_time",
    "current_manifest",
    "load_bars",
    "next_nyse_open",
    "write_bars",
]
