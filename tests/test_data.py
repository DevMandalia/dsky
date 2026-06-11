"""Tests for data/bars.py, data/manifest.py, and the time-discipline gate.

All tests use synthetic bars and monkeypatched HTTP. The live Polygon
API is exercised separately (one-off, manual, with a real API key).
"""
import hashlib
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Self

import polars as pl
import pytest

from dsky.clock import FrozenClock
from dsky.data.bars import (
    compute_available_time,
    current_manifest,
    load_bars,
    next_nyse_open,
    write_bars,
)
from dsky.data.manifest import record_manifest
from dsky.data.providers.polygon import PolygonProvider
from dsky.db.engine import open_db
from dsky.db.events import verify_chain
from dsky.db.projections import rebuild_projections

# A Monday 12:00 UTC -- well after the 14:30 UTC previous-day open.
_T0 = datetime(2026, 1, 12, 12, 0, 0, tzinfo=UTC)


def _to_utc(d: datetime) -> datetime:
    """Normalise any datetime to datetime.timezone.utc for comparison.

    Polars round-trips tz-aware values through ``zoneinfo.ZoneInfo``,
    so a value written as ``datetime.timezone.utc`` reads back as
    ``ZoneInfo("UTC")``. The two represent the same instant but are
    not ``==`` because ``datetime.__eq__`` compares the tzinfo object.
    This helper collapses the comparison to the UTC instant.
    """
    if d.tzinfo is None:
        return d.replace(tzinfo=UTC)
    return d.astimezone(UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_db() -> Iterator[sqlite3.Connection]:
    """Fresh in-memory DB with the dsky schema applied."""
    conn = open_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(at=_T0)


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    """A fresh directory for Parquet files; cleaned up by pytest."""
    d = tmp_path / "parquet"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Synthetic bar data
# ---------------------------------------------------------------------------

def _synthetic_spy_bars() -> pl.DataFrame:
    """3 daily bars for SPY, dated Mon/Tue/Wed 2026-01-12/13/14."""
    return pl.DataFrame({
        "event_time": [
            datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 13, 0, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 14, 0, 0, 0, tzinfo=UTC),
        ],
        "open":   [450.0, 452.0, 451.0],
        "high":   [455.0, 456.0, 454.0],
        "low":    [449.0, 451.0, 450.0],
        "close":  [454.0, 455.0, 453.0],
        "volume": [100_000_000, 110_000_000, 105_000_000],
    })


# ---------------------------------------------------------------------------
# Available-time rule
# ---------------------------------------------------------------------------

class TestNextNyseOpen:
    """The default rule: next NYSE market open, 9:30 AM ET, weekdays only."""

    def test_monday_bar_returns_tuesday_open(self) -> None:
        # Monday 2026-01-12 -> Tuesday 2026-01-13 09:30 ET = 14:30 UTC (EST, Jan).
        result = next_nyse_open(datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC))
        assert result == datetime(2026, 1, 13, 14, 30, tzinfo=UTC)

    def test_friday_bar_returns_monday_open(self) -> None:
        # Friday 2026-01-16 -> Monday 2026-01-19 09:30 ET = 14:30 UTC.
        result = next_nyse_open(datetime(2026, 1, 16, 16, 0, 0, tzinfo=UTC))
        assert result == datetime(2026, 1, 19, 14, 30, tzinfo=UTC)

    def test_saturday_bar_returns_monday_open(self) -> None:
        # Saturday 2026-01-17 -> Monday 2026-01-19.
        result = next_nyse_open(datetime(2026, 1, 17, 0, 0, 0, tzinfo=UTC))
        assert result == datetime(2026, 1, 19, 14, 30, tzinfo=UTC)

    def test_sunday_bar_returns_monday_open(self) -> None:
        # Sunday 2026-01-18 -> Monday 2026-01-19.
        result = next_nyse_open(datetime(2026, 1, 18, 0, 0, 0, tzinfo=UTC))
        assert result == datetime(2026, 1, 19, 14, 30, tzinfo=UTC)

    def test_returns_utc_aware_datetime(self) -> None:
        result = next_nyse_open(datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC))
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(0)


def test_compute_available_time_uses_default_when_no_rule() -> None:
    event_time = datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC)
    assert compute_available_time(event_time) == next_nyse_open(event_time)


def test_compute_available_time_accepts_custom_rule() -> None:
    """A user-supplied rule overrides the default. The rule is the spec."""
    event_time = datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC)

    def add_one_hour(et: datetime) -> datetime:
        return et + timedelta(hours=1)

    assert compute_available_time(event_time, rule=add_one_hour) == \
        event_time + timedelta(hours=1)


# ---------------------------------------------------------------------------
# write_bars + record_manifest: provenance is recorded BEFORE the file
# ---------------------------------------------------------------------------

class TestWriteBarsRecordsManifest:
    """The manifest event must exist in the log after write_bars() returns."""

    def test_manifest_event_in_log(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"

        event_id = write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )

        row = mem_db.execute(
            "SELECT event_type, entity_id, actor, payload "
            "FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        assert row["event_type"] == "data.manifest_recorded"
        assert row["entity_id"] == "synthetic:SPY"
        assert row["actor"] == "code:data"

        payload = json.loads(row["payload"])
        assert payload["vendor"] == "synthetic"
        assert payload["symbol"] == "SPY"
        assert payload["date_start"] == "2026-01-12"
        assert payload["date_end"] == "2026-01-14"
        assert payload["row_count"] == 3
        assert len(payload["content_hash"]) == 64  # SHA-256 hex
        assert payload["parquet_path"] == str(path.resolve())
        assert payload["fetch_ts"] == clock.now().isoformat()

    def test_parquet_file_written(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"

        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )

        assert path.exists()
        loaded = pl.read_parquet(path)
        assert loaded.height == 3
        assert "event_time" in loaded.columns
        assert "available_time" in loaded.columns

    def test_manifest_records_correct_content_hash(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        """The recorded content hash matches SHA-256 of the written file."""
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"

        event_id = write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )

        recorded_hash = json.loads(
            mem_db.execute("SELECT payload FROM events WHERE id = ?",
                           (event_id,)).fetchone()["payload"],
        )["content_hash"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert recorded_hash == actual_hash

    def test_hash_chain_remains_intact(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        assert verify_chain(mem_db) is None


class TestWriteBarsFillsAvailableTime:
    """If the caller didn't supply available_time, compute it from event_time."""

    def test_fills_in_default_available_time(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()  # no available_time column
        path = parquet_dir / "SPY.parquet"

        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )

        loaded = pl.read_parquet(path)
        # Mon -> Tue 14:30 UTC, Tue -> Wed 14:30 UTC, Wed -> Thu 14:30 UTC.
        actual = [_to_utc(dt) for dt in loaded["available_time"].to_list()]
        assert actual == [
            datetime(2026, 1, 13, 14, 30, tzinfo=UTC),
            datetime(2026, 1, 14, 14, 30, tzinfo=UTC),
            datetime(2026, 1, 15, 14, 30, tzinfo=UTC),
        ]

    def test_respects_supplied_available_time(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        """If available_time is supplied, do NOT overwrite it."""
        far_future = datetime(2099, 1, 1, tzinfo=UTC)
        bars = _synthetic_spy_bars().with_columns(
            pl.Series("available_time", [far_future, far_future, far_future]),
        )
        path = parquet_dir / "SPY.parquet"

        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )

        loaded = pl.read_parquet(path)
        actual = [_to_utc(dt) for dt in loaded["available_time"].to_list()]
        assert actual == [far_future, far_future, far_future]

    def test_custom_rule_applied_when_filling(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"

        def rule(et: datetime) -> datetime:
            # available_time = event_time + 1 minute (deterministic, simple).
            return et + timedelta(minutes=1)

        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock, available_time_rule=rule,
        )

        loaded = pl.read_parquet(path)
        actual = [_to_utc(dt) for dt in loaded["available_time"].to_list()]
        assert actual == [
            datetime(2026, 1, 12, 0, 1, tzinfo=UTC),
            datetime(2026, 1, 13, 0, 1, tzinfo=UTC),
            datetime(2026, 1, 14, 0, 1, tzinfo=UTC),
        ]


class TestManifestProvenance:
    """The data_manifest projection is rebuilt from the events log."""

    def test_projection_empty_before_rebuild(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        rows = mem_db.execute("SELECT * FROM data_manifest").fetchall()
        assert rows == []

    def test_projection_populated_after_rebuild(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        rebuild_projections(mem_db)
        rows = mem_db.execute("SELECT * FROM data_manifest").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["vendor"] == "synthetic"
        assert row["symbol"] == "SPY"
        assert row["row_count"] == 3

    def test_projection_upserts_on_refetch(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        clock2 = FrozenClock(at=_T0 + timedelta(hours=1))

        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock2,
        )
        rebuild_projections(mem_db)
        rows = mem_db.execute("SELECT * FROM data_manifest").fetchall()
        # UPSERT on (vendor, symbol): one row, but the fetch_ts is updated.
        assert len(rows) == 1
        assert rows[0]["fetch_ts"] == clock2.now().isoformat()

    def test_projection_separates_vendors(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path1 = parquet_dir / "poly_SPY.parquet"
        path2 = parquet_dir / "av_SPY.parquet"

        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path1,
            vendor="polygon", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path2,
            vendor="alpha_vantage", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        rebuild_projections(mem_db)
        rows = mem_db.execute(
            "SELECT vendor FROM data_manifest WHERE symbol = 'SPY' ORDER BY vendor",
        ).fetchall()
        assert [r["vendor"] for r in rows] == ["alpha_vantage", "polygon"]


# ---------------------------------------------------------------------------
# load_bars: the time-discipline gate (the headline test)
# ---------------------------------------------------------------------------

class TestLoadBarsTimeGate:
    """load_bars is the ONLY public read path. It must enforce
    available_time <= decision_time structurally -- the caller cannot
    bypass the gate."""

    def test_returns_only_rows_past_or_at_decision_time(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        rebuild_projections(mem_db)

        # Wednesday 2026-01-14 15:00 UTC.
        # Monday  bar (avail Tue 14:30)  -> included
        # Tuesday bar (avail Wed 14:30)  -> included
        # Wednesday bar (avail Thu 14:30) -> excluded
        decision_time = datetime(2026, 1, 14, 15, 0, 0, tzinfo=UTC)
        result = load_bars(mem_db, "SPY", decision_time)

        assert result.height == 2
        ets = [_to_utc(t) for t in result["event_time"].to_list()]
        assert ets == [
            datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 13, 0, 0, 0, tzinfo=UTC),
        ]

    def test_exact_boundary_included(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        """A bar with available_time EQUAL TO decision_time is INCLUDED.

        This is the central correctness case: the spec is
        ``available_time <= decision_time`` (not strictly less than).
        """
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        rebuild_projections(mem_db)

        # The Tuesday bar's available_time is exactly 2026-01-14 14:30 UTC.
        # Set decision_time to that exact instant. The Tuesday bar is at
        # the boundary; it must be INCLUDED.
        decision_time = datetime(2026, 1, 14, 14, 30, 0, tzinfo=UTC)
        result = load_bars(mem_db, "SPY", decision_time)

        ets = [_to_utc(t) for t in result["event_time"].to_list()]
        # Monday  (avail Tue 14:30)  -> included
        # Tuesday (avail Wed 14:30)  -> included (boundary, ==)
        # Wednesday (avail Thu 14:30) -> excluded
        assert datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC) in ets
        assert datetime(2026, 1, 13, 0, 0, 0, tzinfo=UTC) in ets
        assert datetime(2026, 1, 14, 0, 0, 0, tzinfo=UTC) not in ets

    def test_future_row_excluded(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        """A bar with available_time > decision_time is EXCLUDED."""
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        rebuild_projections(mem_db)

        # Monday 12:00 UTC. The Monday bar (avail Tue 14:30) is the
        # closest available bar, but its available_time is 26.5 hours
        # in the future. Nothing should be returned.
        decision_time = datetime(2026, 1, 12, 12, 0, 0, tzinfo=UTC)
        result = load_bars(mem_db, "SPY", decision_time)
        assert result.height == 0

    def test_no_manifest_raises_lookup_error(
        self, mem_db,
    ) -> None:
        with pytest.raises(LookupError):
            load_bars(mem_db, "GHOST", datetime(2026, 1, 12, tzinfo=UTC))

    def test_no_manifest_after_rebuild_still_raises(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        """If the parquet file exists but the manifest was never written,
        load_bars must refuse. Provenance is required to read data.
        """
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        bars.write_parquet(path)  # write the file directly, no manifest
        rebuild_projections(mem_db)
        with pytest.raises(LookupError):
            load_bars(mem_db, "SPY", datetime(2030, 1, 1, tzinfo=UTC))

    def test_manifest_pointing_to_missing_file_raises(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        """If the manifest points to a file that doesn't exist, refuse.

        This is the failure mode when the Parquet write fails AFTER the
        manifest is recorded: the manifest points to nothing. load_bars
        must refuse to silently return empty data.
        """
        bars = _synthetic_spy_bars()
        fake_path = parquet_dir / "GHOST.parquet"  # never written
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=fake_path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        # Force the file to disappear.
        fake_path.unlink()
        rebuild_projections(mem_db)
        with pytest.raises(FileNotFoundError):
            load_bars(mem_db, "SPY", datetime(2030, 1, 1, tzinfo=UTC))

    def test_decision_time_with_different_timezone_is_normalised(
        self, mem_db, clock, parquet_dir,
    ) -> None:
        """A decision_time in ET (UTC-5) is correctly compared to UTC bars."""
        bars = _synthetic_spy_bars()
        path = parquet_dir / "SPY.parquet"
        write_bars(
            conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
            vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
            clock=clock,
        )
        rebuild_projections(mem_db)
        # Same instant as 2026-01-14 15:00 UTC, but in ET.
        et = timezone(timedelta(hours=-5))
        decision_time = datetime(2026, 1, 14, 10, 0, 0, tzinfo=et)
        result = load_bars(mem_db, "SPY", decision_time)
        assert result.height == 2


# ---------------------------------------------------------------------------
# current_manifest lookup
# ---------------------------------------------------------------------------

def test_current_manifest_returns_latest_for_symbol(
    mem_db, clock, parquet_dir,
) -> None:
    bars = _synthetic_spy_bars()
    path = parquet_dir / "SPY.parquet"
    write_bars(
        conn=mem_db, symbol="SPY", bars=bars, parquet_path=path,
        vendor="synthetic", date_start="2026-01-12", date_end="2026-01-14",
        clock=clock,
    )
    rebuild_projections(mem_db)
    manifest = current_manifest(mem_db, "SPY")
    assert manifest is not None
    assert manifest["vendor"] == "synthetic"
    assert manifest["symbol"] == "SPY"
    assert manifest["row_count"] == 3


def test_current_manifest_returns_none_for_unknown_symbol(mem_db) -> None:
    assert current_manifest(mem_db, "GHOST") is None


# ---------------------------------------------------------------------------
# record_manifest: direct API
# ---------------------------------------------------------------------------

def test_record_manifest_appends_event(mem_db, clock, tmp_path) -> None:
    event_id = record_manifest(
        conn=mem_db,
        vendor="polygon",
        symbol="SPY",
        fetch_ts=clock.now().isoformat(),
        date_start="2026-01-01",
        date_end="2026-01-31",
        row_count=21,
        content_hash="abc" * 21 + "abcd",  # 64 hex chars
        parquet_path=str(tmp_path / "spy.parquet"),
        clock=clock,
    )
    row = mem_db.execute(
        "SELECT event_type, entity_id FROM events WHERE id = ?",
        (event_id,),
    ).fetchone()
    assert row["event_type"] == "data.manifest_recorded"
    assert row["entity_id"] == "polygon:SPY"


# ---------------------------------------------------------------------------
# PolygonProvider: no network, monkeypatched
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Mimics urllib.response.addinfourl for the monkeypatch."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_polygon_provider_fetches_bars_without_network(
    monkeypatch, mem_db, clock, parquet_dir,
) -> None:
    """The provider's HTTP call is the only network egress; tests
    monkeypatch it and never hit the real Polygon API.
    """
    # Polygon returns millisecond timestamps.
    fake_response = {
        "results": [
            {"t": 1734652800000, "o": 450.0, "h": 455.0,
             "l": 449.0, "c": 454.0, "v": 100_000_000},
            {"t": 1734739200000, "o": 452.0, "h": 456.0,
             "l": 451.0, "c": 455.0, "v": 110_000_000},
        ],
    }

    def fake_urlopen(url: str, *_args: object, **_kwargs: object) -> _FakeResponse:
        # Sanity-check: the URL goes to Polygon and carries the symbol + key.
        assert "api.polygon.io" in url
        assert "SPY" in url
        assert "test-key" in url
        return _FakeResponse(fake_response)

    monkeypatch.setattr(
        "dsky.data.providers.polygon._urlopen", fake_urlopen,
    )

    path = parquet_dir / "SPY.parquet"
    provider = PolygonProvider(api_key="test-key")
    result = provider.fetch_daily_bars(
        symbol="SPY",
        start_date="2024-12-20",
        end_date="2024-12-22",
        conn=mem_db,
        clock=clock,
        parquet_path=path,
    )

    assert result == path
    assert path.exists()
    rebuild_projections(mem_db)
    loaded = load_bars(mem_db, "SPY", datetime(2030, 1, 1, tzinfo=UTC))
    assert loaded.height == 2
    # Both bars have event_time in Dec 2024.
    for et in loaded["event_time"].to_list():
        assert et.year == 2024
        assert et.month == 12


def test_polygon_provider_reads_key_from_env(
    monkeypatch, mem_db, clock, parquet_dir,
) -> None:
    """If no api_key is passed, the provider falls back to POLYGON_API_KEY env."""

    def fake_urlopen(url: str, *_args: object, **_kwargs: object) -> _FakeResponse:
        assert "env-key" in url
        return _FakeResponse({"results": []})

    monkeypatch.setattr("dsky.data.providers.polygon._urlopen", fake_urlopen)
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")

    provider = PolygonProvider()  # no api_key arg
    path = parquet_dir / "QQQ.parquet"
    provider.fetch_daily_bars(
        symbol="QQQ", start_date="2024-12-20", end_date="2024-12-22",
        conn=mem_db, clock=clock, parquet_path=path,
    )
    # No assertion failure in fake_urlopen = key was read from env.


def test_polygon_provider_empty_results_writes_empty_parquet(
    monkeypatch, mem_db, clock, parquet_dir,
) -> None:
    """An empty result set still records the manifest with row_count=0."""
    def fake_urlopen(url: str, *_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse({"results": []})

    monkeypatch.setattr("dsky.data.providers.polygon._urlopen", fake_urlopen)

    path = parquet_dir / "EMPTY.parquet"
    provider = PolygonProvider(api_key="k")
    provider.fetch_daily_bars(
        symbol="EMPTY", start_date="2024-12-20", end_date="2024-12-22",
        conn=mem_db, clock=clock, parquet_path=path,
    )

    rebuild_projections(mem_db)
    manifest = current_manifest(mem_db, "EMPTY")
    assert manifest is not None
    assert manifest["row_count"] == 0
