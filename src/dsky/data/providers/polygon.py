"""Polygon.io REST client for fetching daily aggregate bars.

Fetches ``/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}`` and
converts the response to a polars DataFrame with ``event_time``
millisecond timestamps converted to tz-aware UTC datetimes. Then calls
``data.bars.write_bars`` to persist the data and record provenance.

Network isolation
-----------------
The HTTP call is funnelled through a single module-level alias
(``_urlopen``) so tests can monkeypatch it without touching the real
``urllib.request``. Production code uses ``urllib.request.urlopen``
directly.
"""
import json
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from dsky.clock import Clock
from dsky.data.bars import write_bars

# Single point of HTTP egress. Tests monkeypatch this symbol.
_urlopen = urllib.request.urlopen

VENDOR_NAME = "polygon"
_BASE_URL = "https://api.polygon.io/v2/aggs/ticker"


class PolygonAuthError(RuntimeError):
    """Raised when Polygon returns 401/403 -- the API key is bad or missing."""


class PolygonProvider:
    """Polygon.io daily-bars provider.

    Parameters
    ----------
    api_key:
        Polygon API key. If ``None``, falls back to the
        ``POLYGON_API_KEY`` environment variable.
    timeout:
        HTTP timeout in seconds (passed to ``urlopen``).

    """

    def __init__(self, api_key: str | None = None, *, timeout: float = 30.0) -> None:
        """Store the API key (from arg or env) and the HTTP timeout."""
        if api_key is None:
            api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            msg = (
                "POLYGON_API_KEY is required: pass api_key=... or set the "
                "POLYGON_API_KEY environment variable"
            )
            raise PolygonAuthError(msg)
        self._api_key = api_key
        self._timeout = timeout

    def fetch_daily_bars(  # noqa: PLR0913
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        conn: sqlite3.Connection,
        clock: Clock,
        *,
        parquet_path: str | Path,
    ) -> Path:
        """Fetch daily bars from Polygon and persist to Parquet.

        See :class:`dsky.data.providers.BarProvider` for the full
        contract. This implementation:

        1. Calls ``GET /v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}``
        2. Parses the JSON ``results`` array into a polars DataFrame
           with ``event_time`` (tz-aware UTC) and OHLCV columns
        3. Calls :func:`dsky.data.bars.write_bars` to persist and
           record the manifest event

        Returns the path to the written Parquet file.
        """
        url = (
            f"{_BASE_URL}/{symbol}/range/1/day/{start_date}/{end_date}"
            f"?adjusted=true&sort=asc&apiKey={self._api_key}"
        )

        try:
            with _urlopen(url, timeout=self._timeout) as resp:
                payload: dict[str, Any] = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                msg = f"Polygon auth failed (HTTP {e.code}); check POLYGON_API_KEY"
                raise PolygonAuthError(msg) from e
            raise

        bars = self._results_to_dataframe(payload)

        # write_bars records the manifest event BEFORE writing the file.
        # The provider is not responsible for the manifest -- it is
        # write_bars' job.
        write_bars(
            conn=conn,
            symbol=symbol,
            bars=bars,
            parquet_path=parquet_path,
            vendor=VENDOR_NAME,
            date_start=start_date,
            date_end=end_date,
            clock=clock,
            actor=f"code:{VENDOR_NAME}",
        )
        return Path(parquet_path)

    @staticmethod
    def _results_to_dataframe(payload: dict[str, Any]) -> pl.DataFrame:
        """Convert Polygon's ``results`` array to a polars DataFrame.

        Polygon's response shape::

            {
              "results": [
                {"t": 1734652800000, "o": 450.0, "h": 455.0,
                 "l": 449.0, "c": 454.0, "v": 100000000, "vw": 452.3,
                 "n": 12345},
                ...
              ],
              ...
            }

        ``t`` is the bar's open time in **milliseconds since epoch
        (UTC)**. The conversion to tz-aware UTC datetime is done here
        so the rest of the pipeline (write_bars, load_bars) only deals
        with datetimes.
        """
        results = payload.get("results") or []
        if not results:
            # Return an empty DataFrame with the expected schema.
            return pl.DataFrame(
                schema={
                    "event_time": pl.Datetime(time_zone="UTC"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                },
            )

        rows: dict[str, list[Any]] = {
            "event_time": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        utc = UTC
        for r in results:
            ms = int(r["t"])
            dt = datetime.fromtimestamp(ms / 1000.0, tz=utc)
            rows["event_time"].append(dt)
            rows["open"].append(float(r.get("o", 0.0)))
            rows["high"].append(float(r.get("h", 0.0)))
            rows["low"].append(float(r.get("l", 0.0)))
            rows["close"].append(float(r.get("c", 0.0)))
            rows["volume"].append(int(r.get("v", 0)))
        return pl.DataFrame(rows, schema_overrides={
            "event_time": pl.Datetime(time_zone="UTC"),
        })


__all__ = ["PolygonAuthError", "PolygonProvider"]
