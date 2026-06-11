"""Pluggable market data provider clients.

The :class:`BarProvider` protocol is the abstraction: any vendor that
implements ``fetch_daily_bars(symbol, start_date, end_date, conn,
clock, *, parquet_path)`` is a drop-in replacement. Future vendors
(Alpha Vantage, IEX, Databento) live alongside :mod:`polygon` and
implement the same interface.
"""
import sqlite3
from pathlib import Path
from typing import Protocol

from dsky.clock import Clock


class BarProvider(Protocol):
    """The interface every market-data vendor must satisfy.

    A provider is responsible for:
      1. Fetching daily bars from its source (HTTP, file, etc.).
      2. Building a polars DataFrame with ``event_time`` (and
         optionally ``available_time``) columns.
      3. Calling ``data.bars.write_bars`` to persist the data, which
         also records the manifest event in the log.
    """

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
        """Fetch daily bars for ``symbol`` over ``[start_date, end_date]``.

        Parameters
        ----------
        symbol:
            Ticker symbol, e.g. ``"SPY"``.
        start_date, end_date:
            ISO-8601 dates (``YYYY-MM-DD``) inclusive.
        conn:
            Open sqlite3 connection; the provider appends its
            manifest event to the events log via ``write_bars``.
        clock:
            Source of the current timestamp for the manifest event.
        parquet_path:
            Where the provider should write the Parquet file. Parent
            directories are created as needed.

        Returns
        -------
        Path
            The path the Parquet file was written to (typically the
            same as ``parquet_path``).

        """
        ...


__all__ = ["BarProvider"]
