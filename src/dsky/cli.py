"""Command-line entry point dispatching to lifecycle and research operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    import sqlite3

import typer

from dsky.clock import SystemClock
from dsky.config import db_path, default_actor
from dsky.db.engine import open_db
from dsky.lifecycle.registry import load_pre_registered_spec
from dsky.ops import (
    BOUNCE_HYPOTHESIS,
    OpResult,
    default_parquet_path,
    op_backtest,
    op_capture,
    op_ledger_check,
    op_preregister,
    op_promote,
    op_review,
    op_robustness,
    op_specify,
    op_status,
    op_verify_chain,
)

app = typer.Typer(
    help="Direction-Sky: pre-registered, append-only quant research lifecycle.",
    no_args_is_help=True,
)


def _resolve_db(db: Path | None) -> Path:
    return db if db is not None else db_path()


def _run_with_db(db: Path | None, fn: object) -> None:
    conn = open_db(_resolve_db(db))
    try:
        if callable(fn):
            fn(conn)
    finally:
        conn.close()


def _emit(result: OpResult) -> None:
    """Print human summary and event id (Step 11 contract)."""
    typer.echo(result.summary)
    if result.event_id > 0:
        typer.echo(f"event_id: {result.event_id}")


@app.callback()
def main(
    ctx: typer.Context,
    db: Annotated[
        Path | None,
        typer.Option(help="SQLite event-log path (default: DSKY_DB or dsky.db)"),
    ] = None,
) -> None:
    """Global options applied before subcommands."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


def _db_from_ctx(ctx: typer.Context) -> Path | None:
    obj = ctx.obj
    if not isinstance(obj, dict):
        return None
    raw = obj.get("db")
    return raw if isinstance(raw, Path) else None


@app.command()
def capture(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument(help="Opaque idea identifier")],
    hypothesis: Annotated[
        str,
        typer.Option(help="Natural-language hypothesis text"),
    ] = BOUNCE_HYPOTHESIS,
    asset: Annotated[str, typer.Option(help="Primary asset symbol")] = "SPY",
    actor: Annotated[str | None, typer.Option(help="Actor string")] = None,
) -> None:
    """Register a new idea (``idea.registered`` → CAPTURED)."""
    clock = SystemClock()
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        _emit(op_capture(
            conn,
            entity_id=entity_id,
            hypothesis=hypothesis,
            asset=asset,
            actor=actor or default_actor(),
            clock=clock,
        ))

    _run_with_db(db, _do)


@app.command()
def specify(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument()],
    hypothesis: Annotated[str, typer.Option()] = BOUNCE_HYPOTHESIS,
    asset: Annotated[str, typer.Option()] = "SPY",
    actor: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Freeze structured spec draft and transition to SPECIFIED."""
    clock = SystemClock()
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        _emit(op_specify(
            conn,
            entity_id=entity_id,
            hypothesis=hypothesis,
            asset=asset,
            actor=actor or default_actor(),
            clock=clock,
        ))

    _run_with_db(db, _do)


@app.command()
def preregister(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument()],
    actor: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Hash and freeze spec (``spec.pre_registered``) → PRE_REGISTERED."""
    clock = SystemClock()
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        _emit(op_preregister(
            conn,
            entity_id=entity_id,
            actor=actor or default_actor(),
            clock=clock,
        ))

    _run_with_db(db, _do)


@app.command()
def backtest(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument()],
    actor: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Train-window backtest → BACKTESTED (requires pre-registration)."""
    clock = SystemClock()
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        hypothesis = load_pre_registered_spec(conn, entity_id)
        path = default_parquet_path(hypothesis.asset)
        _emit(op_backtest(
            conn,
            entity_id=entity_id,
            actor=actor or default_actor(),
            clock=clock,
            parquet_path=path,
        ))

    _run_with_db(db, _do)


@app.command()
def robustness(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument()],
    actor: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Holdout robustness battery → ROBUSTNESS_CHECKED."""
    clock = SystemClock()
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        hypothesis = load_pre_registered_spec(conn, entity_id)
        path = default_parquet_path(hypothesis.asset)
        _emit(op_robustness(
            conn,
            entity_id=entity_id,
            actor=actor or default_actor(),
            clock=clock,
            parquet_path=path,
        ))

    _run_with_db(db, _do)


@app.command()
def review(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument()],
    verdict: Annotated[str, typer.Option(help="approve or reject")],
    rationale: Annotated[str, typer.Option(help="Human rationale")],
    actor: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Record ``review.decided``; reject path transitions to REJECTED + ledger."""
    clock = SystemClock()
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        _emit(op_review(
            conn,
            entity_id=entity_id,
            verdict=verdict,
            rationale=rationale,
            actor=actor or default_actor(),
            clock=clock,
        ))

    _run_with_db(db, _do)


@app.command()
def promote(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument()],
    approval_type: Annotated[
        str,
        typer.Option(help="context, watchlist, or paper_tradeable"),
    ],
    actor: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Record ``idea.promoted`` and transition to APPROVED."""
    clock = SystemClock()
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        _emit(op_promote(
            conn,
            entity_id=entity_id,
            approval_type=approval_type,
            actor=actor or default_actor(),
            clock=clock,
        ))

    _run_with_db(db, _do)


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument(help="Entity to inspect")],
) -> None:
    """Print full ordered event history for an entity."""
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        typer.echo(op_status(conn, entity_id))

    _run_with_db(db, _do)


@app.command("verify-chain")
def verify_chain_cmd(ctx: typer.Context) -> None:
    """Verify the append-only hash chain over all events."""
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        result = op_verify_chain(conn)
        _emit(result)

    _run_with_db(db, _do)


@app.command("ledger-check")
def ledger_check_cmd(
    ctx: typer.Context,
    entity_id: Annotated[str, typer.Argument()],
) -> None:
    """Check rejection ledger for a duplicate fingerprint."""
    db = _db_from_ctx(ctx)

    def _do(conn: sqlite3.Connection) -> None:
        _emit(op_ledger_check(conn, entity_id))

    _run_with_db(db, _do)


if __name__ == "__main__":
    app()
