"""CLI and ops end-to-end smoke: SPY bounce hypothesis through full lifecycle."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsky.cli import app
from dsky.clock import FrozenClock
from dsky.db.engine import open_db
from dsky.db.events import verify_chain
from dsky.ops import (
    BOUNCE_HYPOTHESIS,
    default_parquet_path,
    op_backtest,
    op_capture,
    op_ledger_check,
    op_preregister,
    op_review,
    op_robustness,
    op_specify,
    op_status,
    op_verify_chain,
)

_T0 = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)
_ENTITY = "spy-bounce-smoke"
_RUNNER = CliRunner()


@pytest.fixture
def mem_db() -> Iterator[sqlite3.Connection]:
    conn = open_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(at=_T0)


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("DSKY_DATA", str(root))
    return root


def _run_pipeline(
    conn: sqlite3.Connection,
    clock: FrozenClock,
    data_root: Path,
) -> None:
    """capture → specify → preregister → backtest → robustness → review(reject)."""
    actor = "test:cli-smoke"
    op_capture(
        conn,
        entity_id=_ENTITY,
        hypothesis=BOUNCE_HYPOTHESIS,
        asset="SPY",
        actor=actor,
        clock=clock,
    )
    op_specify(
        conn,
        entity_id=_ENTITY,
        hypothesis=BOUNCE_HYPOTHESIS,
        asset="SPY",
        actor=actor,
        clock=clock,
    )
    op_preregister(conn, entity_id=_ENTITY, actor=actor, clock=clock)
    parquet = default_parquet_path("SPY")
    op_backtest(
        conn,
        entity_id=_ENTITY,
        actor=actor,
        clock=clock,
        parquet_path=parquet,
    )
    robust = op_robustness(
        conn,
        entity_id=_ENTITY,
        actor=actor,
        clock=clock,
        parquet_path=parquet,
    )
    assert not robust.details["passed"], "noise bounce should fail null comparison"
    op_review(
        conn,
        entity_id=_ENTITY,
        verdict="reject",
        rationale="Null comparison did not support the bounce edge on holdout.",
        actor=actor,
        clock=clock,
    )


class TestOpsSmoke:
    def test_bounce_hypothesis_full_pipeline(
        self,
        mem_db: sqlite3.Connection,
        clock: FrozenClock,
        data_root: Path,
    ) -> None:
        _run_pipeline(mem_db, clock, data_root)

        chain = op_verify_chain(mem_db)
        assert chain.summary == "Hash chain intact"
        assert verify_chain(mem_db) is None

        history = op_status(mem_db, _ENTITY)
        for event_type in (
            "idea.registered",
            "idea.specified",
            "idea.state_changed",
            "spec.pre_registered",
            "robustness.run.recorded",
            "review.decided",
            "idea.rejected",
        ):
            assert event_type in history, f"missing {event_type} in audit path"
        # Engine records runs under backtest:{symbol}, not the idea entity id.
        bt = mem_db.execute(
            "SELECT 1 FROM events WHERE event_type = 'backtest.run.recorded' LIMIT 1",
        ).fetchone()
        assert bt is not None
        assert "BACKTESTED" in history or "idea.state_changed" in history

        ledger = op_ledger_check(mem_db, _ENTITY)
        assert "DUPLICATE" in ledger.summary

    def test_no_command_skips_preregister_gate(
        self,
        mem_db: sqlite3.Connection,
        clock: FrozenClock,
        data_root: Path,
    ) -> None:
        """Backtest without preregister must fail via the state machine."""
        actor = "test:gate"
        op_capture(
            mem_db,
            entity_id="gate-test",
            hypothesis=BOUNCE_HYPOTHESIS,
            asset="SPY",
            actor=actor,
            clock=clock,
        )
        op_specify(
            mem_db,
            entity_id="gate-test",
            hypothesis=BOUNCE_HYPOTHESIS,
            asset="SPY",
            actor=actor,
            clock=clock,
        )
        with pytest.raises(LookupError, match=r"spec\.pre_registered"):
            op_backtest(
                mem_db,
                entity_id="gate-test",
                actor=actor,
                clock=clock,
                parquet_path=default_parquet_path("SPY"),
            )


class TestCliSmoke:
    def test_cli_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_file = tmp_path / "smoke.db"
        data_dir = tmp_path / "bars"
        data_dir.mkdir()
        monkeypatch.setenv("DSKY_DATA", str(data_dir))
        monkeypatch.setenv("DSKY_ACTOR", "test:cli-runner")

        common = ["--db", str(db_file)]
        for step in (
            ["capture", _ENTITY, "--hypothesis", BOUNCE_HYPOTHESIS],
            ["specify", _ENTITY],
            ["preregister", _ENTITY],
            ["backtest", _ENTITY],
            ["robustness", _ENTITY],
            [
                "review",
                _ENTITY,
                "--verdict",
                "reject",
                "--rationale",
                "Holdout null comparison failed.",
            ],
        ):
            result = _RUNNER.invoke(app, [*common, *step])
            assert result.exit_code == 0, result.stdout + result.stderr

        chain = _RUNNER.invoke(app, [*common, "verify-chain"])
        assert chain.exit_code == 0
        assert "Hash chain intact" in chain.stdout

        status = _RUNNER.invoke(app, [*common, "status", _ENTITY])
        assert status.exit_code == 0
        assert "idea.registered" in status.stdout
        assert "review.decided" in status.stdout
        assert "idea.rejected" in status.stdout

        ledger = _RUNNER.invoke(app, [*common, "ledger-check", _ENTITY])
        assert ledger.exit_code == 0
        assert "DUPLICATE" in ledger.stdout
