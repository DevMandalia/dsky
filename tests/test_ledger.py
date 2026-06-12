"""Tests for the rejection ledger."""
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from dsky.clock import FrozenClock
from dsky.db.engine import open_db
from dsky.research.ledger import (
    check_duplicate,
    fingerprint,
    record_rejection,
)
from dsky.research.spec import (
    ComputableRule,
    HypothesisSpec,
    SuccessCriteria,
    TimeWindow,
)

_T0 = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def mem_db() -> Iterator[sqlite3.Connection]:
    conn = open_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(at=_T0)


def _base_spec(**kwargs: object) -> HypothesisSpec:
    defaults = {
        "asset": "SPY",
        "signal_definition": ComputableRule(
            kind="Momentum",
            parameters=(("lookback_days", 20),),
        ),
        "entry_rule": ComputableRule(kind="entry", parameters=()),
        "exit_rule": ComputableRule(kind="exit", parameters=()),
        "prediction": "original prediction wording",
        "success_criteria": SuccessCriteria(
            metric="total_return", threshold=0.0, comparator="gt",
        ),
        "train_window": TimeWindow(start="2020-01-01", end="2023-12-31"),
        "holdout_window": TimeWindow(start="2024-01-01", end="2024-12-31"),
        "required_null_models": ("buy_and_hold",),
    }
    defaults.update(kwargs)
    return HypothesisSpec(**defaults)  # type: ignore[arg-type]


class TestFingerprint:
    def test_cosmetic_prediction_change_same_fingerprint(self) -> None:
        a = _base_spec(prediction="SPY momentum predicts upside")
        b = _base_spec(prediction="  spy   MOMENTUM predicts UPSIDE!!!  ")
        assert fingerprint(a) == fingerprint(b)

    def test_signal_kind_whitespace_and_case_normalized(self) -> None:
        a = _base_spec(
            signal_definition=ComputableRule(
                kind="  MoMeNtUm  ",
                parameters=(("lookback_days", 20),),
            ),
        )
        b = _base_spec(
            signal_definition=ComputableRule(
                kind="momentum",
                parameters=(("lookback_days", 20),),
            ),
        )
        assert fingerprint(a) == fingerprint(b)

    def test_different_signal_kind_different_fingerprint(self) -> None:
        a = _base_spec()
        b = _base_spec(
            signal_definition=ComputableRule(
                kind="mean_reversion",
                parameters=(("lookback_days", 20),),
            ),
        )
        assert fingerprint(a) != fingerprint(b)


class TestRecordRejection:
    def test_record_rejection_updates_ledger_projection(
        self, mem_db, clock,
    ) -> None:
        spec = _base_spec()
        record_rejection(
            mem_db, entity_id="idea-1", spec=spec,
            reason="null not significant", actor="code:sieve", clock=clock,
        )
        row = mem_db.execute(
            "SELECT fingerprint, asset, reason FROM rejection_ledger",
        ).fetchone()
        assert row is not None
        assert row["fingerprint"] == fingerprint(spec)
        assert row["asset"] == "SPY"
        assert row["reason"] == "null not significant"


class TestCheckDuplicate:
    def test_near_identical_rejected_idea_is_flagged(
        self, mem_db, clock,
    ) -> None:
        original = _base_spec()
        record_rejection(
            mem_db, entity_id="idea-1", spec=original,
            reason="no edge", actor="code:sieve", clock=clock,
        )

        cosmetic = _base_spec(
            prediction="completely different prose, same signal",
            entry_rule=ComputableRule(kind="different_entry", parameters=()),
        )
        warning = check_duplicate(mem_db, cosmetic)
        assert warning is not None
        assert warning.fingerprint == fingerprint(original)
        assert warning.reason == "no edge"

    def test_no_warning_for_fresh_signal(self, mem_db, clock) -> None:
        record_rejection(
            mem_db, entity_id="idea-1", spec=_base_spec(),
            reason="no edge", actor="code:sieve", clock=clock,
        )
        different = _base_spec(
            signal_definition=ComputableRule(
                kind="vol_breakout", parameters=(),
            ),
        )
        assert check_duplicate(mem_db, different) is None
