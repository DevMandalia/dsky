"""Tests for HypothesisSpec, pre-registration, and lifecycle gates."""
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from dsky.clock import FrozenClock
from dsky.db.engine import open_db
from dsky.db.events import append_event
from dsky.lifecycle.machine import (
    PreRegistrationRequired,
    assert_backtest_allowed,
    read_success_criteria,
    transition,
)
from dsky.lifecycle.registry import (
    AlreadyPreRegistered,
    pre_register,
    verify_stored_spec_hash,
)
from dsky.lifecycle.states import State
from dsky.research.spec import (
    ComputableRule,
    HypothesisSpec,
    IncompleteSpec,
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


def _complete_spec(**overrides: object) -> HypothesisSpec:
    """A valid, complete hypothesis spec for tests."""
    base = HypothesisSpec(
        asset="SPY",
        signal_definition=ComputableRule(
            kind="momentum",
            parameters=(("lookback_days", 20),),
        ),
        entry_rule=ComputableRule(
            kind="threshold_cross",
            parameters=(("direction", "above"), ("level", 0.0)),
        ),
        exit_rule=ComputableRule(
            kind="holding_period",
            parameters=(("max_days", 5),),
        ),
        prediction="SPY 20-day momentum predicts positive next-week return",
        success_criteria=SuccessCriteria(
            metric="sharpe_ratio",
            threshold=0.5,
            comparator="gt",
        ),
        train_window=TimeWindow(start="2020-01-01", end="2023-12-31"),
        holdout_window=TimeWindow(start="2024-01-01", end="2024-12-31"),
        required_null_models=("buy_and_hold", "random_entry_null"),
    )
    if not overrides:
        return base
    fields = {
        "asset": base.asset,
        "signal_definition": base.signal_definition,
        "entry_rule": base.entry_rule,
        "exit_rule": base.exit_rule,
        "prediction": base.prediction,
        "success_criteria": base.success_criteria,
        "train_window": base.train_window,
        "holdout_window": base.holdout_window,
        "required_null_models": base.required_null_models,
    }
    fields.update(overrides)
    return HypothesisSpec(**fields)  # type: ignore[arg-type]


def _register_and_specify(conn: sqlite3.Connection, entity_id: str, clock: FrozenClock) -> None:
    append_event(
        conn=conn,
        event_type="idea.registered",
        entity_id=entity_id,
        actor="seeder",
        payload={"name": entity_id, "asset": "SPY"},
        clock=clock,
    )
    transition(conn, entity_id, State.SPECIFIED, "alice", clock)


class TestHypothesisSpecImmutability:
    def test_frozen_spec_refuses_field_mutation(self) -> None:
        spec = _complete_spec()
        with pytest.raises(FrozenInstanceError):
            spec.asset = "QQQ"  # type: ignore[misc]

    def test_frozen_nested_rule_refuses_mutation(self) -> None:
        spec = _complete_spec()
        with pytest.raises(FrozenInstanceError):
            spec.signal_definition.kind = "other"  # type: ignore[misc]


class TestHypothesisSpecCompleteness:
    def test_rejects_empty_asset(self) -> None:
        with pytest.raises(IncompleteSpec, match="asset"):
            _complete_spec(asset="  ").validate_completeness()

    def test_rejects_unsorted_rule_parameters(self) -> None:
        bad_rule = ComputableRule(
            kind="momentum",
            parameters=(("z", 1), ("a", 2)),
        )
        with pytest.raises(IncompleteSpec, match="parameters must be sorted"):
            _complete_spec(signal_definition=bad_rule).validate_completeness()

    def test_rejects_overlapping_train_and_holdout(self) -> None:
        with pytest.raises(IncompleteSpec, match="holdout_window"):
            _complete_spec(
                holdout_window=TimeWindow(start="2023-06-01", end="2024-12-31"),
            ).validate_completeness()

    def test_rejects_empty_null_models(self) -> None:
        with pytest.raises(IncompleteSpec, match="required_null_models"):
            _complete_spec(required_null_models=()).validate_completeness()


class TestHypothesisSpecHashing:
    def test_content_hash_matches_recomputation_from_frozen_spec(self) -> None:
        spec = _complete_spec()
        spec.validate_completeness()
        h1 = spec.content_hash()
        h2 = HypothesisSpec(
            asset=spec.asset,
            signal_definition=spec.signal_definition,
            entry_rule=spec.entry_rule,
            exit_rule=spec.exit_rule,
            prediction=spec.prediction,
            success_criteria=spec.success_criteria,
            train_window=spec.train_window,
            holdout_window=spec.holdout_window,
            required_null_models=spec.required_null_models,
        ).content_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_stored_hash_matches_recomputation_after_pre_register(
        self, mem_db, clock,
    ) -> None:
        spec = _complete_spec()
        pre_register(mem_db, "idea-1", spec, "alice", clock)
        assert verify_stored_spec_hash(mem_db, "idea-1") is True

        row = mem_db.execute(
            "SELECT payload FROM events WHERE event_type = 'spec.pre_registered'",
        ).fetchone()
        payload = json.loads(row["payload"])
        assert payload["spec_hash"] == spec.content_hash()
        assert payload["canonical"] == spec.canonical_json()


class TestPreRegister:
    def test_pre_register_appends_spec_pre_registered_event(
        self, mem_db, clock,
    ) -> None:
        spec = _complete_spec()
        event_id = pre_register(mem_db, "idea-1", spec, "alice", clock)

        row = mem_db.execute(
            "SELECT event_type, entity_id, actor FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        assert row["event_type"] == "spec.pre_registered"
        assert row["entity_id"] == "idea-1"
        assert row["actor"] == "alice"

    def test_pre_register_rejects_incomplete_spec(self, mem_db, clock) -> None:
        spec = _complete_spec(prediction="")
        with pytest.raises(IncompleteSpec):
            pre_register(mem_db, "idea-1", spec, "alice", clock)

    def test_pre_register_twice_raises(self, mem_db, clock) -> None:
        spec = _complete_spec()
        pre_register(mem_db, "idea-1", spec, "alice", clock)
        with pytest.raises(AlreadyPreRegistered):
            pre_register(mem_db, "idea-1", spec, "bob", clock)


class TestLifecyclePreRegistrationGates:
    def test_backtest_without_pre_registered_event_raises(
        self, mem_db, clock,
    ) -> None:
        _register_and_specify(mem_db, "idea-1", clock)
        with pytest.raises(PreRegistrationRequired, match="backtest"):
            assert_backtest_allowed(mem_db, "idea-1")

    def test_read_success_criteria_without_pre_registered_raises(
        self, mem_db, clock,
    ) -> None:
        _register_and_specify(mem_db, "idea-1", clock)
        with pytest.raises(PreRegistrationRequired, match="read_success_criteria"):
            read_success_criteria(mem_db, "idea-1")

    def test_transition_to_backtested_without_pre_registered_raises(
        self, mem_db, clock,
    ) -> None:
        """PRE_REGISTERED state alone is insufficient -- the event must exist."""
        _register_and_specify(mem_db, "idea-1", clock)
        transition(mem_db, "idea-1", State.PRE_REGISTERED, "alice", clock)
        with pytest.raises(PreRegistrationRequired):
            transition(mem_db, "idea-1", State.BACKTESTED, "code:backtest", clock)

    def test_backtest_allowed_after_pre_register(
        self, mem_db, clock,
    ) -> None:
        _register_and_specify(mem_db, "idea-1", clock)
        spec = _complete_spec()
        pre_register(mem_db, "idea-1", spec, "alice", clock)
        transition(mem_db, "idea-1", State.PRE_REGISTERED, "alice", clock)

        assert_backtest_allowed(mem_db, "idea-1")  # does not raise
        criteria = read_success_criteria(mem_db, "idea-1")
        assert criteria.metric == "sharpe_ratio"
        assert criteria.threshold == 0.5

        transition(mem_db, "idea-1", State.BACKTESTED, "code:backtest", clock)
