"""Frozen, hash-addressed hypothesis spec.

A :class:`HypothesisSpec` is the pre-registered claim: asset, signal,
entry/exit rules, prediction, success criteria, train/holdout windows,
and required null models. The spec must be complete before
pre-registration; after pre-registration it is immutable (frozen
dataclass + content hash over the canonical JSON form).
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Any


class IncompleteSpec(ValueError):  # noqa: N818 -- named per test spec
    """Raised when a :class:`HypothesisSpec` fails completeness validation."""

    def __init__(self, field: str, reason: str) -> None:
        """Record which field failed and why."""
        self.field = field
        self.reason = reason
        super().__init__(f"incomplete spec: {field!r} -- {reason}")


# JSON-serializable scalar for rule parameters.
RuleScalar = str | int | float | bool


@dataclass(frozen=True)
class ComputableRule:
    """A structured, computable trading rule.

    ``kind`` names the rule family (e.g. ``"momentum"``,
    ``"threshold_cross"``). ``parameters`` is a sorted tuple of
    ``(name, value)`` pairs so the canonical form is stable.
    """

    kind: str
    parameters: tuple[tuple[str, RuleScalar], ...] = ()


@dataclass(frozen=True)
class SuccessCriteria:
    """How a backtest outcome is judged against the pre-registered claim."""

    metric: str
    threshold: float
    comparator: str  # "gt" | "gte" | "lt" | "lte"


@dataclass(frozen=True)
class TimeWindow:
    """An inclusive ISO-8601 date window (``YYYY-MM-DD``)."""

    start: str
    end: str


@dataclass(frozen=True)
class HypothesisSpec:
    """A frozen, pre-registerable hypothesis.

    Every field is required for completeness. The spec is immutable
    after construction (``frozen=True``); pre-registration records a
    content hash of the canonical JSON so tampering is detectable.
    """

    asset: str
    signal_definition: ComputableRule
    entry_rule: ComputableRule
    exit_rule: ComputableRule
    prediction: str
    success_criteria: SuccessCriteria
    train_window: TimeWindow
    holdout_window: TimeWindow
    required_null_models: tuple[str, ...]

    def validate_completeness(self) -> None:
        """Raise :class:`IncompleteSpec` if any required field is underspecified."""
        if not self.asset.strip():
            field, reason = "asset", "must be a non-empty string"
            raise IncompleteSpec(field, reason)

        for name, rule in (
            ("signal_definition", self.signal_definition),
            ("entry_rule", self.entry_rule),
            ("exit_rule", self.exit_rule),
        ):
            if not rule.kind.strip():
                raise IncompleteSpec(name, "kind must be a non-empty string")
            if not _parameters_sorted(rule.parameters):
                raise IncompleteSpec(
                    name,
                    "parameters must be sorted by name for canonical hashing",
                )

        if not self.prediction.strip():
            field, reason = "prediction", "must be a non-empty string"
            raise IncompleteSpec(field, reason)

        _validate_success_criteria(self.success_criteria)
        _validate_time_window("train_window", self.train_window)
        _validate_time_window("holdout_window", self.holdout_window)

        if self.train_window.end >= self.holdout_window.start:
            field, reason = (
                "holdout_window",
                "must start strictly after train_window ends",
            )
            raise IncompleteSpec(field, reason)

        if not self.required_null_models:
            field, reason = (
                "required_null_models",
                "must name at least one null model",
            )
            raise IncompleteSpec(field, reason)
        if any(not m.strip() for m in self.required_null_models):
            field, reason = (
                "required_null_models",
                "null model names must be non-empty strings",
            )
            raise IncompleteSpec(field, reason)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict in canonical form."""
        return {
            "asset": self.asset,
            "signal_definition": _rule_to_dict(self.signal_definition),
            "entry_rule": _rule_to_dict(self.entry_rule),
            "exit_rule": _rule_to_dict(self.exit_rule),
            "prediction": self.prediction,
            "success_criteria": {
                "metric": self.success_criteria.metric,
                "threshold": self.success_criteria.threshold,
                "comparator": self.success_criteria.comparator,
            },
            "train_window": {
                "start": self.train_window.start,
                "end": self.train_window.end,
            },
            "holdout_window": {
                "start": self.holdout_window.start,
                "end": self.holdout_window.end,
            },
            "required_null_models": list(self.required_null_models),
        }

    def canonical_json(self) -> str:
        """Deterministic JSON bytes-as-string for hashing."""
        return json.dumps(
            self.to_canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def content_hash(self) -> str:
        """SHA-256 hex digest of :meth:`canonical_json`."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_event_payload(cls, payload: dict[str, Any]) -> "HypothesisSpec":
        """Reconstruct a spec from a ``spec.pre_registered`` event payload."""
        spec_dict = payload["spec"]
        return cls(
            asset=str(spec_dict["asset"]),
            signal_definition=_rule_from_dict(spec_dict["signal_definition"]),
            entry_rule=_rule_from_dict(spec_dict["entry_rule"]),
            exit_rule=_rule_from_dict(spec_dict["exit_rule"]),
            prediction=str(spec_dict["prediction"]),
            success_criteria=SuccessCriteria(
                metric=str(spec_dict["success_criteria"]["metric"]),
                threshold=float(spec_dict["success_criteria"]["threshold"]),
                comparator=str(spec_dict["success_criteria"]["comparator"]),
            ),
            train_window=TimeWindow(
                start=str(spec_dict["train_window"]["start"]),
                end=str(spec_dict["train_window"]["end"]),
            ),
            holdout_window=TimeWindow(
                start=str(spec_dict["holdout_window"]["start"]),
                end=str(spec_dict["holdout_window"]["end"]),
            ),
            required_null_models=tuple(spec_dict["required_null_models"]),
        )


def _parameters_sorted(params: tuple[tuple[str, RuleScalar], ...]) -> bool:
    names = [p[0] for p in params]
    return names == sorted(names)


def _validate_success_criteria(sc: SuccessCriteria) -> None:
    if not sc.metric.strip():
        field, reason = "success_criteria.metric", "must be non-empty"
        raise IncompleteSpec(field, reason)
    if sc.comparator not in {"gt", "gte", "lt", "lte"}:
        field = "success_criteria.comparator"
        reason = f"must be one of gt/gte/lt/lte, got {sc.comparator!r}"
        raise IncompleteSpec(field, reason)


def _validate_time_window(name: str, window: TimeWindow) -> None:
    if not window.start.strip() or not window.end.strip():
        raise IncompleteSpec(name, "start and end must be non-empty ISO dates")
    if window.start > window.end:
        raise IncompleteSpec(name, "start must be on or before end")


def _rule_to_dict(rule: ComputableRule) -> dict[str, Any]:
    return {
        "kind": rule.kind,
        "parameters": [[k, v] for k, v in rule.parameters],
    }


def _rule_from_dict(data: dict[str, Any]) -> ComputableRule:
    raw_params = data.get("parameters") or []
    params = tuple((str(p[0]), p[1]) for p in raw_params)
    return ComputableRule(kind=str(data["kind"]), parameters=params)


__all__ = [
    "ComputableRule",
    "HypothesisSpec",
    "IncompleteSpec",
    "RuleScalar",
    "SuccessCriteria",
    "TimeWindow",
]
