"""Rejection ledger: fingerprint specs and record duplicate ideas."""

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass

from dsky.clock import Clock
from dsky.db.events import append_event
from dsky.db.projections import rebuild_projections
from dsky.research.spec import ComputableRule, HypothesisSpec, RuleScalar

_WHITESPACE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    """Collapse whitespace and lower-case for cosmetic-stable comparison."""
    return _WHITESPACE.sub(" ", value.strip().lower())


def _normalize_scalar(value: RuleScalar) -> RuleScalar:
    if isinstance(value, str):
        return _normalize_text(value)
    return value


def normalize_signal(rule: ComputableRule) -> ComputableRule:
    """Normalize a signal rule so trivial rewordings hash identically."""
    return ComputableRule(
        kind=_normalize_text(rule.kind),
        parameters=tuple(
            (name, _normalize_scalar(val))
            for name, val in rule.parameters
        ),
    )


def _fingerprint_payload(spec: HypothesisSpec) -> dict[str, object]:
    """Canonical dict for fingerprinting: asset + normalized signal only."""
    signal = normalize_signal(spec.signal_definition)
    return {
        "asset": spec.asset.strip().upper(),
        "signal_definition": {
            "kind": signal.kind,
            "parameters": [[k, v] for k, v in signal.parameters],
        },
    }


def fingerprint(spec: HypothesisSpec) -> str:
    """Stable SHA-256 over asset + normalized signal definition.

    Entry/exit rules, prediction wording, and window dates are excluded
    so cosmetic rewordings cannot dodge the rejection ledger.
    """
    canonical = json.dumps(
        _fingerprint_payload(spec),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class DuplicateWarning:
    """A prior rejection with the same (or near-identical) fingerprint."""

    fingerprint: str
    asset: str
    reason: str
    rejected_ts: str


def check_duplicate(
    conn: sqlite3.Connection,
    spec: HypothesisSpec,
) -> DuplicateWarning | None:
    """Warn if ``spec`` matches a fingerprint already in the rejection ledger."""
    fp = fingerprint(spec)
    row = conn.execute(
        """
        SELECT fingerprint, asset, reason, rejected_ts
        FROM rejection_ledger
        WHERE fingerprint = ?
        ORDER BY rejected_ts DESC
        LIMIT 1
        """,
        (fp,),
    ).fetchone()
    if row is None:
        return None
    return DuplicateWarning(
        fingerprint=str(row["fingerprint"]),
        asset=str(row["asset"]),
        reason=str(row["reason"]),
        rejected_ts=str(row["rejected_ts"]),
    )


def record_rejection(  # noqa: PLR0913
    conn: sqlite3.Connection,
    entity_id: str,
    spec: HypothesisSpec,
    reason: str,
    actor: str,
    clock: Clock,
) -> int:
    """Append ``idea.rejected`` and refresh the rejection-ledger projection."""
    fp = fingerprint(spec)
    event_id = append_event(
        conn=conn,
        event_type="idea.rejected",
        entity_id=entity_id,
        actor=actor,
        payload={
            "fingerprint": fp,
            "asset": spec.asset,
            "reason": reason,
        },
        clock=clock,
    )
    rebuild_projections(conn)
    return event_id


__all__ = [
    "DuplicateWarning",
    "check_duplicate",
    "fingerprint",
    "normalize_signal",
    "record_rejection",
]
