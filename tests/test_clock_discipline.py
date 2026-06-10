"""Clock discipline: clock.py is the only place datetime.now() may be called."""
from datetime import UTC, datetime
from pathlib import Path

from dsky.clock import FrozenClock, SystemClock

# --- Discipline scan -----------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "dsky"
ALLOWED_FILE = SRC_ROOT / "clock.py"
FORBIDDEN_NEEDLES = ("datetime.now(", "datetime.utcnow(")


def test_no_datetime_now_outside_clock() -> None:
    """No .py file under src/dsky other than clock.py may call datetime.now(.).

    The clock.py module is the only sanctioned source of "now" in this codebase.
    """
    def report(path: Path, needle: str) -> str:
        rel = path.relative_to(SRC_ROOT.parent.parent)
        return f"  {rel}: contains {needle!r}"

    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.resolve() == ALLOWED_FILE.resolve():
            continue
        text = path.read_text(encoding="utf-8")
        offenders.extend(report(path, n) for n in FORBIDDEN_NEEDLES if n in text)
    assert not offenders, (
        "datetime.now() / datetime.utcnow() called outside clock.py:\n"
        + "\n".join(offenders)
    )


# --- Behaviour -----------------------------------------------------------------
def test_frozen_clock_returns_exact_instant() -> None:
    """FrozenClock returns the exact instant it was constructed with."""
    instant = datetime(2024, 1, 15, 12, 30, 45, tzinfo=UTC)
    clock = FrozenClock(at=instant)
    assert clock.now() == instant
    assert clock.now() == clock.now()


def test_system_clock_now_is_tz_aware_utc() -> None:
    """SystemClock.now() returns a timezone-aware datetime in UTC."""
    now = SystemClock().now()
    assert isinstance(now, datetime)
    assert now.tzinfo is not None
    assert now.tzinfo == UTC
