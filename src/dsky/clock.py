"""Time source used in place of datetime.now() throughout the codebase."""
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """Anything that can answer 'what time is it now?' in tz-aware UTC."""

    def now(self) -> datetime:
        """Return the current instant as a timezone-aware UTC datetime."""


class SystemClock:
    """Real-time clock. The default choice for production runs."""

    def now(self) -> datetime:
        """Return the real current time as a timezone-aware UTC datetime."""
        return datetime.now(tz=UTC)


class FrozenClock:
    """Clock that always returns the same instant. Use in tests."""

    def __init__(self, at: datetime) -> None:
        """Store the fixed instant that subsequent now() calls will return."""
        self._at = at

    def now(self) -> datetime:
        """Return the frozen instant."""
        return self._at
