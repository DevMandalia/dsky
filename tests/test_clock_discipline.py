"""Clock discipline: every datetime.now() call in this repo must live in clock.py.

We use the AST to find actual Call nodes targeting datetime.now / .utcnow, so
strings in docstrings, comments, and tuple literals are not flagged. This file
imports datetime for typing and FrozenClock/SystemClock for behavioural tests,
but it never invokes datetime.now() — only clock.now().
"""
import ast
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from dsky.clock import FrozenClock, SystemClock

# --- Discipline scan -----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "dsky"

# clock.py IS the sanctioned source of datetime.now(). It is the only file
# in the repo allowed to invoke it.
ALLOWED_FILES: frozenset[Path] = frozenset({SRC_ROOT / "clock.py"})

# Directories we never want to scan: third-party, caches, generated artefacts.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".uv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        ".eggs",
    }
)


def _iter_python_files() -> Iterator[Path]:
    for path in sorted(REPO_ROOT.rglob("*.py")):
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        yield path


def _collect_datetime_aliases(tree: ast.AST) -> set[str]:
    """Return every name in `tree` that refers to the datetime class."""
    aliases: set[str] = {"datetime"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            for alias in node.names:
                if alias.name == "datetime":
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "datetime":
                    aliases.add(alias.asname or "datetime")
    return aliases


def _is_datetime_receiver(node: ast.AST, datetime_aliases: set[str]) -> bool:
    """True if `node` is the datetime class — direct name, import alias, or
    fully-qualified `datetime.datetime` reference."""
    if isinstance(node, ast.Name):
        return node.id in datetime_aliases
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "datetime"
        and isinstance(node.value, ast.Name)
        and node.value.id == "datetime"
    )


def _now_call_nodes(tree: ast.AST, datetime_aliases: set[str]) -> Iterator[ast.Call]:
    """Yield every Call node whose target is datetime.now(...) or .utcnow(...)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"now", "utcnow"}:
            continue
        if _is_datetime_receiver(func.value, datetime_aliases):
            yield node


def _find_datetime_now_calls(text: str) -> list[tuple[int, str]]:
    """Return (line_no, source) for every .now() / .utcnow() call on the
    datetime class (literal name, import alias, or fully-qualified form).
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    aliases = _collect_datetime_aliases(tree)
    return [(n.lineno, ast.unparse(n)) for n in _now_call_nodes(tree, aliases)]


def test_no_datetime_now_outside_clock() -> None:
    """No .py file in this repo other than clock.py may invoke datetime.now()
    or datetime.utcnow() in any form (direct, aliased, or fully-qualified).
    """
    allowed_resolved = {p.resolve() for p in ALLOWED_FILES}
    offenders: list[str] = []
    for path in _iter_python_files():
        if path.resolve() in allowed_resolved:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, src in _find_datetime_now_calls(text):
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"  {rel}:{lineno}: {src}")
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
