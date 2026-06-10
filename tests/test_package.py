"""Smoke test: the dsky package imports cleanly."""
import dsky


def test_dsky_imports() -> None:
    assert dsky.__name__ == "dsky"
