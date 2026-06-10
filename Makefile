.PHONY: test lint typecheck check

# Run the test suite.
test:
	uv run pytest

# Lint with ruff.
lint:
	uv run ruff check src tests

# Static type check with mypy.
typecheck:
	uv run mypy src

# All of the above. Local default target.
check: lint typecheck test
