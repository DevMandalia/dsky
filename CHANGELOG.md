# Changelog

All notable changes to dsky are recorded in the git commit log
(see `git log --oneline`). This file exists so anyone landing on
the repo can see the high-level shape of the project at a glance.

## Current state

- **Event log** (`src/dsky/db/events.py`): hash-chained, append-only,
  single write path via `append_event(..., clock: Clock)`. Genesis
  seed: `"dsky-genesis-v1"`. `verify_chain(conn)` returns the id of
  the first tampered row, or `None` if intact.

- **Projections** (`src/dsky/db/projections.py`): derived state rebuilt
  by replaying the event log in id order. Atomic via
  `try/except/rollback/raise`. Three live tables
  (`current_ideas`, `rejection_ledger`, `playbooks`) and two
  placeholders (`paper_positions`, `attribution_history`).

- **Clock** (`src/dsky/clock.py`): `Clock` protocol with two
  implementations — `SystemClock` (production) and `FrozenClock`
  (tests). The only legal source of "now".

- **Discipline scans** (`tests/test_clock_discipline.py`,
  `tests/test_events.py`): two AST-based scans. The clock scan
  covers the whole repo and fails the suite on any
  `datetime.now()` outside `clock.py`. The write scan covers
  `src/dsky/` and fails on any raw `INSERT/UPDATE/DELETE` outside
  `db/events.py` and `db/projections.py`.

- **Test suite**: 29 tests, ~0.08s. Ruff `select=ALL` and mypy
  `strict=true` on `src/dsky` both clean.

See `docs/architecture.html` for the engineering-wiki visual
overview (eight sections, real identifiers, real diagrams).
