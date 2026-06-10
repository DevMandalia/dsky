# Project Invariants

These rules are binding on every task in this project.

---

**Determinism**: no LLM call ever produces a financial calculation, statistic, price, fill, or risk figure. Those come only from code in `research/`, `data/`, `lifecycle/`.

**Single write path**: the only function that writes to the database is `append_event()` in `db/events.py`. Nothing in `advisory/` may import `db.events`.

**Append-only**: decision-relevant facts are immutable events. Current state is a projection, never edited in place.

**Time discipline**: every data point carries `event_time` and `available_time`. The backtest may only read points where `available_time <= decision_time`. "Now" comes only from `clock.py`, never `datetime.now()` directly.

**Pre-registration before backtest**: a spec must be frozen and hashed before any backtest reads data; the lifecycle state machine must make the illegal order impossible.

**Tests are the spec.** When I ask for a test first, write the failing test before the implementation. Never weaken a test to make code pass — ask me.

**Prefer clarity over cleverness.** Do not add dependencies without asking. Implement only the unit I ask for in each prompt; do not scaffold ahead.
