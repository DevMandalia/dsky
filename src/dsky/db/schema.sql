-- dsky event log: append-only, hash-chained, indexed for entity and type scans.

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    entity_id   TEXT    NOT NULL,
    actor       TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    prev_hash   TEXT    NOT NULL,
    hash        TEXT    NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_events_entity  ON events (entity_id, id);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events (event_type, id);

-- -----------------------------------------------------------------------
-- Projections (read models)
--
-- Derived state, rebuilt by db/projections.py:rebuild_projections().
-- AGENTS.md: "Current state is a projection, never edited in place."
-- No code path other than rebuild_projections() may write to these tables.
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS current_ideas (
    entity_id      TEXT    PRIMARY KEY,         -- one row per idea
    current_state  TEXT    NOT NULL,            -- registered | approved | rejected | deprecated | ...
    last_event_id  INTEGER NOT NULL,            -- source of truth for "where I am"
    updated_ts     TEXT    NOT NULL             -- ts of that last event
);

CREATE TABLE IF NOT EXISTS rejection_ledger (
    fingerprint  TEXT    NOT NULL,
    asset        TEXT    NOT NULL,
    reason       TEXT    NOT NULL,
    rejected_ts  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rejection_ledger_fingerprint ON rejection_ledger (fingerprint);

CREATE TABLE IF NOT EXISTS playbooks (
    entity_id          TEXT    PRIMARY KEY,     -- one row per playbook
    approval_type      TEXT    NOT NULL,        -- paper | live
    monitoring_status  TEXT    NOT NULL,        -- active | paused | halted
    last_event_id      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_positions (
    -- Placeholder; populated by a later step (research/backtest paper fills).
    id  INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS attribution_history (
    -- Placeholder; populated by a later step (research/attribution rolls).
    id  INTEGER PRIMARY KEY
);
