-- dsky event log: append-only, hash-chained, indexed for entity and type scans.

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,          -- ISO-8601 UTC, e.g. 2024-01-15T12:30:45.000000+00:00
    event_type  TEXT    NOT NULL,
    entity_id   TEXT    NOT NULL,
    actor       TEXT    NOT NULL,          -- e.g. human:D, code:backtest, llm:critic
    payload     TEXT    NOT NULL,          -- canonical JSON (sorted keys, no extra whitespace)
    prev_hash   TEXT    NOT NULL,
    hash        TEXT    NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_events_entity  ON events (entity_id, id);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events (event_type, id);
