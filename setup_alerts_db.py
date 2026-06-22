"""
setup_alerts_db.py — Create or verify the unified bet_alerts database.

Sport-agnostic: MLB, WNBA, and future models all write into the same table.
Safe to re-run — CREATE TABLE/INDEX IF NOT EXISTS, never touches existing rows.

Usage:
    python setup_alerts_db.py
    python setup_alerts_db.py --db /home/picks/alerts.db
    ALERTS_DB_PATH=/home/picks/alerts.db python setup_alerts_db.py
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent / "alerts.db"

_DDL = """
CREATE TABLE IF NOT EXISTS bet_alerts (
    id              INTEGER PRIMARY KEY,
    sport           TEXT    NOT NULL,       -- 'MLB', 'WNBA', 'NFL', etc.
    model_version   TEXT,                   -- 'v1.0'
    alert_date      TEXT    NOT NULL,       -- 'YYYY-MM-DD' in ET
    alert_time      TEXT,                   -- UTC ISO-8601 timestamp
    game_id         TEXT,                   -- Kambi event_id or MLB game_id
    player_name     TEXT,                   -- NULL for game-level bets
    market_type     TEXT,                   -- 'Total Runs', 'Player Points'
    direction       TEXT,                   -- 'OVER', 'UNDER', 'YES'
    line            REAL,                   -- threshold or total
    odds            INTEGER,                -- American odds at alert time
    predicted_value REAL,                   -- model's point/stat prediction
    model_prob      REAL,                   -- P(win) from model
    implied_prob    REAL,                   -- P(win) from market at alert time
    edge_prob       REAL,                   -- model_prob - implied_prob
    ev              REAL,                   -- edge_prob * (decimal_odds - 1)
    kelly_half      REAL,                   -- half-Kelly fraction (NULL for single-sided YES bets)
    opening_line    REAL,                   -- line/odds when market first opened (MLB: total, WNBA: NULL)
    alert_line      REAL,                   -- line/odds when alert was written (MLB: total, WNBA: NULL)
    closing_odds    INTEGER,                -- American odds at tipoff (filled by grader)
    closing_implied REAL,                   -- 1/decimal(closing_odds) (filled by grader)
    actual_result   REAL,                   -- actual score or stat (filled by grader)
    result          TEXT    DEFAULT 'PENDING', -- 'WIN', 'LOSS', 'PUSH', 'PENDING'
    profit_units    REAL,                   -- +/- at $1 flat stake (filled by grader)
    clv             REAL,                   -- closing_implied - alert_implied (prob units)
    clv_beat        INTEGER,                -- 1 = positive CLV, 0 = negative, NULL = no close available
    notified        INTEGER NOT NULL DEFAULT 0,  -- 1 = SMS sent
    notified_at     TEXT,                   -- UTC ISO timestamp when SMS was sent
    graded          INTEGER NOT NULL DEFAULT 0,  -- 1 = result filled in
    graded_at       TEXT,                   -- UTC ISO timestamp when graded
    notes           TEXT
);

-- Fast lookup: queries by sport + date (daily summary, grading)
CREATE INDEX IF NOT EXISTS idx_bet_alerts_sport_date
    ON bet_alerts(sport, alert_date);

-- Fast lookup: grading job queries notified=1 AND graded=0
CREATE INDEX IF NOT EXISTS idx_bet_alerts_notified
    ON bet_alerts(notified, graded);

-- Dedup guard: one alert row per (sport, date, game, player, direction, line).
-- Second cron run does INSERT OR IGNORE + conditional UPDATE (no duplicate rows).
CREATE UNIQUE INDEX IF NOT EXISTS idx_bet_alerts_dedup
    ON bet_alerts(sport, alert_date, game_id, COALESCE(player_name, ''), direction, line);
"""


def setup(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    conn = sqlite3.connect(db_path)
    conn.executescript(_DDL)
    conn.commit()

    row_count = conn.execute("SELECT COUNT(*) FROM bet_alerts").fetchone()[0]
    conn.close()

    verb = "Created" if is_new else "Verified"
    print(f"{verb}: {db_path}")
    print(f"  Existing rows: {row_count}")
    print("  Indexes: idx_bet_alerts_sport_date, idx_bet_alerts_notified, "
          "idx_bet_alerts_dedup (unique)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Create or verify the unified bet_alerts SQLite database."
    )
    p.add_argument(
        "--db", default=None, metavar="PATH",
        help=f"Path to alerts.db (default: ALERTS_DB_PATH env var or {_DEFAULT_PATH})",
    )
    args = p.parse_args(argv)

    db_path = args.db or os.environ.get("ALERTS_DB_PATH") or str(_DEFAULT_PATH)
    setup(db_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
