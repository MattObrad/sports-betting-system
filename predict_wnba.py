"""
predict_wnba.py — Daily WNBA Player Points prediction script.

Pipeline:
    1. VPS Postgres: load today's WNBA games + Player Points milestone props
                    (most recent snapshot per player/event/line)
    2. VPS Postgres: load game totals from odds_snapshots (for display context)
    3. sports.db:    match Kambi player names via id_aliases → internal_player_id
    4. sports.db:    rolling-5 weighted average → predicted_points + minutes_last_5
    5. Sigma:        bucket minutes_last_5 → sigma from calibration JSON
    6. Probability:  P(hit milestone) = prob_over(predicted, line, sigma)
    7. Implied:      raw 1/decimal(over_odds) — VIG INCLUDED, NOT de-vigged
                     (single-sided market; props_snapshots.under_odds is 100%
                     NULL, so there is no Under price to de-vig against → the
                     resulting edge is conservative by design)
    8. Edge filter:  edge >= min_edge_prob AND model_prob >= min_model_prob
    9. VPS write:    insert into edges + predictions tables
   10. SMS:          notify per qualifying edge

Exit codes:
    0  Completed normally (0 or more edges is not an error)
    1  Fatal error (DB unavailable, config missing, etc.)
    2  No WNBA games scheduled today
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from scipy.stats import norm

_DIR = Path(__file__).resolve().parent

load_dotenv(_DIR / ".env", override=False, encoding="utf-8-sig")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_DIR / "logs" / "predict_wnba.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else _DIR / "config.json"
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def cfg(config: dict, *keys, default=None):
    node = config
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


# ---------------------------------------------------------------------------
# Rolling-5 weighted prediction (mirrors SportsProjects baseline_model.py)
# ---------------------------------------------------------------------------
# Weights: alpha=0.4, decay=0.6, most-recent-first, then re-normalized.
_ALPHA = 0.4
_DECAY = 0.6
_N = 5
_WEIGHTS_RAW = [_ALPHA * _DECAY ** i for i in range(_N)]  # most-recent-first


def weighted_predict(prior_points_recent_first: list[float]) -> float | None:
    """
    Exponentially-weighted average of up to N=5 prior games.
    prior_points_recent_first: most-recent game first.
    Returns None if list is empty.
    Re-normalizes over however many games are available (K < N fallback).
    """
    if not prior_points_recent_first:
        return None
    k = len(prior_points_recent_first)
    w = _WEIGHTS_RAW[:k]
    w_sum = sum(w)
    w_norm = [wi / w_sum for wi in w]
    return sum(w_norm[i] * prior_points_recent_first[i] for i in range(k))


# ---------------------------------------------------------------------------
# Normal probability helpers
# ---------------------------------------------------------------------------

def prob_over(predicted: float, line: float, sigma: float) -> float:
    """P(X > line) under Normal(predicted, sigma). Uses survival function."""
    return float(norm.sf(line, loc=predicted, scale=sigma))


def decimal_from_american(american_odds: int) -> float:
    if american_odds > 0:
        return round(american_odds / 100 + 1, 6)
    else:
        return round(-100 / american_odds + 1, 6)


def implied_prob(american_odds: int) -> float:
    """Raw implied probability — VIG INCLUDED. This is NOT a de-vigged price.

    Player Points milestones are single-sided: props_snapshots.under_odds is
    100% NULL (verified 2026-05-31: 0 of 30,007 Player Points rows carry an
    Under). With no Under side and no certain low ladder rung to anchor on,
    the vig cannot be stripped without an arbitrary assumption — so we keep
    the raw 1/decimal on purpose.

    Consequence: implied_prob > fair prob, so edge = model_prob - implied_prob
    is UNDERSTATED (conservative) — the model must beat fair + the full vig.
    If an Under price ever starts appearing, switch to
    src.betting.odds_math.no_vig_probabilities(over_odds, under_odds).
    """
    return 1.0 / decimal_from_american(american_odds)


def market_expected_points(lines_odds: list[tuple[float, int]]) -> float | None:
    """
    Market-implied expected point total for ONE player, extracted from the
    milestone ladder as the area under the (vig-included) survival curve:
        E[X] = ∫ S(x) dx  ≈ trapezoid over the points
        (0, 1.0), (T_1, p_1), ..., (T_k, p_k), (T_k + step, 0)
    where p_i = 1/decimal(over_odds_i) is the raw implied P(X >= T_i).

    Returns None if fewer than 2 ladder points (can't anchor).

    NOTES
      * Survival-integral form, NOT the literal left-edge bucket sum
        sum(T_i*(p_i - p_{i+1})): that form is biased ~2 pts low and silently
        drops the probability mass below the lowest line. The integral handles
        both tails.
      * VIG-INCLUDED (single-sided market; no Under to de-vig against), so this
        runs slightly high — i.e. a conservative anchor (pulls the model down a
        touch less than a true no-vig market would).
    """
    pts = sorted({(float(L), int(o)) for L, o in lines_odds}, key=lambda x: x[0])
    if len(pts) < 2:
        return None
    xs = [0.0]
    ss = [1.0]
    for T, o in pts:
        p = 1.0 / decimal_from_american(o)
        ss.append(min(p, ss[-1]))      # enforce monotone non-increasing survival
        xs.append(T)
    step = pts[-1][0] - pts[-2][0]
    xs.append(pts[-1][0] + (step if step > 0 else 3.0))
    ss.append(0.0)
    return sum(0.5 * (ss[i - 1] + ss[i]) * (xs[i] - xs[i - 1])
               for i in range(1, len(xs)))


# ---------------------------------------------------------------------------
# Sigma calibration lookup
# ---------------------------------------------------------------------------

def load_calibration(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lookup_sigma(
    calibration: dict,
    model_name: str,
    model_version: str,
    minutes_last_5: float | None,
    bucket_edges: list[int],
    bucket_labels: list[str],
    fallback_label: str,
) -> tuple[float | None, str]:
    """Return (sigma, bucket_label_used). Mirrors SportsProjects lookup_sigma."""
    key = f"{model_name}__{model_version}"
    models = calibration.get("models", {})
    if key not in models:
        return (None, "no_calibration")
    buckets = models[key]
    fallback = buckets.get(fallback_label, {"sigma": None})

    if minutes_last_5 is None or math.isnan(minutes_last_5):
        return (fallback["sigma"], fallback_label)

    if minutes_last_5 < bucket_edges[0]:
        label = bucket_labels[0]
    elif minutes_last_5 < bucket_edges[1]:
        label = bucket_labels[1]
    else:
        label = bucket_labels[2]

    chosen = buckets.get(label, {"sigma": None})
    if chosen["sigma"] is None:
        return (fallback["sigma"], fallback_label)
    return (chosen["sigma"], label)


# ---------------------------------------------------------------------------
# EdgeBet
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ARBITRAGE / INFORMATION-EDGE DETECTORS (model-free)
# ---------------------------------------------------------------------------
# These run independently of the (anchored, edge-less) points model. Per the
# market-anchoring finding, the points model has no box-score edge; the only
# plausible edges are (a) mechanical ladder pricing errors and (b) reacting to
# line movement / news faster than the book. These detectors target (a) and (b).

def detect_ladder_incoherence(player_props: list[dict]) -> list[dict]:
    """
    Detect monotonicity violations in ONE player's milestone ladder AT ONE
    POINT IN TIME (a single snapshot). Pure arbitrage — no model needed.

    LAW: P(X >= T) is non-increasing in T, so a higher threshold must have a
    LOWER (or equal) implied probability than a lower threshold. Equivalently,
    the higher threshold must pay MORE (larger decimal odds). A violation
    P_implied(higher) > P_implied(lower) means the LOWER threshold STRICTLY
    DOMINATES the higher: it wins whenever the higher wins (a player scoring
    20 also scored 15) AND pays at least as much. Betting the higher line is
    then never correct — a pure pricing error.

    player_props: list of dicts with keys 'line', 'over_odds' (one snapshot).
    Returns a list of violation dicts (low/high lines, probs, and severity).

    DEDUP: the same line can appear multiple times in one snapshot with
    different prices (data reality — verified 9,950 such WNBA cases). For
    monotonicity we collapse each DISTINCT line to its BEST available price
    (lowest implied prob = highest payout) and only compare STRICTLY
    increasing thresholds. Same-line/different-price is a separate phenomenon,
    not a ladder inversion, and is excluded here.
    """
    best: dict[float, float] = {}   # line -> best (lowest) implied prob
    for p in player_props:
        if p.get("over_odds") is None:
            continue
        T = float(p["line"])
        ip = implied_prob(int(p["over_odds"]))
        if T not in best or ip < best[T]:
            best[T] = ip
    rungs = sorted(best.items(), key=lambda x: x[0])   # [(line, implied), ...]
    flags: list[dict] = []
    for i in range(len(rungs) - 1):
        lo_T, p_lo = rungs[i]
        hi_T, p_hi = rungs[i + 1]
        if hi_T <= lo_T:          # strictly-increasing guard (defensive)
            continue
        if p_hi > p_lo + 1e-9:    # higher threshold priced MORE likely → illegal
            flags.append({
                "low_line": lo_T, "p_low": p_lo,
                "high_line": hi_T, "p_high": p_hi,
                "severity": p_hi - p_lo,   # prob-unit inversion magnitude
                "notes": (f"P({hi_T:.0f}+)={p_hi:.3f} > P({lo_T:.0f}+)={p_lo:.3f} "
                          f"— pricing error (lower line dominates: better/equal "
                          f"payout AND easier to hit)"),
            })
    return flags


def detect_line_movement(open_odds: int, current_odds: int,
                         model_prob: float | None = None,
                         threshold_pts: int = 20) -> dict | None:
    """
    Classify open→current over-odds movement for one player/threshold.

      TYPE A 'SHARP'      : odds SHORTENED >= threshold_pts (move <= -T).
                            Money came in on the over — follow the action.
      TYPE B 'VALUE_FADE' : odds LENGTHENED >= threshold_pts (move >= +T)
                            AND the model still likes the over (model edge > 0
                            vs the *current* price). Combined contrarian signal.

    Returns a dict (type, american_move, implied_move_pp, ...) or None if no
    qualifying movement.

    CAVEAT (honest): the trigger uses the raw American-points delta the spec
    asked for, but American odds are discontinuous across +/-100, so the same
    point delta means very different things (e.g. -110→+110 is +220 'points'
    but only ~+4pp of probability). We therefore ALSO report implied_move_pp
    (prob-unit move) which is the statistically sound measure; prefer it when
    the two disagree.
    """
    move = current_odds - open_odds   # American-points delta, per spec
    implied_move_pp = (implied_prob(current_odds) - implied_prob(open_odds)) * 100.0
    if move <= -threshold_pts:
        return {"type": "SHARP", "american_move": move,
                "implied_move_pp": implied_move_pp}
    if move >= threshold_pts and model_prob is not None:
        cur_implied = implied_prob(current_odds)
        if model_prob - cur_implied > 0:   # model still likes the over
            return {"type": "VALUE_FADE", "american_move": move,
                    "implied_move_pp": implied_move_pp}
    return None


@dataclass
class EdgeBet:
    player_name: str
    market_type: str
    line: float
    over_odds: int
    event_id: str
    home_team: str
    away_team: str
    game_time: datetime
    predicted_points: float
    minutes_last_5: float | None
    sigma: float
    sigma_bucket: str
    model_prob: float
    implied_prob_val: float
    edge: float
    game_total: float | None = None
    implied_team_total: float | None = None
    n_prior_games: int = 0

    @property
    def threshold_label(self) -> str:
        return f"{math.ceil(self.line)}+"

    @property
    def ev(self) -> float:
        """
        True expected value per $1 staked:
            ev = model_prob × decimal_odds − 1

        The previous form `edge × (decimal_odds − 1)` was WRONG: it omitted
        the returned stake and understated EV by exactly `edge`, which
        mis-ranked thresholds inside select_best_line. (For reference, the
        algebraic identity is model_prob×decimal − 1 == edge × decimal when
        edge is measured against the raw 1/decimal implied prob — note the
        factor is `decimal`, not `decimal − 1`.)
        Used to rank multiple thresholds for the same player.
        """
        return self.model_prob * decimal_from_american(self.over_odds) - 1.0

    @property
    def game_time_et_str(self) -> str:
        et_dt = self.game_time + timedelta(hours=-4)
        return et_dt.strftime("%-I:%M%p ET").lower()

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"


# ---------------------------------------------------------------------------
# VPS Postgres: load today's props
# ---------------------------------------------------------------------------
_PROPS_SQL = """
WITH latest AS (
    SELECT DISTINCT ON (ps.event_id, ps.player_name, ps.line)
        ps.event_id,
        ps.player_name,
        ps.market_type,
        ps.line,
        ps.over_odds,
        ps.snapshot_time
    FROM props_snapshots ps
    WHERE ps.market_type = 'Player Points'
      AND ps.over_odds IS NOT NULL
    ORDER BY ps.event_id, ps.player_name, ps.line, ps.snapshot_time DESC, ps.over_odds DESC
)
SELECT
    l.event_id,
    l.player_name,
    l.market_type,
    l.line,
    l.over_odds,
    l.snapshot_time,
    g.home_team,
    g.away_team,
    g.game_time,
    g.status
FROM latest l
JOIN games g ON g.event_id = l.event_id
WHERE g.league = 'WNBA'
  AND DATE(g.game_time AT TIME ZONE 'America/New_York') = %s
ORDER BY l.player_name, l.line
"""

_GAME_TOTAL_SQL = """
SELECT DISTINCT ON (os.event_id)
    os.event_id,
    os.line AS game_total
FROM odds_snapshots os
WHERE os.event_id = ANY(%s)
  AND os.market_type = 'Total Points'
  AND os.outcome = 'Over'
ORDER BY os.event_id, os.snapshot_time DESC
"""


def load_todays_props(pg_cur, today_et: str) -> list[dict]:
    pg_cur.execute(_PROPS_SQL, (today_et,))
    cols = [d[0] for d in pg_cur.description]
    return [dict(zip(cols, row)) for row in pg_cur.fetchall()]


def load_game_totals(pg_cur, event_ids: list[str]) -> dict[str, float]:
    if not event_ids:
        return {}
    pg_cur.execute(_GAME_TOTAL_SQL, (event_ids,))
    return {row[0]: float(row[1]) for row in pg_cur.fetchall()}


# ---------------------------------------------------------------------------
# sports.db helpers
# ---------------------------------------------------------------------------

def connect_alerts_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


_MATERIAL_THRESH = 0.005   # 0.5 prob-unit change triggers a row update


def upsert_bet_alert(
    conn: sqlite3.Connection,
    bet: "EdgeBet",
    today: str,
    model_version: str,
) -> dict:
    """
    INSERT OR IGNORE + conditional UPDATE on the dedup unique index.
    Returns {'id': int|None, 'action': 'inserted'|'updated'|'unchanged', 'notified': int}.

    Second cron run (1pm → 6pm): if odds or model_prob changed materially,
    update the existing row so the row always shows the freshest prediction.
    notified stays unchanged so we never re-SMS a bet that was already sent.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO bet_alerts
            (sport, model_version, alert_date, alert_time, game_id, player_name,
             market_type, direction, line, odds,
             predicted_value, model_prob, implied_prob, edge_prob, ev,
             result)
        VALUES ('WNBA', ?, ?, ?, ?, ?,
                'Player Points', 'YES', ?, ?,
                ?, ?, ?, ?, ?,
                'PENDING')
        """,
        (model_version, today, now, bet.event_id, bet.player_name,
         bet.line, bet.over_odds,
         bet.predicted_points, bet.model_prob, bet.implied_prob_val,
         bet.edge, bet.ev),
    )
    conn.commit()

    if cur.rowcount == 1:
        return {"id": cur.lastrowid, "action": "inserted", "notified": 0}

    # Row already existed — fetch current state
    row = conn.execute(
        """
        SELECT id, model_prob, odds, notified
        FROM bet_alerts
        WHERE sport = 'WNBA' AND alert_date = ? AND game_id = ?
          AND COALESCE(player_name, '') = COALESCE(?, '') AND direction = 'YES' AND line = ?
        """,
        (today, bet.event_id, bet.player_name, bet.line),
    ).fetchone()

    if row is None:
        return {"id": None, "action": "error", "notified": 0}

    row_id, existing_prob, existing_odds, notified = (
        row["id"], row["model_prob"], row["odds"], row["notified"]
    )

    # Update if model_prob or odds changed materially
    prob_delta  = abs(bet.model_prob - (existing_prob or 0))
    odds_moved  = bet.over_odds != existing_odds
    if prob_delta > _MATERIAL_THRESH or odds_moved:
        conn.execute(
            """
            UPDATE bet_alerts
            SET alert_time = ?, odds = ?, predicted_value = ?,
                model_prob = ?, implied_prob = ?, edge_prob = ?, ev = ?
            WHERE id = ?
            """,
            (now, bet.over_odds, bet.predicted_points,
             bet.model_prob, bet.implied_prob_val, bet.edge, bet.ev,
             row_id),
        )
        conn.commit()
        return {"id": row_id, "action": "updated", "notified": notified}

    return {"id": row_id, "action": "unchanged", "notified": notified}


def mark_alerts_notified(conn: sqlite3.Connection, alert_ids: list[int]) -> None:
    if not alert_ids:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.executemany(
        "UPDATE bet_alerts SET notified = 1, notified_at = ? WHERE id = ?",
        [(now, aid) for aid in alert_ids],
    )
    conn.commit()


def connect_sports_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def lookup_player_id(sports_conn: sqlite3.Connection, kambi_name: str) -> int | None:
    row = sports_conn.execute(
        "SELECT internal_id FROM id_aliases "
        "WHERE source = 'kambi' AND source_id = ? LIMIT 1",
        (kambi_name,),
    ).fetchone()
    return None if row is None else int(row[0])


_ACTIVE_FILTER = "did_not_play = 0 AND minutes > 0 AND points IS NOT NULL"

_PRIOR_GAMES_SQL = (
    f"SELECT game_date, internal_game_id, points, minutes "
    f"FROM wnba_player_box_scores "
    f"WHERE internal_player_id = ? "
    f"  AND game_date < ? "
    f"  AND {_ACTIVE_FILTER} "
    f"ORDER BY game_date DESC, internal_game_id DESC "
    f"LIMIT ?"
)

_SEASON_AVG_SQL = (
    "SELECT AVG(points) FROM wnba_player_box_scores "
    "WHERE internal_player_id = ? "
    "  AND game_date >= ? AND game_date < ? "
    "  AND did_not_play = 0 AND minutes > 0 AND points IS NOT NULL"
)


def compute_rolling_prediction(
    sports_conn: sqlite3.Connection,
    player_id: int,
    as_of_date: str,
    n: int = 5,
    rolling_weight: float = 0.4,
    season_weight: float = 0.6,
) -> tuple[float | None, float | None, int]:
    """
    Returns (predicted_points, minutes_last_5, n_prior_games).

    Blends rolling-5 weighted average with season-to-date average to
    reduce mean-reversion bias from hot/cold streaks:
        pred = rolling_weight × rolling5 + season_weight × season_avg

    Falls back to rolling5 alone when no season-to-date data exists
    (e.g. player's first few games of the calendar year).
    """
    rows = sports_conn.execute(_PRIOR_GAMES_SQL, (player_id, as_of_date, n)).fetchall()
    if not rows:
        return (None, None, 0)

    points_recent_first  = [float(r["points"]) for r in rows]
    minutes_recent_first = [float(r["minutes"]) for r in rows]

    rolling5    = weighted_predict(points_recent_first)
    minutes_last_5 = (sum(minutes_recent_first) / len(minutes_recent_first)
                      if len(rows) >= n else None)

    # Season-to-date average (current calendar year, strictly before as_of_date)
    season_start = f"{as_of_date[:4]}-01-01"
    szn_row = sports_conn.execute(_SEASON_AVG_SQL,
                                  (player_id, season_start, as_of_date)).fetchone()
    season_avg = float(szn_row[0]) if szn_row and szn_row[0] is not None else None

    if season_avg is not None:
        predicted = rolling_weight * rolling5 + season_weight * season_avg
    else:
        predicted = rolling5   # first games of the year: no season avg yet

    return (round(predicted, 2), minutes_last_5, len(rows))


# ---------------------------------------------------------------------------
# VPS write helpers
# ---------------------------------------------------------------------------

_INSERT_EDGE_SQL = """
INSERT INTO edges (event_id, detected_at, market_type, player_name,
                   predicted_value, market_line, edge_pct, notified)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_PREDICTION_SQL = """
INSERT INTO predictions (event_id, created_at, sport, market_type,
                         predicted_value, model_name)
VALUES (%s, %s, %s, %s, %s, %s)
"""


def write_to_vps(pg_conn, edges: list[EdgeBet], all_props: list[dict], model_name: str) -> None:
    now = datetime.now(timezone.utc)
    cur = pg_conn.cursor()
    try:
        # Write all predictions (not just edges)
        for prop in all_props:
            if prop.get("predicted_points") is not None:
                cur.execute(_INSERT_PREDICTION_SQL, (
                    prop["event_id"],
                    now,
                    "WNBA",
                    prop["market_type"],
                    round(prop["predicted_points"], 4),
                    model_name,
                ))

        # Write edge records
        for bet in edges:
            cur.execute(_INSERT_EDGE_SQL, (
                bet.event_id,
                now,
                bet.market_type,
                bet.player_name,
                round(bet.predicted_points, 4),
                bet.line,
                round(bet.edge * 100, 2),  # store as percentage
                False,
            ))

        pg_conn.commit()
        log.info("Wrote %d predictions and %d edges to VPS.", len(all_props), len(edges))
    except Exception as exc:
        pg_conn.rollback()
        log.error("VPS write failed: %s", exc)


def mark_edges_notified(pg_conn, edges: list[EdgeBet]) -> None:
    if not edges:
        return
    cur = pg_conn.cursor()
    for bet in edges:
        cur.execute(
            "UPDATE edges SET notified = TRUE "
            "WHERE event_id = %s AND player_name = %s AND market_type = %s "
            "  AND market_line = %s AND notified = FALSE",
            (bet.event_id, bet.player_name, bet.market_type, bet.line),
        )
    pg_conn.commit()


# ---------------------------------------------------------------------------
# Best-line selection
# ---------------------------------------------------------------------------

def select_agreement_alerts(edges: list[EdgeBet],
                            min_agreement: int = 2) -> list[EdgeBet]:
    """
    Agreement-gated selection — replaces argmax-EV per player.

    WHY: picking the single highest-EV threshold per player is an argmax over
    many noisy estimates, so it preferentially surfaces the threshold with the
    largest positive estimation error (optimizer's curse) — the bet most
    likely to be noise.

    INSTEAD: a player qualifies for an alert only if >= min_agreement of their
    thresholds INDEPENDENTLY clear the edge/prob floors (convergent evidence).
    For a qualifying player we then alert on the LOWEST qualifying line
    (= highest P(hit)) — the most conservative expression of the signal, not
    the highest-EV tail. All qualifying edges are still written to the DB;
    only agreement-confirmed players are SMS'd.
    """
    from collections import defaultdict
    groups: dict[tuple, list[EdgeBet]] = defaultdict(list)
    for bet in edges:
        groups[(bet.player_name, bet.event_id)].append(bet)

    selected: list[EdgeBet] = [
        min(bets, key=lambda b: b.line)
        for bets in groups.values()
        if len(bets) >= min_agreement
    ]
    return sorted(selected, key=lambda b: b.model_prob, reverse=True)


# ---------------------------------------------------------------------------
# Bet slip printing
# ---------------------------------------------------------------------------

def print_bet_slip(
    all_props: list[dict],
    edges: list[EdgeBet],
    best_edges: list[EdgeBet],
    today: str,
    show_all: bool = False,
) -> None:
    """
    show_all=False (default): print only the best-EV line per player (SMS candidates).
    show_all=True (--show-all-lines): also print the full table of every threshold.
    edges are always written to the DB regardless of this flag.
    """
    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  WNBA PLAYER POINTS — %s", today)
    log.info("══════════════════════════════════════════════════════════════")

    # ── Full all-lines table (only when --show-all-lines) ────────────────────
    if show_all:
        log.info("  %-22s %-6s %-7s %-6s %-6s %-7s %-6s %-7s  %s",
                 "Player", "Line", "Odds", "Pred", "σ", "P(hit)", "Impl", "Edge", "EV")
        log.info("  " + "─" * 86)
        for prop in all_props:
            pred   = prop.get("predicted_points")
            m_prob = prop.get("model_prob")
            i_prob = prop.get("implied_prob_val")
            e      = prop.get("edge", 0.0)
            ev     = prop.get("ev", 0.0)
            flag   = " *" if prop.get("is_edge") else ""
            log.info(
                "  %-22s %4.1f+  %+6d  %5.1f  %5.1f  %5.1f%%  %5.1f%%  %+5.1f%%  %+5.3f%s",
                prop["player_name"][:22],
                prop["line"],
                prop["over_odds"],
                pred if pred is not None else float("nan"),
                prop.get("sigma", float("nan")),
                (m_prob or 0) * 100,
                (i_prob or 0) * 100,
                e * 100,
                ev,
                flag,
            )
        log.info("")

    # ── Best-line SMS candidates ─────────────────────────────────────────────
    suppressed = len(edges) - len(best_edges)
    if best_edges:
        log.info("  ── BEST LINES — SMS ALERTS (%d player%s) %s",
                 len(best_edges),
                 "" if len(best_edges) == 1 else "s",
                 f"[{suppressed} lower-EV threshold{'s' if suppressed != 1 else ''} logged to DB only]"
                 if suppressed else "")
        log.info("  %-22s %-6s %-7s %-6s %-7s %-6s %-7s",
                 "Player", "Line", "Odds", "Pred", "P(hit)", "Edge", "EV")
        log.info("  " + "─" * 68)
        for bet in best_edges:
            gt_str = f"  GT:{bet.game_total:.0f}" if bet.game_total else ""
            log.info(
                "  %-22s %4s  %+6d  %5.1f  %6.1f%%  %+5.1f%%  %+5.3f%s",
                bet.player_name[:22], bet.threshold_label,
                bet.over_odds, bet.predicted_points,
                bet.model_prob * 100, bet.edge * 100, bet.ev,
                gt_str,
            )
    else:
        log.info("  No qualifying edges today.")

    if not show_all and edges:
        log.info("  (run with --show-all-lines to see all %d thresholds)", len(all_props))

    log.info("══════════════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Daily WNBA Player Points prediction — edge detection + SMS."
    )
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                   help="Predict for a specific ET date (default: today ET).")
    p.add_argument("--no-notify", action="store_true",
                   help="Run pipeline but skip SMS notifications.")
    p.add_argument("--no-save", action="store_true",
                   help="Print results only; do not write to VPS DB.")
    p.add_argument("--show-all-lines", action="store_true",
                   help="Print every threshold for every player (edge + EV) before the best-line summary.")
    p.add_argument("--test-sms", action="store_true",
                   help="Send a test SMS then exit.")
    p.add_argument("--config", default=None,
                   help="Path to config.json (default: config.json in script dir).")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    # -- Test SMS ------------------------------------------------------------
    if args.test_sms:
        try:
            from notify import send_test_sms
            success = send_test_sms(config)
            return 0 if success else 1
        except ImportError:
            log.error("notify.py not found.")
            return 1

    try:
        from notify import send_edge_sms
        has_notify = True
    except ImportError:
        has_notify = False
        log.warning("notify.py not found — SMS disabled.")

    # -- Date (ET) -----------------------------------------------------------
    et_offset = timedelta(hours=cfg(config, "prediction", "et_offset_hours", default=-4))
    today = args.date or (datetime.now(timezone.utc) + et_offset).date().isoformat()
    log.info("=== WNBA predictions for %s ===", today)

    # -- Config --------------------------------------------------------------
    model_name    = cfg(config, "model", "name", default="baseline_weighted_rolling_5")
    model_version = cfg(config, "model", "version", default="1.0")
    # SPORTS_DB_PATH env var takes precedence — VPS sets this to /home/picks/sports.db
    sports_db     = (os.environ.get("SPORTS_DB_PATH")
                     or cfg(config, "model", "sports_db_path",
                             default="D:/SportsProjects/data/sports.db"))
    # SIGMA_CAL_PATH env var takes precedence — VPS sets this to /home/picks/sigma_calibration_v1.0.json
    sigma_path    = (os.environ.get("SIGMA_CAL_PATH")
                     or cfg(config, "model", "sigma_calibration_path",
                             default="D:/SportsProjects/data/models/sigma_calibration_v1.0.json"))
    # CALIBRATION_PATH env var takes precedence — VPS path to the isotonic pickle
    calibration_path = (os.environ.get("CALIBRATION_PATH")
                        or cfg(config, "model", "calibration_path", default=None))
    bucket_edges  = cfg(config, "model", "minutes_bucket_edges", default=[15, 28])
    bucket_labels = cfg(config, "model", "minutes_bucket_labels",
                        default=["<15", "15-28", "28+"])
    fallback_lbl    = cfg(config, "model", "sigma_fallback_label", default="fallback_overall")
    sigma_multiplier  = cfg(config, "model", "sigma_multiplier",    default=1.0)
    rolling_weight    = cfg(config, "model", "blend_rolling_weight", default=0.4)
    season_weight     = cfg(config, "model", "blend_season_weight",  default=0.6)
    min_edge_prob     = cfg(config, "betting", "min_edge_prob",       default=0.06)
    min_model_prob  = cfg(config, "betting", "min_model_prob", default=0.60)
    edge_shrink     = cfg(config, "betting", "edge_shrink",           default=0.60)
    min_agreement   = cfg(config, "betting", "min_agreement_thresholds", default=2)
    max_odds      = cfg(config, "betting", "max_odds_american", default=-300)
    n_lookback    = cfg(config, "model", "n_lookback", default=5)
    vps_cfg       = cfg(config, "vps_db", default={})
    # ALERTS_DB_PATH env var — set to /home/picks/alerts.db on VPS
    alerts_db_path = os.environ.get("ALERTS_DB_PATH") or cfg(config, "alerts_db_path", default=None)

    # -- Load sigma calibration ----------------------------------------------
    try:
        calibration = load_calibration(sigma_path)
    except FileNotFoundError:
        log.error("Sigma calibration not found: %s", sigma_path)
        return 1

    # -- Load isotonic probability calibrator (optional, graceful) -----------
    # Maps raw Normal-CDF prob -> calibrated prob (corrects overconfidence).
    # If the pickle or sklearn/joblib is missing, fall back to RAW probs.
    calibrator = None
    if calibration_path:
        try:
            import joblib
            calibrator = joblib.load(calibration_path)
            log.info("Loaded probability calibrator: %s", calibration_path)
        except Exception as exc:
            log.warning("Calibrator unavailable (%s) — using RAW probabilities.", exc)
    else:
        log.info("No calibration_path configured — using RAW probabilities.")

    # -- Connect to VPS Postgres ---------------------------------------------
    try:
        pg_conn = psycopg2.connect(
            host=(os.environ.get("VPS_DB_HOST") or vps_cfg.get("host", "198.199.77.143")),
            port=vps_cfg.get("port", 5432),
            database=vps_cfg.get("database", "picksdb"),
            user=vps_cfg.get("user", "picksuser"),
            password=vps_cfg.get("password", "password"),
        )
        pg_cur = pg_conn.cursor()
        log.info("Connected to VPS Postgres.")
    except Exception as exc:
        log.error("Cannot connect to VPS Postgres: %s", exc)
        return 1

    # -- Load today's props --------------------------------------------------
    props = load_todays_props(pg_cur, today)
    if not props:
        log.info("No WNBA Player Points props found for %s.", today)
        pg_conn.close()
        return 2

    log.info("Loaded %d Player Points props for %s.", len(props), today)

    # -- Load game totals (for display context) ------------------------------
    event_ids = list({p["event_id"] for p in props})
    game_totals = load_game_totals(pg_cur, event_ids)
    log.info("Game totals available for %d/%d events.", len(game_totals), len(event_ids))

    # -- Connect to sports.db ------------------------------------------------
    try:
        sports_conn = connect_sports_db(sports_db)
        log.info("Connected to sports.db: %s", sports_db)
    except Exception as exc:
        log.error("Cannot open sports.db %s: %s", sports_db, exc)
        pg_conn.close()
        return 1

    # -- Process each prop ---------------------------------------------------
    all_props: list[dict] = []
    edges: list[EdgeBet] = []
    skipped_no_alias = 0
    skipped_no_pred = 0
    skipped_odds_cap = 0

    # Market-implied expected points per (player, event) from the FULL ladder
    # (all lines, incl. odds-capped favourites) — used to anchor the prediction.
    from collections import defaultdict as _dd
    _ladder: dict = _dd(list)
    for _p in props:
        if _p.get("over_odds") is not None:
            _ladder[(_p["player_name"], _p["event_id"])].append(
                (float(_p["line"]), int(_p["over_odds"])))
    market_exp_by_pe = {pe: market_expected_points(lo) for pe, lo in _ladder.items()}

    for prop in props:
        kambi_name  = prop["player_name"]
        line        = float(prop["line"])
        over_odds   = int(prop["over_odds"])
        event_id    = prop["event_id"]
        home_team   = prop["home_team"]
        away_team   = prop["away_team"]
        game_time   = prop["game_time"]  # datetime with tz

        # Game date in ET for lookback boundary
        game_date_et = (game_time + et_offset).date().isoformat()

        # Odds cap: skip extreme favourites (model has low resolution there)
        if over_odds < max_odds:
            skipped_odds_cap += 1
            continue

        # -- Player ID lookup --
        player_id = lookup_player_id(sports_conn, kambi_name)
        if player_id is None:
            log.debug("No alias for '%s' — skipping.", kambi_name)
            skipped_no_alias += 1
            continue

        # -- Blended prediction (rolling-5 × 0.4 + season avg × 0.6) --
        predicted, minutes_last_5, n_prior = compute_rolling_prediction(
            sports_conn, player_id, game_date_et, n=n_lookback,
            rolling_weight=rolling_weight, season_weight=season_weight,
        )
        if predicted is None:
            log.debug("%s: no prior games — cannot predict.", kambi_name)
            skipped_no_pred += 1
            continue

        # -- Market anchoring: blend the model prediction 50/50 with the
        #    market's own implied expected points. The market line becomes an
        #    anchor on the POINT prediction, not just a comparison target.
        model_pred = predicted
        market_exp = market_exp_by_pe.get((kambi_name, event_id))
        final_pred = (0.5 * model_pred + 0.5 * market_exp
                      if market_exp is not None else model_pred)

        # -- Sigma (calibrated × multiplier = effective) --
        sigma, sigma_bucket = lookup_sigma(
            calibration, model_name, model_version,
            minutes_last_5, bucket_edges, bucket_labels, fallback_lbl,
        )
        if sigma is None:
            log.warning("%s: no sigma available — skipping.", kambi_name)
            continue
        effective_sigma = sigma * sigma_multiplier

        # -- Probabilities on the MARKET-ANCHORED prediction → calibrated --
        raw_prob = prob_over(final_pred, line, effective_sigma)
        m_prob = (float(calibrator.predict([raw_prob])[0])
                  if calibrator is not None else raw_prob)
        i_prob = implied_prob(over_odds)
        raw_edge = m_prob - i_prob
        # Shrink the edge toward zero (~40% of apparent edge is noise).
        edge_val = raw_edge * edge_shrink
        # Disagreement penalty: a large model-vs-market gap is a RED FLAG
        # (when the model strongly disagrees with the market, the market is
        # usually right — confirmed by negative CLV on big-gap bets), so halve.
        if market_exp is not None and abs(model_pred - market_exp) > 4.0:
            edge_val *= 0.5

        # -- Game context --
        gt = game_totals.get(event_id)
        implied_tt = gt / 2 if gt else None

        # -- Record --
        is_edge = (edge_val >= min_edge_prob) and (m_prob >= min_model_prob)
        ev_val  = m_prob * decimal_from_american(over_odds) - 1.0  # true EV: p×dec − 1

        row = {
            "player_name": kambi_name,
            "market_type": prop["market_type"],
            "line": line,
            "over_odds": over_odds,
            "event_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "game_time": game_time,
            "predicted_points": final_pred,
            "model_pred": model_pred,
            "market_expected": market_exp,
            "minutes_last_5": minutes_last_5,
            "sigma": effective_sigma,
            "sigma_bucket": sigma_bucket,
            "model_prob": m_prob,
            "raw_model_prob": raw_prob,
            "implied_prob_val": i_prob,
            "edge": edge_val,
            "raw_edge": raw_edge,
            "ev": ev_val,
            "game_total": gt,
            "implied_team_total": implied_tt,
            "n_prior_games": n_prior,
            "is_edge": is_edge,
        }
        all_props.append(row)

        if is_edge:
            bet = EdgeBet(
                player_name=kambi_name,
                market_type=prop["market_type"],
                line=line,
                over_odds=over_odds,
                event_id=event_id,
                home_team=home_team,
                away_team=away_team,
                game_time=game_time,
                predicted_points=final_pred,
                minutes_last_5=minutes_last_5,
                sigma=effective_sigma,
                sigma_bucket=sigma_bucket,
                model_prob=m_prob,
                implied_prob_val=i_prob,
                edge=edge_val,
                game_total=gt,
                implied_team_total=implied_tt,
                n_prior_games=n_prior,
            )
            edges.append(bet)

    log.info(
        "Processed %d props: %d edges | %d skipped (no alias) | "
        "%d skipped (no pred) | %d skipped (odds cap)",
        len(all_props), len(edges), skipped_no_alias,
        skipped_no_pred, skipped_odds_cap,
    )

    # -- Agreement-gated selection (replaces argmax-EV; resists optimizer's curse)
    best_edges = select_agreement_alerts(edges, min_agreement)
    log.info(
        "Selection: %d qualifying threshold%s → %d player%s with >=%d agreeing "
        "threshold%s (alerting the lowest/safest qualifying line each).",
        len(edges),      "" if len(edges)      == 1 else "s",
        len(best_edges), "" if len(best_edges) == 1 else "s",
        min_agreement,   "" if min_agreement   == 1 else "s",
    )

    # -- Print slip ----------------------------------------------------------
    print_bet_slip(all_props, edges, best_edges, today,
                   show_all=args.show_all_lines)

    # -- VPS write: ALL edges logged, not just best -------------------------
    if not args.no_save:
        write_to_vps(pg_conn, edges, all_props, f"{model_name}_v{model_version}")

    sports_conn.close()

    # -- alerts.db upsert + dedup-aware SMS routing -------------------------
    # If ALERTS_DB_PATH is set: write all best_edges, only SMS ones with notified=0.
    # If not set (local testing without alerts.db): fall back to SMS all best_edges.
    alerts_conn = None
    sms_candidates: list[EdgeBet] = []
    alert_ids_to_notify: list[int] = []

    if alerts_db_path and not args.no_save:
        try:
            alerts_conn = connect_alerts_db(alerts_db_path)
            log.info("Connected to alerts.db: %s", alerts_db_path)
        except Exception as exc:
            log.warning("alerts.db unavailable (%s) — SMS dedup disabled.", exc)

    if alerts_conn:
        for bet in best_edges:
            result = upsert_bet_alert(
                alerts_conn, bet, today, f"{model_name}_v{model_version}"
            )
            log.debug(
                "alerts.db %s: %s %s (notified=%d)",
                result["action"], bet.player_name, bet.threshold_label, result["notified"],
            )
            if result["notified"] == 0:
                sms_candidates.append(bet)
                if result["id"] is not None:
                    alert_ids_to_notify.append(result["id"])
        log.info(
            "alerts.db: %d/%d best-edge alerts are new (not yet notified today).",
            len(sms_candidates), len(best_edges),
        )
    else:
        sms_candidates = best_edges   # no dedup DB: SMS all best edges

    # -- Notify: only new (not-yet-notified) best lines ----------------------
    if args.no_notify:
        log.info("Notifications skipped (--no-notify).")
    elif not has_notify:
        log.warning("notify.py unavailable — SMS skipped.")
    elif not sms_candidates:
        log.info("No new alerts to send SMS for.")
    else:
        import time as _time
        from notify import send_edge_sms
        for bet in sms_candidates:
            send_edge_sms(bet, config)
            _time.sleep(2)
        if alerts_conn:
            mark_alerts_notified(alerts_conn, alert_ids_to_notify)
        if not args.no_save:
            mark_edges_notified(pg_conn, sms_candidates)

    if alerts_conn:
        alerts_conn.close()

    pg_conn.close()
    log.info("=== Done. %d qualifying threshold%s → %d SMS alert%s. ===",
             len(edges),     "" if len(edges)      == 1 else "s",
             len(best_edges), "" if len(best_edges) == 1 else "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
