"""
daily_results.py -- Verification layer and daily results for MLB totals bets.

Runs at 11pm ET via cron after games finish.  Looks up today's edge bets,
fetches actual final totals and per-game closing lines, and produces:
  - WIN / LOSS / PUSH / PENDING result per bet
  - Closing line value (CLV) per bet
  - Rolling performance metrics (7-day, 30-day, all-time)
  - One consolidated results SMS (via notify.py)
  - Upsert to daily_results SQLite table

PENDING games (status != 'final' at 11pm ET) are excluded from all
performance calculations.  Rerun manually with --date the following
morning after suspended/postponed games are resolved.

Usage:
    python daily_results.py                        # tonight's results (ET)
    python daily_results.py --date 2024-09-15      # backfill / retry a date
    python daily_results.py --no-notify            # save results, skip SMS
    python daily_results.py --no-save              # print only
    python daily_results.py --performance          # rolling stats, no SMS
"""

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import pandas as pd

from models.monte_carlo import american_to_payout
from utils.config import cfg_get, load_config

try:
    from notify import send_results_discord
    HAS_NOTIFY = True
except ImportError:
    HAS_NOTIFY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS daily_results (
    run_date          TEXT    NOT NULL,
    game_id           TEXT    NOT NULL,
    game_label        TEXT,               -- 'BOS @ NYY'
    bet_direction     TEXT,
    market_line       REAL,
    closing_line      REAL,
    actual_total      REAL,
    home_score        INTEGER,
    away_score        INTEGER,
    result            TEXT,               -- 'WIN' | 'LOSS' | 'PUSH' | 'PENDING'
    juice             INTEGER,
    kelly_half        REAL,
    recommended_units REAL,
    profit_units      REAL,
    clv               REAL,
    clv_beat          INTEGER,            -- 1 | 0 | NULL
    predicted_total   REAL,              -- ensemble predicted total runs
    PRIMARY KEY (run_date, game_id)
);
"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DailyResult:
    run_date:          str
    game_id:           str
    game_label:        str            # 'BOS @ NYY'
    bet_direction:     str            # 'OVER' | 'UNDER'
    market_line:       float
    closing_line:      float          # None stored as 0.0 when missing
    actual_total:      float          # None stored as 0.0 when PENDING
    home_score:        int
    away_score:        int
    result:            str            # 'WIN' | 'LOSS' | 'PUSH' | 'PENDING'
    juice:             int
    kelly_half:        float
    recommended_units: float
    profit_units:      float          # 0.0 when PENDING
    clv:               float          # 0.0 when unavailable
    clv_beat:          int            # 1 | 0 | -1 (unknown)
    predicted_total:   float = 0.0   # from edge_bets.predicted_total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_todays_bets(con: sqlite3.Connection, run_date: str) -> pd.DataFrame:
    """Load edge_bets placed on run_date."""
    return pd.read_sql_query(
        "SELECT * FROM edge_bets WHERE run_date = ?",
        con,
        params=(run_date,),
    )


def get_game_info(con: sqlite3.Connection, today: str) -> dict:
    """
    Return a dict keyed by game_id with actual scores, status, start time,
    and team abbreviations.
    """
    df = pd.read_sql_query(
        """
        SELECT  g.game_id,
                g.total_runs,
                g.home_score,
                g.away_score,
                g.status,
                g.game_time_utc,
                ht.team_code  AS home_code,
                at2.team_code AS away_code
        FROM    games g
        LEFT JOIN teams ht  ON g.home_team_id = ht.team_id
        LEFT JOIN teams at2 ON g.away_team_id = at2.team_id
        WHERE   g.game_date = ?
        """,
        con,
        params=(today,),
    )
    # row._asdict() was removed in newer pandas; Series.to_dict() is the replacement
    return {str(int(float(row.game_id))): row.to_dict() for _, row in df.iterrows()}


def get_closing_lines(con: sqlite3.Connection, today: str) -> dict:
    """
    Return a dict keyed by game_id with the closing line (last odds_snapshot
    before each game's individual game_time_utc).

    Uses the (game_id, snapshot_time) index for efficiency.
    """
    df = pd.read_sql_query(
        """
        SELECT  os.game_id,
                os.total_line  AS closing_line
        FROM    odds_snapshots os
        JOIN    games g ON os.game_id = g.game_id
        WHERE   g.game_date = ?
          AND   g.game_time_utc IS NOT NULL
          AND   os.snapshot_time = (
                    SELECT MAX(os2.snapshot_time)
                    FROM   odds_snapshots os2
                    WHERE  os2.game_id      = g.game_id
                      AND  os2.snapshot_time < g.game_time_utc
                )
        """,
        con,
        params=(today,),
    )
    return {str(int(float(row.game_id))): float(row.closing_line) for _, row in df.iterrows()
            if row.closing_line is not None}


# ---------------------------------------------------------------------------
# Per-bet calculations
# ---------------------------------------------------------------------------

def compute_result(
    direction: str,
    market_line: float,
    actual_total: float | None,
    juice: int,
    recommended_units: float,
    status: str,
) -> tuple:
    """
    Return (result, profit_units).
    PENDING when status != 'final' or actual_total is None.
    """
    if (status or "").lower() != "final" or actual_total is None:
        return "PENDING", 0.0

    if direction == "OVER":
        if actual_total > market_line:
            result = "WIN"
        elif actual_total < market_line:
            result = "LOSS"
        else:
            result = "PUSH"
    else:  # UNDER
        if actual_total < market_line:
            result = "WIN"
        elif actual_total > market_line:
            result = "LOSS"
        else:
            result = "PUSH"

    payout = american_to_payout(juice)
    if result == "WIN":
        profit = recommended_units * payout
    elif result == "LOSS":
        profit = -recommended_units
    else:
        profit = 0.0

    return result, round(profit, 4)


def compute_clv(
    direction: str,
    market_line: float,
    closing_line: float | None,
) -> tuple:
    """
    Return (clv, clv_beat) or (0.0, -1) when closing line unavailable.

    OVER bet:  CLV = closing_line - market_line
               positive = we got an easier number than the close (good)
    UNDER bet: CLV = market_line - closing_line
               positive = we got a lower number than the close (good)
    """
    if closing_line is None:
        return 0.0, -1

    if direction == "OVER":
        clv = closing_line - market_line
    else:
        clv = market_line - closing_line

    clv = round(clv, 3)
    return clv, (1 if clv > 0 else 0)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_daily_results(
    bets_df: pd.DataFrame,
    game_info: dict,
    closing_lines: dict,
    run_date: str,
) -> list:
    """Assemble one DailyResult per edge bet."""
    results = []
    for _, row in bets_df.iterrows():
        gid = str(int(float(row.game_id)))
        info = game_info.get(gid, {})

        status       = info.get("status", "unknown")
        actual_total = info.get("total_runs")
        # pd.notna guard: NaN from NULL DB columns is truthy, so `or 0` keeps NaN
        # and int(NaN) crashes. Use explicit notna check instead.
        _hs = info.get("home_score"); home_score = int(_hs) if pd.notna(_hs) else 0
        _as = info.get("away_score"); away_score = int(_as) if pd.notna(_as) else 0
        home_code    = info.get("home_code") or "HOME"
        away_code    = info.get("away_code") or "AWAY"
        game_label   = f"{away_code} @ {home_code}"
        closing_line = closing_lines.get(gid)

        result, profit = compute_result(
            direction        = row.bet_direction,
            market_line      = float(row.market_line),
            actual_total     = float(actual_total) if actual_total is not None else None,
            juice            = int(row.juice),
            recommended_units= float(row.recommended_units),
            status           = status,
        )

        clv, clv_beat = compute_clv(
            direction    = row.bet_direction,
            market_line  = float(row.market_line),
            closing_line = closing_line,
        )

        pred_total = float(row.predicted_total) if pd.notna(getattr(row, "predicted_total", None)) else 0.0
        results.append(DailyResult(
            run_date          = run_date,
            game_id           = gid,
            game_label        = game_label,
            bet_direction     = row.bet_direction,
            market_line       = float(row.market_line),
            closing_line      = closing_line or 0.0,
            actual_total      = float(actual_total) if actual_total is not None else 0.0,
            home_score        = int(home_score),
            away_score        = int(away_score),
            result            = result,
            juice             = int(row.juice),
            kelly_half        = float(row.kelly_half),
            recommended_units = float(row.recommended_units),
            profit_units      = profit,
            clv               = clv,
            clv_beat          = clv_beat,
            predicted_total   = pred_total,
        ))

    return results


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_performance(con: sqlite3.Connection, window_days: int | None) -> dict:
    """
    Compute rolling performance metrics from daily_results.
    window_days=None means all-time.
    """
    if window_days is not None:
        date_floor = pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days)
        date_str   = date_floor.strftime("%Y-%m-%d")
        df = pd.read_sql_query(
            "SELECT * FROM daily_results WHERE run_date >= ? AND result != 'PENDING'",
            con, params=(date_str,),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM daily_results WHERE result != 'PENDING'",
            con,
        )

    if df.empty:
        return {"n_bets": 0}

    resolved = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]
    n_bets   = len(resolved)
    n_wins   = (resolved["result"] == "WIN").sum()
    n_losses = (resolved["result"] == "LOSS").sum()
    n_pushes = (resolved["result"] == "PUSH").sum()

    wagered = resolved["recommended_units"].sum()
    profit  = resolved["profit_units"].sum()
    roi     = (profit / wagered * 100) if wagered > 0 else 0.0

    has_clv    = resolved["clv_beat"] >= 0
    clv_bets   = resolved[has_clv]
    clv_rate   = (clv_bets["clv_beat"] == 1).sum() / len(clv_bets) if len(clv_bets) > 0 else None
    avg_clv    = clv_bets["clv"].mean() if len(clv_bets) > 0 else None

    return {
        "n_bets":    int(n_bets),
        "n_wins":    int(n_wins),
        "n_losses":  int(n_losses),
        "n_pushes":  int(n_pushes),
        "win_rate":  round(n_wins / (n_wins + n_losses), 4) if (n_wins + n_losses) > 0 else 0.0,
        "wagered":   round(float(wagered), 2),
        "profit":    round(float(profit),  2),
        "roi_pct":   round(float(roi),     2),
        "clv_rate":  round(float(clv_rate), 4) if clv_rate is not None else None,
        "avg_clv":   round(float(avg_clv),  3) if avg_clv  is not None else None,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_daily_report(
    results: list,
    perf_windows: dict,
    run_date: str,
) -> None:
    """Print bet-by-bet results table and rolling performance summary."""
    resolved = [r for r in results if r.result != "PENDING"]
    pending  = [r for r in results if r.result == "PENDING"]

    bar = "-" * 80
    print(f"\n=== Daily Results -- {run_date} ===")

    if not results:
        print("  No bets placed today.\n")
    else:
        print(bar)
        print(f"  {'Game':<18} {'Dir':<5} {'Line':>5} {'Close':>6} "
              f"{'Actual':>7} {'Result':<7} {'Profit':>7}  CLV")
        print(bar)
        for r in results:
            close_str  = f"{r.closing_line:.1f}" if r.closing_line  else "  n/a"
            actual_str = f"{r.actual_total:.0f} ({r.away_score}-{r.home_score})" \
                         if r.result != "PENDING" else "  ---"
            clv_str    = f"{r.clv:+.2f}" if r.clv_beat >= 0 else "  n/a"
            profit_str = f"{r.profit_units:+.2f}u" if r.result != "PENDING" else ""
            print(
                f"  {r.game_label:<18} {r.bet_direction:<5} "
                f"{r.market_line:>5.1f} {close_str:>6} "
                f"{actual_str:>7} {r.result:<7} {profit_str:>7}  {clv_str}"
            )
        print(bar)

        net = sum(r.profit_units for r in resolved)
        print(f"  {len(resolved)} resolved | {len(pending)} pending | Net: {net:+.2f}u")

    # Performance tables
    labels = [("7-day", 7), ("30-day", 30), ("All-time", None)]
    print(f"\n{'Window':<10} {'Bets':>5} {'W-L-P':>10} {'WR%':>6} "
          f"{'ROI%':>7} {'CLV%':>6} {'AvgCLV':>8}")
    print("-" * 60)
    for label, days in labels:
        p = perf_windows.get(label, {})
        if not p.get("n_bets"):
            print(f"  {label:<10} {'--':>5}")
            continue
        wlp       = f"{p['n_wins']}-{p['n_losses']}-{p['n_pushes']}"
        wr        = f"{p['win_rate']*100:.1f}%" if p["n_bets"] else "--"
        roi       = f"{p['roi_pct']:+.1f}%"
        clv_rate  = f"{p['clv_rate']*100:.1f}%" if p.get("clv_rate") is not None else "--"
        avg_clv   = f"{p['avg_clv']:+.3f}" if p.get("avg_clv") is not None else "--"
        print(f"  {label:<10} {p['n_bets']:>5} {wlp:>10} {wr:>6} "
              f"{roi:>7} {clv_rate:>6} {avg_clv:>8}")
    print()


def _build_results_sms(results: list, perf_30d: dict, run_date: str) -> str:
    """Build the consolidated results SMS body."""
    resolved = [r for r in results if r.result != "PENDING"]
    lines    = [f"MLB Results {run_date}"]

    for r in resolved:
        score_str  = f"({r.away_score}-{r.home_score})" if r.result != "PENDING" else ""
        profit_str = f"{r.profit_units:+.2f}u"
        lines.append(
            f"{r.game_label} {r.bet_direction} {r.market_line}: "
            f"{r.result} {score_str} {profit_str}"
        )

    net = sum(r.profit_units for r in resolved)
    roi = perf_30d.get("roi_pct")
    clv = perf_30d.get("clv_rate")

    summary_parts = [f"Net: {net:+.2f}u"]
    if roi is not None:
        summary_parts.append(f"30d ROI: {roi:+.1f}%")
    if clv is not None:
        summary_parts.append(f"CLV: {clv*100:.0f}%")
    lines.append(" | ".join(summary_parts))

    return "\n".join(lines)


def _send_no_results_discord(run_date: str, n_pending: int = 0) -> None:
    """Post a brief 'no results today' embed to DISCORD_WEBHOOK_RESULTS."""
    import json, urllib.request as _ur
    webhook = os.environ.get("DISCORD_WEBHOOK_RESULTS", "").strip()
    if not webhook:
        log.info("DISCORD_WEBHOOK_RESULTS not set — no-results notice skipped.")
        return
    if n_pending:
        desc = f"**{n_pending}** bet{'s' if n_pending != 1 else ''} pending — results after games finish."
    else:
        desc = "No MLB edge bets placed today."
    payload = {
        "embeds": [{
            "title":       f"⚾ MLB Results — {run_date}",
            "description": desc,
            "color":       9807270,   # grey
            "footer":      {"text": "ObServatory MLB Model"},
        }]
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = _ur.Request(webhook, data=data,
                           headers={"Content-Type": "application/json",
                                    "User-Agent": "ObServatory/1.0"},
                           method="POST")
        with _ur.urlopen(req, timeout=10) as r:
            if r.status in (200, 204):
                log.info("Discord no-results notice sent for %s.", run_date)
            else:
                log.warning("Discord returned HTTP %d for no-results notice.", r.status)
    except Exception as exc:
        log.error("Discord no-results notice failed: %s", exc)


def send_results_sms(
    results: list,
    perf_30d: dict,
    cfg: dict,
    run_date: str,
) -> None:
    """Post a daily results embed to Discord DISCORD_WEBHOOK_RESULTS."""
    if not HAS_NOTIFY:
        log.warning("notify.py unavailable -- results notification skipped.")
        return

    resolved = [r for r in results if r.result != "PENDING"]
    if not resolved:
        log.info("No resolved bets -- results notification not sent.")
        return

    try:
        send_results_discord(results, perf_30d, run_date)
    except Exception as exc:
        log.error("Results Discord notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_daily_results(
    con: sqlite3.Connection,
    results: list,
    run_date: str,
) -> None:
    # Schema validation: compare actual columns to expected DDL. If any
    # required column is missing, drop and recreate (all PENDING rows have no
    # value — safe to lose). CREATE TABLE IF NOT EXISTS is a no-op on old tables.
    required = {"run_date", "game_id", "game_label", "bet_direction", "market_line",
                "closing_line", "actual_total", "home_score", "away_score",
                "result", "juice", "kelly_half", "recommended_units",
                "profit_units", "clv", "clv_beat", "predicted_total"}
    try:
        existing = {row[1] for row in con.execute("PRAGMA table_info(daily_results)")}
        if existing and not required.issubset(existing):
            log.warning("daily_results schema outdated (%d/%d cols). Recreating.",
                        len(existing), len(required))
            con.execute("DROP TABLE IF EXISTS daily_results")
    except Exception:
        pass
    con.execute(_DDL)
    rows = []
    for r in results:
        d = asdict(r)
        # Store None-sentinel values as SQL NULL where appropriate
        d["closing_line"]  = d["closing_line"]  or None
        d["actual_total"]  = d["actual_total"]  if d["result"] != "PENDING" else None
        d["profit_units"]  = d["profit_units"]  if d["result"] != "PENDING" else None
        d["clv"]            = d["clv"]           if d["clv_beat"] >= 0 else None
        d["clv_beat"]       = d["clv_beat"]      if d["clv_beat"] >= 0 else None
        d["predicted_total"]= d["predicted_total"] if d["predicted_total"] else None
        rows.append(d)

    con.executemany(
        """
        INSERT OR REPLACE INTO daily_results
            (run_date, game_id, game_label, bet_direction, market_line,
             closing_line, actual_total, home_score, away_score,
             result, juice, kelly_half, recommended_units,
             profit_units, clv, clv_beat, predicted_total)
        VALUES
            (:run_date, :game_id, :game_label, :bet_direction, :market_line,
             :closing_line, :actual_total, :home_score, :away_score,
             :result, :juice, :kelly_half, :recommended_units,
             :profit_units, :clv, :clv_beat, :predicted_total)
        """,
        rows,
    )
    con.commit()
    log.info("Saved %d result(s) for %s.", len(results), run_date)


# ---------------------------------------------------------------------------
# alerts.db upsert  (unified bet tracking — sport='MLB')
# ---------------------------------------------------------------------------

def _decimal_from_american(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (-100 / odds + 1)


def upsert_to_alerts_db(
    alerts_db_path: str,
    results: list,
    bets_df: pd.DataFrame,
    run_date: str,
    model_version: str,
) -> None:
    """
    INSERT OR REPLACE each DailyResult into bet_alerts (sport='MLB').

    Called after save_daily_results() so grading is already final.
    notified=1 always: predict_mlb.py already sent the SMS.
    graded=1 for WIN/LOSS/PUSH, graded=0 for PENDING (re-run next morning).

    MLB CLV is stored in run units (closing_line - market_line for OVER),
    not probability units.  clv_beat semantics are the same: 1 = favorable.
    """
    from pathlib import Path as _Path
    if not _Path(alerts_db_path).exists():
        log.warning("alerts.db not found at %s — skipping write.", alerts_db_path)
        return

    # Build lookup dict from bets_df for prob/predicted fields (keyed by game_id)
    extra: dict[str, dict] = {}
    for _, row in bets_df.iterrows():
        gid = str(int(float(row.game_id)))
        extra[gid] = {
            "predicted_value": float(row.predicted_total) if pd.notna(row.predicted_total) else None,
            "model_prob":      float(row.p_win)           if pd.notna(row.p_win)           else None,
            "implied_prob":    float(row.implied_prob)    if pd.notna(row.implied_prob)     else None,
            "edge_prob":       float(row.prob_edge)       if pd.notna(row.prob_edge)        else None,
        }

    now = datetime.now(ET).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for r in results:
        gid = r.game_id
        ex  = extra.get(gid, {})

        edge_prob = ex.get("edge_prob")
        ev = (edge_prob * (_decimal_from_american(r.juice) - 1.0)
              if edge_prob is not None else None)

        graded       = 0 if r.result == "PENDING" else 1
        actual       = r.actual_total if r.result != "PENDING" else None
        profit       = r.profit_units if r.result != "PENDING" else None
        clv_val      = r.clv     if r.clv_beat >= 0 else None
        clv_beat_val = r.clv_beat if r.clv_beat >= 0 else None

        rows.append((
            "MLB",                        # sport
            model_version,                # model_version
            run_date,                     # alert_date
            now,                          # alert_time
            gid,                          # game_id
            None,                         # player_name  (game-level)
            "Total Runs",                 # market_type
            r.bet_direction,              # direction
            r.market_line,                # line
            r.juice,                      # odds
            ex.get("predicted_value"),    # predicted_value
            ex.get("model_prob"),         # model_prob
            ex.get("implied_prob"),       # implied_prob
            edge_prob,                    # edge_prob
            ev,                           # ev
            r.kelly_half,                 # kelly_half
            None,                         # opening_line  (not tracked in DailyResult)
            r.market_line,                # alert_line    (line at alert time)
            None,                         # closing_odds  (MLB CLV is in line units)
            None,                         # closing_implied
            actual,                       # actual_result
            r.result,                     # result
            profit,                       # profit_units
            clv_val,                      # clv  (run units for MLB)
            clv_beat_val,                 # clv_beat
            1,                            # notified  (predict_mlb.py already sent SMS)
            now,                          # notified_at
            graded,                       # graded
            now if graded else None,      # graded_at
            r.game_label,                 # notes
        ))

    try:
        conn = sqlite3.connect(alerts_db_path)
        conn.executemany(
            """
            INSERT OR REPLACE INTO bet_alerts
                (sport, model_version, alert_date, alert_time, game_id, player_name,
                 market_type, direction, line, odds,
                 predicted_value, model_prob, implied_prob, edge_prob, ev,
                 kelly_half, opening_line, alert_line,
                 closing_odds, closing_implied,
                 actual_result, result, profit_units, clv, clv_beat,
                 notified, notified_at, graded, graded_at, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        conn.commit()
        n_graded = sum(1 for r in results if r.result != "PENDING")
        log.info(
            "alerts.db: wrote %d MLB bet%s (%d graded, %d pending) → %s",
            len(rows), "" if len(rows) == 1 else "s",
            n_graded, len(rows) - n_graded,
            alerts_db_path,
        )
        conn.close()
    except Exception as exc:
        log.error("alerts.db write failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify today's edge bets against actual results and update DB."
    )
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Verify a specific date (default: today in ET). Use for backfill / PENDING retry.",
    )
    p.add_argument(
        "--no-notify",
        action="store_true",
        help="Compute and save results, skip results SMS",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Print results only; do not write to DB",
    )
    p.add_argument(
        "--performance",
        action="store_true",
        help="Print rolling performance stats and exit (no SMS, no DB write)",
    )
    p.add_argument("--db",        default="mlb_data.db")
    p.add_argument("--config",    default=None)
    p.add_argument(
        "--alerts-db", default=None, metavar="PATH",
        help="Path to alerts.db for unified tracking "
             "(default: ALERTS_DB_PATH env var, then skip).",
    )
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    cfg  = load_config(args.config)

    # ET date with 4-hour look-back so the cron at 04:00 UTC (= midnight EDT,
    # = 11pm EST) always labels results as the previous evening's game date:
    #   EDT: midnight June 5 EDT  − 4h = 8pm June 4 EDT  → "2026-06-04" ✓
    #   EST: 11pm    June 4 EST   − 4h = 7pm June 4 EST  → "2026-06-04" ✓
    # The --date flag overrides this entirely for manual backfills / retries.
    today = args.date or (datetime.now(ET) - timedelta(hours=4)).date().isoformat()
    log.info("=== Daily results verification for %s ===", today)

    alerts_db = args.alerts_db or os.environ.get("ALERTS_DB_PATH")
    model_ver  = cfg_get(cfg, "model", "version", default="v1")

    con = sqlite3.connect(args.db)

    # -- Discord dedup guard -------------------------------------------------
    # Tracks whether a Discord post has already been sent for `today` so that
    # a manual re-run or race condition never produces duplicate channel messages.
    # Stored in _collector_state under key "results_discord_sent:YYYY-MM-DD".
    # Bypassed when --date is explicitly provided (backfill / retry must re-send).
    _dedup_key   = f"results_discord_sent:{today}"
    _manual_date = args.date is not None   # True  → skip dedup, allow re-send

    def _already_notified() -> bool:
        if _manual_date:
            return False
        return bool(con.execute(
            "SELECT 1 FROM _collector_state WHERE key = ?", (_dedup_key,)
        ).fetchone())

    def _mark_notified() -> None:
        con.execute(
            "INSERT OR REPLACE INTO _collector_state (key, value) VALUES (?, ?)",
            (_dedup_key, "1"),
        )
        con.commit()

    # -- Performance-only mode -----------------------------------------------
    if args.performance:
        perf_windows = {
            "7-day":   compute_performance(con, 7),
            "30-day":  compute_performance(con, 30),
            "All-time": compute_performance(con, None),
        }
        print_daily_report([], perf_windows, today)
        con.close()
        sys.exit(0)

    # -- Load today's bets ---------------------------------------------------
    bets_df = load_todays_bets(con, today)

    if bets_df.empty:
        log.info("No bets placed on %s -- sending no-results notice.", today)
        if not args.no_notify and HAS_NOTIFY:
            if _already_notified():
                log.info("Discord already sent for %s -- skipping duplicate.", today)
            else:
                _send_no_results_discord(today)
                _mark_notified()
        con.close()
        sys.exit(0)

    log.info("Found %d edge bet(s) for %s.", len(bets_df), today)

    # -- Fetch actuals and closing lines ------------------------------------
    game_info    = get_game_info(con, today)
    closing_lines = get_closing_lines(con, today)

    # -- Build results -------------------------------------------------------
    results = build_daily_results(bets_df, game_info, closing_lines, today)

    n_resolved = sum(1 for r in results if r.result != "PENDING")
    n_pending  = sum(1 for r in results if r.result == "PENDING")
    log.info("%d resolved, %d pending.", n_resolved, n_pending)

    # -- Save ----------------------------------------------------------------
    if not args.no_save:
        save_daily_results(con, results, today)
        if alerts_db:
            upsert_to_alerts_db(alerts_db, results, bets_df, today, model_ver)

    # -- Performance windows -------------------------------------------------
    perf_windows = {
        "7-day":    compute_performance(con, 7),
        "30-day":   compute_performance(con, 30),
        "All-time": compute_performance(con, None),
    }

    # -- Report --------------------------------------------------------------
    print_daily_report(results, perf_windows, today)

    # -- Notify --------------------------------------------------------------
    if args.no_notify:
        log.info("Results SMS skipped (--no-notify).")
    elif n_resolved == 0:
        log.info("All %d bet(s) still pending — sending pending notice.", n_pending)
        if HAS_NOTIFY:
            if _already_notified():
                log.info("Discord already sent for %s -- skipping duplicate.", today)
            else:
                _send_no_results_discord(today, n_pending=n_pending)
                _mark_notified()
    else:
        if _already_notified():
            log.info("Discord already sent for %s -- skipping duplicate.", today)
        else:
            send_results_sms(results, perf_windows["30-day"], cfg, today)
            _mark_notified()

    con.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
