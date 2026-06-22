#!/usr/bin/env python3
"""
backtest_tennis_live_config.py

Retroactive backtest: apply the CURRENT live alert config to ALL predictions
in tennis.db for 2026-06-03 through 2026-06-12.

Current live config (tennis_config_vps.json):
  min_edge           = 0.08  (8%)
  min_model_prob     = 0.70  (70%)
  max_odds_american  = 300
  min_career_matches = 10    (both players)
  extreme_flag block : |alert_edge| > 0.20 → rejected
  overround          >= 1.0  (sub-100% overround = data error)

Goal: "what if this config had been running since June 3?"
The actual live system only used these thresholds from June 9 onward
(prior to that, old 0.55/0.55 thresholds were in effect).

Read-only: does not write to any database, does not touch alerts.db.
Run:  python3 /home/picks/backtest_tennis_live_config.py
"""

from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path

# ── Live config ────────────────────────────────────────────────────────────────
MIN_EDGE           = 0.08
MIN_MODEL_PROB     = 0.70
MAX_ODDS_AMERICAN  = 300
MIN_CAREER_MATCHES = 10

BACKTEST_START = "2026-06-03"
BACKTEST_END   = "2026-06-12"

DB_PATH = Path("/home/picks/tennis.db")


# ── Helpers (mirrors predict_tennis.py) ───────────────────────────────────────

def american_to_decimal(odds: int) -> float:
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


# ── Name matching (verbatim from grade_tennis_predictions.py) ─────────────────

def _norm(name: str) -> str:
    nfd = unicodedata.normalize("NFD", name.lower())
    return "".join(c for c in nfd if not unicodedata.combining(c)).strip()


def _names_match(stored: str, lookup: str) -> bool:
    """
    Handles two real-world mismatches between tennisexplorer-scraped rows
    and Kambi/Sackmann full names:
      1. Hyphens:      "Diana Ioana Simionescu"  <-> "Diana-Ioana Simionescu"
      2. Abbreviated:  "Scott K."  <-> "Katrina Scott"
    """
    def clean(s: str) -> str:
        return _norm(s).replace("-", " ").replace(".", "").strip()

    sc = clean(stored)
    lc = clean(lookup)
    if sc == lc:
        return True

    sp = sc.split()
    lp = lc.split()
    if len(sp) >= 2 and len(sp[-1]) == 1 and len(lp) >= 2:
        abbr_last    = " ".join(sp[:-1])
        abbr_initial = sp[-1]
        full_last    = " ".join(lp[1:])
        full_initial = lp[0][0]
        if abbr_last == full_last and abbr_initial == full_initial:
            return True

    return False


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_matches_played(conn: sqlite3.Connection, player_name: str,
                       stored_elo: float, before_date: str) -> int:
    """
    Look up matches_played from player_elo.

    The predictions table stores Kambi display names; player_elo uses Sackmann
    names.  Priority:
      1. Exact name match (works when Kambi == Sackmann spelling, ~80% of cases)
      2. Stored elo == 1500.0 exactly → player had no history → 0 matches
      3. Elo-value match within ±0.01 (finds players where name spelling differs
         but their Elo value was retrieved correctly at prediction time)
    """
    row = conn.execute(
        """SELECT matches_played FROM player_elo
           WHERE player_name = ? AND match_date < ?
           ORDER BY match_date DESC, id DESC LIMIT 1""",
        (player_name, before_date),
    ).fetchone()
    if row:
        return int(row[0])

    # Starting Elo: player wasn't in DB (no prior matches found)
    if abs(stored_elo - 1500.0) < 0.001:
        return 0

    # Elo-value lookup: real players with unique non-default Elo
    row = conn.execute(
        """SELECT matches_played FROM player_elo
           WHERE ABS(overall_elo - ?) < 0.01
             AND match_date < ?
           ORDER BY ABS(overall_elo - ?), match_date DESC LIMIT 1""",
        (stored_elo, before_date, stored_elo),
    ).fetchone()
    return int(row[0]) if row else 0


def find_result(conn: sqlite3.Connection, p1_name: str, p2_name: str,
                pred_date: str) -> int | None:
    """
    Return 1 if p1 won, 0 if p1 lost, None if result not in matches.
    Same -1/+2 day window and _names_match logic as the production grader.
    """
    rows = conn.execute(
        """SELECT winner_name, loser_name FROM matches
           WHERE tourney_date BETWEEN date(?, '-1 day') AND date(?, '+2 days')""",
        (pred_date, pred_date),
    ).fetchall()
    for winner, loser in rows:
        if _names_match(winner, p1_name) and _names_match(loser, p2_name):
            return 1
        if _names_match(loser, p1_name) and _names_match(winner, p2_name):
            return 0
    return None


def profit_units(won: bool, odds: int) -> float:
    """$1 flat stake.  WIN: decimal - 1.0.  LOSS: -1.0."""
    dec = american_to_decimal(odds)
    return (dec - 1.0) if won else -1.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")

    max_match = conn.execute("SELECT MAX(tourney_date) FROM matches").fetchone()[0]

    # Pull all predictions in window, ordered so later writes overwrite earlier
    rows = conn.execute(
        """SELECT id, prediction_date, event_id,
                  player1_name, player2_name,
                  player1_overall_elo, player2_overall_elo,
                  player1_model_prob, player2_model_prob,
                  player1_kambi_odds, player2_kambi_odds,
                  player1_fair_prob, player2_fair_prob,
                  player1_edge, player2_edge
           FROM predictions
           WHERE prediction_date BETWEEN ? AND ?
           ORDER BY created_at ASC, id ASC""",
        (BACKTEST_START, BACKTEST_END),
    ).fetchall()

    # Dedup by event_id (last write wins — same as grade_tennis_predictions.py)
    latest: dict[str, tuple] = {}
    for r in rows:
        latest[r[2]] = r
    rows = list(latest.values())
    total_preds = len(rows)

    W = 74
    print("=" * W)
    print("  TENNIS BACKTEST: Current Live Config Applied Retroactively")
    print(f"  Period: {BACKTEST_START} .. {BACKTEST_END}")
    print("=" * W)
    print(f"  Config:  min_edge={MIN_EDGE:.0%}  min_prob={MIN_MODEL_PROB:.0%}  "
          f"max_odds={MAX_ODDS_AMERICAN:+d}  min_career={MIN_CAREER_MATCHES}")
    print(f"  matches latest tourney_date: {max_match}")
    print(f"  Total predictions (unique events): {total_preds}")
    print()

    # ── Apply gates in the same order as predict_tennis.qualifies() ───────────
    gate_count = {
        "edge":      0,   # alert_edge < 0.08
        "prob":      0,   # alert_prob < 0.70
        "extreme":   0,   # |alert_edge| > 0.20
        "max_odds":  0,   # |alert_odds| > 300
        "overround": 0,   # overround < 1.0
        "ev":        0,   # alert_ev <= 0
        "history":   0,   # either player < 10 career matches
    }
    null_skip  = 0
    qualifying: list[dict] = []

    for r in rows:
        (row_id, pred_date, event_id,
         p1_name, p2_name,
         p1_elo_val, p2_elo_val,
         p1_model_prob, p2_model_prob,
         p1_odds, p2_odds,
         p1_fair, p2_fair,
         p1_edge, p2_edge) = r

        if None in (p1_model_prob, p2_model_prob, p1_odds, p2_odds,
                    p1_fair, p2_fair, p1_edge, p2_edge):
            null_skip += 1
            continue

        p1_edge = float(p1_edge)
        p2_edge = float(p2_edge)

        # Reconstruct alert side (mirrors predict_tennis.predict_match)
        if p1_edge >= p2_edge:
            alert_player  = p1_name
            alert_odds    = int(p1_odds)
            alert_prob    = float(p1_model_prob)
            alert_fair    = float(p1_fair)
            alert_edge    = p1_edge
            alert_elo_val = float(p1_elo_val) if p1_elo_val is not None else 1500.0
            opp_name      = p2_name
            opp_elo_val   = float(p2_elo_val) if p2_elo_val is not None else 1500.0
            p1_is_alert   = True
        else:
            alert_player  = p2_name
            alert_odds    = int(p2_odds)
            alert_prob    = float(p2_model_prob)
            alert_fair    = float(p2_fair)
            alert_edge    = p2_edge
            alert_elo_val = float(p2_elo_val) if p2_elo_val is not None else 1500.0
            opp_name      = p1_name
            opp_elo_val   = float(p1_elo_val) if p1_elo_val is not None else 1500.0
            p1_is_alert   = False

        # Gate 1: minimum edge
        if alert_edge < MIN_EDGE:
            gate_count["edge"] += 1
            continue

        # Gate 2: minimum model probability
        if alert_prob < MIN_MODEL_PROB:
            gate_count["prob"] += 1
            continue

        # Gate 3: extreme flag (|edge| > 20% = model error, not real signal)
        if abs(alert_edge) > 0.20:
            gate_count["extreme"] += 1
            continue

        # Gate 4: maximum odds
        if abs(alert_odds) > MAX_ODDS_AMERICAN:
            gate_count["max_odds"] += 1
            continue

        # Gate 5: overround sanity check
        d1 = american_to_decimal(int(p1_odds))
        d2 = american_to_decimal(int(p2_odds))
        overround = 1.0 / d1 + 1.0 / d2
        if overround < 1.0:
            gate_count["overround"] += 1
            continue

        # Gate 6: positive EV
        alert_dec = american_to_decimal(alert_odds)
        alert_ev  = alert_prob * alert_dec - 1.0
        if alert_ev <= 0.0:
            gate_count["ev"] += 1
            continue

        # Gate 7: minimum career matches for both players
        alert_n = get_matches_played(conn, alert_player, alert_elo_val, pred_date)
        opp_n   = get_matches_played(conn, opp_name, opp_elo_val, pred_date)
        if alert_n < MIN_CAREER_MATCHES or opp_n < MIN_CAREER_MATCHES:
            gate_count["history"] += 1
            continue

        qualifying.append({
            "pred_date":    pred_date,
            "event_id":     event_id,
            "alert_player": alert_player,
            "opp_name":     opp_name,
            "p1_name":      p1_name,
            "p2_name":      p2_name,
            "p1_is_alert":  p1_is_alert,
            "alert_odds":   alert_odds,
            "alert_prob":   alert_prob,
            "alert_fair":   alert_fair,
            "alert_edge":   alert_edge,
            "alert_ev":     alert_ev,
            "overround":    overround,
            "alert_n":      alert_n,
            "opp_n":        opp_n,
        })

    # Gate summary
    passed = len(qualifying)
    print(f"  Gate analysis (first-failing-gate counts):")
    print(f"    {'Passed all gates:':<28} {passed}")
    if null_skip:
        print(f"    {'Null/incomplete rows:':<28} {null_skip}")
    for gate, cnt in gate_count.items():
        if cnt:
            pct = cnt / total_preds * 100
            print(f"    {gate + ':':<28} {cnt:>4}  ({pct:.1f}% of predictions)")
    print()

    # ── Grade qualifying bets ──────────────────────────────────────────────────
    graded:  list[dict] = []
    pending: list[dict] = []

    for bet in qualifying:
        y1 = find_result(conn, bet["p1_name"], bet["p2_name"], bet["pred_date"])
        if y1 is None:
            pending.append(bet)
            continue
        won = (y1 == 1) if bet["p1_is_alert"] else (y1 == 0)
        graded.append({
            **bet,
            "won": won,
            "pnl": profit_units(won, bet["alert_odds"]),
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * W)
    print("  RESULTS SUMMARY")
    print("=" * W)
    print(f"  Qualifying bets:          {len(qualifying)}")
    print(f"  Graded (result found):    {len(graded)}")
    print(f"  Pending (no result yet):  {len(pending)}")

    if graded:
        n          = len(graded)
        wins       = sum(1 for g in graded if g["won"])
        losses     = n - wins
        total_pnl  = sum(g["pnl"] for g in graded)
        roi        = total_pnl / n * 100
        avg_odds   = sum(g["alert_odds"] for g in graded) / n
        avg_prob   = sum(g["alert_prob"] for g in graded) / n
        avg_edge   = sum(g["alert_edge"] for g in graded) / n
        avg_ev     = sum(g["alert_ev"] for g in graded) / n

        print()
        print(f"  W-L record:    {wins}-{losses}  ({wins/n*100:.1f}% win rate)")
        print(f"  Total P&L:     {total_pnl:+.3f} units  ($1 flat stake)")
        print(f"  ROI:           {roi:+.1f}%")
        print(f"  Avg odds:      {avg_odds:+.0f}")
        print(f"  Avg model prob:{avg_prob:.1%}")
        print(f"  Avg edge:      {avg_edge:.1%}")
        print(f"  Avg EV:        {avg_ev:.1%}")

        # Per-bet detail table
        print()
        print(f"  {'Date':<12} {'Alert Side':<28} {'Opponent':<25} "
              f"{'Odds':>6} {'Prob':>6} {'Edge':>6} {'EV':>5} {'N/Opp':>7} {'Res':>4} {'P&L':>7}")
        print("  " + "-" * 109)
        for g in sorted(graded, key=lambda x: x["pred_date"]):
            res = "WIN " if g["won"] else "LOSS"
            n_str = f"{g['alert_n']}/{g['opp_n']}"
            print(f"  {g['pred_date']:<12} {g['alert_player']:<28} {g['opp_name']:<25} "
                  f"{g['alert_odds']:>+6d} {g['alert_prob']*100:>5.1f}% "
                  f"{g['alert_edge']*100:>5.1f}% {g['alert_ev']*100:>+4.0f}% "
                  f"{n_str:>7} {res:>4} {g['pnl']:>+7.3f}")

        if pending:
            print()
            print(f"  Pending bets (result not yet in matches):")
            print(f"  {'Date':<12} {'Alert Side':<28} {'Opponent':<25} "
                  f"{'Odds':>6} {'Prob':>6} {'Edge':>6}")
            print("  " + "-" * 82)
            for g in sorted(pending, key=lambda x: x["pred_date"]):
                print(f"  {g['pred_date']:<12} {g['alert_player']:<28} {g['opp_name']:<25} "
                      f"{g['alert_odds']:>+6d} {g['alert_prob']*100:>5.1f}% "
                      f"{g['alert_edge']*100:>5.1f}%")

    # ── Calibration (qualifying bets only) ────────────────────────────────────
    if graded:
        _BINS = [
            ("50-60%", 0.50, 0.60),
            ("60-70%", 0.60, 0.70),
            ("70-80%", 0.70, 0.80),
            ("80-90%", 0.80, 0.90),
            ("90%+",   0.90, 1.01),
        ]
        print()
        print("=" * W)
        print("  CALIBRATION (qualifying bets only, model-prob perspective)")
        print("=" * W)
        print(f"  {'bucket':<8} {'n':>4} {'avg_pred':>9} {'actual%':>8} "
              f"{'units':>8} {'ROI%':>7}")
        print("  " + "-" * 53)
        for label, lo, hi in _BINS:
            sub = [g for g in graded if lo <= g["alert_prob"] < hi]
            if not sub:
                print(f"  {label:<8} {'0':>4} {'--':>9} {'--':>8} {'--':>8} {'--':>7}")
                continue
            ns   = len(sub)
            avgp = sum(g["alert_prob"] for g in sub) / ns * 100
            actp = sum(1 for g in sub if g["won"]) / ns * 100
            u    = sum(g["pnl"] for g in sub)
            r    = u / ns * 100
            print(f"  {label:<8} {ns:>4} {avgp:>8.1f}% {actp:>7.1f}% {u:>+8.3f} {r:>+7.1f}%")
        print("  " + "-" * 53)
        tot_avg = sum(g["alert_prob"] for g in graded) / n * 100
        tot_act = wins / n * 100
        print(f"  {'TOTAL':<8} {n:>4} {tot_avg:>8.1f}% {tot_act:>7.1f}% "
              f"{total_pnl:>+8.3f} {roi:>+7.1f}%")

    # ── Sanity check: 4 alerts that actually fired June 9-12 ──────────────────
    print()
    print("=" * W)
    print("  SANITY CHECK: Alerts That Actually Fired June 9-12")
    print("  (All should appear in the qualifying list above)")
    print("=" * W)

    actual_alerts = conn.execute(
        """SELECT DATE(created_at), player_name, opponent_name,
                  odds, model_prob, edge, result
           FROM alerts
           WHERE DATE(created_at) BETWEEN '2026-06-09' AND '2026-06-12'
           ORDER BY created_at""",
    ).fetchall()

    if not actual_alerts:
        print("  (no rows in tennis.db alerts table for June 9-12)")
    else:
        for fired_date, player, opp, odds, prob, edge, db_result in actual_alerts:
            # Check if this alert is in the qualifying list
            in_bt = any(
                _names_match(player, q["alert_player"]) and q["pred_date"] == fired_date
                for q in qualifying
            )
            # Find graded result for this specific bet
            gr = next(
                (g for g in graded
                 if _names_match(player, g["alert_player"]) and g["pred_date"] == fired_date),
                None,
            )
            if gr:
                outcome = f"{'WIN' if gr['won'] else 'LOSS'}  {gr['pnl']:+.3f}u"
            else:
                pend = next(
                    (p for p in pending
                     if _names_match(player, p["alert_player"]) and p["pred_date"] == fired_date),
                    None,
                )
                outcome = "PENDING" if pend else "no match in graded/pending"

            status = "FOUND  " if in_bt else "MISSING"
            prob_pct = f"{float(prob)*100:.1f}%" if prob is not None else "?"
            edge_pct = f"{float(edge)*100:.1f}%" if edge is not None else "?"
            print(f"  [{status}]  {fired_date}  {player[:28]:<28}  vs {opp[:22]:<22}"
                  f"  odds={int(odds) if odds else '?':+d}  edge={edge_pct}"
                  f"  DB={db_result}  backtest={outcome}")

    conn.close()
    print()
    print("  Done.")


if __name__ == "__main__":
    main()
