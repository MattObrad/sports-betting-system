"""
regrade_wnba_clv.py — Recompute CLV using same-sign closing odds.

The original grader used DISTINCT ON without a sign-matching tiebreaker,
so plus-odds alerts were compared against minus-odds closes (two different
markets sharing one line label in props_snapshots). This gives a spuriously
positive CLV for every plus-odds bet.

This script finds the correct same-cluster closing price (same sign as
alert_odds, most-recent, best-price tie-break) and shows before/after CLV.

Usage:
    python regrade_wnba_clv.py              # dry-run: show table, no writes
    python regrade_wnba_clv.py --write      # also update alerts.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import psycopg2

_DIR       = Path(__file__).resolve().parent
_ALERTS_DB = os.environ.get("ALERTS_DB_PATH") or str(_DIR / "alerts.db")
_VPS_HOST  = os.environ.get("VPS_DB_HOST", "localhost")

# Prefer same-sign close; fall back to most-recent + best-price for any sign.
_CORRECTED_CLOSING_SQL = """
SELECT DISTINCT ON (ps.player_name, ps.line)
    ps.over_odds
FROM props_snapshots ps
JOIN games g ON g.event_id = ps.event_id
WHERE g.event_id       = %s
  AND ps.player_name   = %s
  AND ps.line          = %s
  AND ps.market_type   = 'Player Points'
  AND ps.over_odds     IS NOT NULL
  AND ps.snapshot_time < g.game_time
ORDER BY
    ps.player_name,
    ps.line,
    CASE WHEN SIGN(ps.over_odds) = SIGN(%s::integer) THEN 0 ELSE 1 END,
    ps.snapshot_time DESC,
    ps.over_odds     DESC
"""


def decimal_from_american(odds: int) -> float:
    return odds / 100 + 1 if odds > 0 else -100 / odds + 1


def imp(odds: int) -> float:
    return 1.0 / decimal_from_american(odds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Write corrected CLV values back to alerts.db.")
    args = ap.parse_args()

    db = sqlite3.connect(_ALERTS_DB)
    db.row_factory = sqlite3.Row
    bets = db.execute(
        "SELECT id, player_name, line, odds, implied_prob, game_id, "
        "       closing_odds, closing_implied, clv, clv_beat, result, alert_date "
        "FROM bet_alerts WHERE sport='WNBA' AND graded=1 ORDER BY alert_date, id"
    ).fetchall()

    pg  = psycopg2.connect(host=_VPS_HOST, port=5432,
                           database="picksdb", user="picksuser", password="password")
    cur = pg.cursor()

    HDR = (f"{'id':>3} {'date':10} {'player':20} {'ln':>5} {'alrt':>5} "
           f"{'old_cls':>7} {'new_cls':>7} "
           f"{'old_clv':>8} {'new_clv':>8} {'chg':>7} {'res':4} {'flag'}")
    print(HDR)
    print("-" * len(HDR))

    rows: list[tuple] = []   # (id, new_close, new_impl, new_clv, new_beat, flag)

    for b in bets:
        a_odds    = int(b["odds"])
        a_impl    = float(b["implied_prob"]) if b["implied_prob"] else imp(a_odds)
        old_close = b["closing_odds"]
        old_clv   = b["clv"]

        cur.execute(_CORRECTED_CLOSING_SQL,
                    (b["game_id"], b["player_name"], float(b["line"]), a_odds))
        rec = cur.fetchone()
        new_close = int(rec[0]) if rec else None

        if new_close is not None:
            new_impl = imp(new_close)
            new_clv  = round(new_impl - a_impl, 6)
            new_beat = 1 if new_clv > 0 else 0
        else:
            new_impl = new_clv = None
            new_beat = None

        chg = ((new_clv - old_clv) if (new_clv is not None and old_clv is not None)
               else None)

        if old_close is None or new_close is None:
            flag = "noclose"
        elif (old_close > 0) == (a_odds > 0):
            flag = "ok"         # old close was already same-sign
        else:
            flag = "FIXED"      # corrected cross-cluster error

        def f(v, w):
            return f"{v:+{w}.3f}" if v is not None else " " * w

        print(f"{b['id']:>3} {b['alert_date']:10} {b['player_name'][:20]:20} "
              f"{b['line']:>5.1f} {a_odds:>+5} "
              f"{str(old_close or 'None'):>7} {str(new_close or 'None'):>7} "
              f"{f(old_clv, 8)} {f(new_clv, 8)} {f(chg, 7)} "
              f"{b['result']:4} {flag}")

        rows.append((b["id"], new_close, new_impl, new_clv, new_beat, flag))

    pg.close()

    # --- Summary ---
    total  = len(rows)
    fixed  = sum(1 for r in rows if r[5] == "FIXED")
    ok_ct  = sum(1 for r in rows if r[5] == "ok")
    nc_ct  = sum(1 for r in rows if r[5] == "noclose")

    true_clvs    = [r[3] for r in rows if r[3] is not None]
    pos_clv_ct   = sum(1 for v in true_clvs if v > 0)
    avg_clv      = sum(true_clvs) / len(true_clvs) if true_clvs else None

    wins_pos_clv = sum(1 for b, r in zip(bets, rows)
                       if r[3] is not None and r[3] > 0 and b["result"] == "WIN")
    wins_neg_clv = sum(1 for b, r in zip(bets, rows)
                       if r[3] is not None and r[3] <= 0 and b["result"] == "WIN")

    print()
    print("=== CORRECTED CLV SUMMARY ===")
    print(f"  Bets: {total}  |  sign-FIXED: {fixed}  |  already ok: {ok_ct}  |  no close data: {nc_ct}")
    print(f"  True positive CLV: {pos_clv_ct}/{total}")
    print(f"  Wins w/ pos CLV:  {wins_pos_clv}   Wins w/ neg/zero CLV: {wins_neg_clv}")
    if avg_clv is not None:
        print(f"  Average corrected CLV: {avg_clv:+.4f}  ({avg_clv * 100:+.2f} pp)")
    print()

    if args.write:
        needs = [(r[1], r[2], r[3], r[4], r[0])
                 for r in rows if r[5] == "FIXED" and r[1] is not None]
        if needs:
            db.executemany(
                "UPDATE bet_alerts SET closing_odds=?, closing_implied=?, "
                "clv=?, clv_beat=? WHERE id=?",
                needs,
            )
            db.commit()
            print(f"  Wrote corrected CLV for {len(needs)} rows in alerts.db.")
        else:
            print("  No rows needed correction.")
    else:
        fixable = sum(1 for r in rows if r[5] == "FIXED" and r[1] is not None)
        print(f"  Dry run — {fixable} rows need CLV correction. "
              f"Run with --write to apply.")

    db.close()


if __name__ == "__main__":
    main()
