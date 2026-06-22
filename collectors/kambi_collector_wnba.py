"""
kambi_collector_wnba.py — WNBA-only, high-cadence collector.

Reuses ALL parsing/DB logic from kambi_shared.run_collection — writes to the
same props_snapshots / odds_snapshots / games tables as the core collector.

SELF-GATING: on a day with no WNBA games the listView returns zero events and
collect_sport() returns early, so the expensive per-event props loop never runs
(one cheap listView GET). No schedule/calendar lookup needed.

Cron (VPS clock is UTC; WNBA season runs EDT = UTC-4):
    # 13:00-01:00 ET  ==  17:00-05:00 UTC  (wraps midnight -> two entries)
    */10 17-23 * * *  cd /home/picks && python3 collectors/kambi_collector_wnba.py >> /home/picks/logs/wnba_hf.log 2>&1
    */10 0-5   * * *  cd /home/picks && python3 collectors/kambi_collector_wnba.py >> /home/picks/logs/wnba_hf.log 2>&1

NOTE: run as  python3 collectors/kambi_collector_wnba.py  so that
kambi_shared.py (which lives in collectors/) is on the import path.
"""
from kambi_shared import run_collection

# (display_name, listView path, sport_key) — WNBA only.
SPORT_ENDPOINTS = [
    ("WNBA", "basketball/wnba", "basketball"),
]

if __name__ == "__main__":
    run_collection(SPORT_ENDPOINTS)
