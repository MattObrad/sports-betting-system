"""
kambi_collector_wnba.py — WNBA-only, high-cadence collector.

PURPOSE: instrumentation week for the Kambi-latency go/no-go study
(see D:\\models\\wnba\\CONCLUSIONS.md §7/§9 and SPEC_wnba_collector_cadence.md).
Our normal hourly cadence censors any repricing latency under ~60 min; this
WNBA-only collector is meant to run every ~10 min during the game-day window so
the latency study can resolve sub-hour moves.

Reuses ALL parsing/DB logic from kambi_shared.run_collection — writes to the
same props_snapshots / odds_snapshots / games tables as the core collector.

SELF-GATING: on a day with no WNBA games the listView returns zero events and
collect_sport() returns early, so the expensive per-event props loop never runs
(one cheap listView GET). No schedule/calendar lookup needed.

Cron (verify VPS clock with `timedatectl` first; season = EDT = UTC-4):
    # 11:00-23:00 ET  ==  15:00-03:59 UTC  (wraps midnight -> two entries)
    */10 15-23 * * *  cd /home/picks && /usr/bin/python3 kambi_collector_wnba.py >> /home/picks/logs/wnba_collector.log 2>&1
    */10 0-3   * * *  cd /home/picks && /usr/bin/python3 kambi_collector_wnba.py >> /home/picks/logs/wnba_collector.log 2>&1
    # If the VPS is already on America/New_York, use instead:  */10 11-23 * * *

Leave kambi_collector_core.py and its schedule UNCHANGED — this is additive.
"""
from kambi_shared import run_collection

# (display_name, listView path, sport_key) — WNBA only.
SPORT_ENDPOINTS = [
    ("WNBA", "basketball/wnba", "basketball"),
]

if __name__ == "__main__":
    run_collection(SPORT_ENDPOINTS)
