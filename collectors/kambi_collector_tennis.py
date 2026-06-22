"""
kambi_collector_tennis.py — ITF Women high-cadence snapshot collector.

Polls ITF Women match-winner lines every 10 min during the match window
so CLV can be measured (opening → closing line movement before start).
Writes to the same props_snapshots / games tables as the other collectors.

SELF-GATING: on days with no ITF Women events, the listView returns zero
events and run_collection returns immediately after one cheap GET.

ITF Women matches typically start 10:00–18:00 UTC.
Window: 07:00–20:00 UTC (2 h buffer on each side).

Cron (add to root crontab on VPS — runs 07:00–20:59 UTC every 10 min):

    */10 7-20 * * *  cd /home/picks && python3 collectors/kambi_collector_tennis.py >> /home/picks/logs/tennis_hf.log 2>&1

Also collect Challenger and ITF Men for completeness (same window is fine;
they are sparse and the collector self-gates on empty days).
"""
from kambi_shared import run_collection

SPORT_ENDPOINTS = [
    ("ITF Women", "tennis/itf_women", "tennis"),
    ("ITF Men",   "tennis/itf_men",   "tennis"),
    ("Challengr", "tennis/challenger", "tennis"),
]

if __name__ == "__main__":
    run_collection(SPORT_ENDPOINTS)
