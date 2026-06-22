"""
patch_crontab.py — run on VPS to modify crontab:
  REMOVE: predict_wnba.py runs, grade_wnba evening run
  ADD:    tennis afternoon predict + evening grade
"""
import subprocess, sys

# Pull current crontab
result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
lines = result.stdout.splitlines(keepends=True)

REMOVE_TAGS = [
    'predict_wnba.py',
    'grade_wnba_evening.log',
]

kept = []
removed = []
for line in lines:
    if any(tag in line for tag in REMOVE_TAGS):
        removed.append(line.rstrip())
    else:
        kept.append(line)

NEW_TENNIS = [
    '# Tennis afternoon predict (16:00 UTC) — catches matches posted after 11am run\n',
    '0 16 * * *  cd /home/picks && python3 predict_tennis.py --config /home/picks/tennis_config.json >> /home/picks/logs/predict_tennis_afternoon.log 2>&1\n',
    '# Tennis evening grade pass (22:00 UTC) — grades afternoon/evening match results\n',
    '0 22 * * *  cd /home/picks && ALERTS_DB_PATH=/home/picks/alerts.db TENNIS_DB_PATH=/home/picks/tennis.db python3 grade_tennis_results.py >> /home/picks/logs/grade_tennis_evening2.log 2>&1\n',
]

# Insert after the existing grade_tennis_results evening line
insert_at = next((i for i, l in enumerate(kept) if 'grade_tennis_results_evening' in l), len(kept) - 1)
kept = kept[:insert_at + 1] + NEW_TENNIS + kept[insert_at + 1:]

new_crontab = ''.join(kept)

print('=== REMOVED ===')
for r in removed:
    print('  -', r)

print('\n=== ADDED ===')
for n in NEW_TENNIS:
    print('  +', n.rstrip())

# Install
proc = subprocess.run(['crontab', '-'], input=new_crontab, text=True, capture_output=True)
if proc.returncode == 0:
    print('\ncrontab installed successfully.')
else:
    print('ERROR installing crontab:', proc.stderr)
    sys.exit(1)

# Verify
print('\n=== FINAL CRONTAB ===')
subprocess.run(['crontab', '-l'])
