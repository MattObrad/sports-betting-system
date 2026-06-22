import requests, os
from dotenv import load_dotenv
load_dotenv('/home/picks/.env')

HEADERS = {'User-Agent': 'DiscordBot (https://github.com, 1.0)'}

msg = (
    "\U0001f3c0 WNBA model paused — backtest confirmed no edge "
    "(5W-13L, -49% ROI on clean replay). "
    "Continuing to collect data. "
    "Will re-evaluate when WNBA rebounds model is built."
)

for key in ['DISCORD_WEBHOOK_WNBA', 'DISCORD_WEBHOOK_RESULTS']:
    url = os.getenv(key, '').strip()
    if not url:
        print(f'{key}: not set')
        continue
    try:
        r = requests.post(url, json={'content': msg}, headers=HEADERS, timeout=10)
        print(f'{key}: {r.status_code}')
        if r.status_code in (200, 204):
            break
    except Exception as e:
        print(f'{key}: ERROR - {e}')
