import requests, os
from dotenv import load_dotenv
load_dotenv('/home/picks/.env')

webhooks = {
    'MLB':     os.getenv('DISCORD_WEBHOOK_MLB'),
    'Tennis':  os.getenv('DISCORD_WEBHOOK_TENNIS'),
    'Results': os.getenv('DISCORD_WEBHOOK_RESULTS'),
    'Weekly':  os.getenv('DISCORD_WEBHOOK_WEEKLY'),
    'WNBA':    os.getenv('DISCORD_WEBHOOK_WNBA'),
}

for name, url in webhooks.items():
    if not url:
        print(f'{name}: NOT SET in .env')
        continue
    try:
        r = requests.post(
            url,
            json={'content': f'Test from VPS - {name}'},
            timeout=10,
            headers={
                'User-Agent': 'DiscordBot (https://github.com, 1.0)',
                'Content-Type': 'application/json',
            }
        )
        print(f'{name}: {r.status_code} - {r.text[:200]}')
    except Exception as e:
        print(f'{name}: ERROR - {e}')
