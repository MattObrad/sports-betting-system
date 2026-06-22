"""
replay_wnba_fresh.py -- Fresh WNBA replay.

Uses the SAME logic as the production predict_wnba.py:
  - Survival integral (vig-included) for market_expected_points
  - Isotonic calibrator via joblib.load (NOT pickle)
  - Sigma by minutes bucket from sigma_calibration_v1.0.json
  - Market anchoring: 50% blended_pred + 50% market_expected
  - Min-agreement filter (>= 2 lines qualifying for same player/game)

Validation window: all WNBA Kambi props since May 27.
Golden rule: no alerts.db. No DB writes. Read-only.
"""
import json, sqlite3, psycopg2, math, os, joblib
from collections import defaultdict
from itertools import groupby

# ── Config ──────────────────────────────────────────────────────────────────
cfg     = json.load(open('/home/picks/wnba_config.json'))
bet_cfg = cfg['betting']
mod_cfg = cfg['model']

MIN_EDGE_PROB   = bet_cfg.get('min_edge_prob',            0.06)
MIN_MODEL_PROB  = bet_cfg.get('min_model_prob',           0.60)
EDGE_SHRINK     = bet_cfg.get('edge_shrink',              0.60)
MIN_AGREEMENT   = bet_cfg.get('min_agreement_thresholds',    2)
MAX_ODDS        = bet_cfg.get('max_odds_american',         -300)   # negative = max fav
N               = mod_cfg.get('n_lookback',    5)
ALPHA           = mod_cfg.get('alpha',         0.4)
DECAY           = mod_cfg.get('decay',         0.6)
BLEND_ROLL      = mod_cfg.get('blend_rolling_weight', 0.4)
BLEND_SEASON    = mod_cfg.get('blend_season_weight',  0.6)
MIN_PRIOR       = cfg['prediction'].get('min_prior_games', 1)

print('=== WNBA FRESH REPLAY ===')
print(f'Config: min_edge_prob={MIN_EDGE_PROB}, min_model_prob={MIN_MODEL_PROB}')
print(f'        edge_shrink={EDGE_SHRINK}, min_agreement={MIN_AGREEMENT}')
print(f'        max_odds={MAX_ODDS}')
print(f'        blend: {BLEND_ROLL*100:.0f}% rolling + {BLEND_SEASON*100:.0f}% season')
print()

# ── Sigma calibration ────────────────────────────────────────────────────────
_sig_raw = json.load(open('/home/picks/sigma_calibration_v1.0.json'))
_sig_key = list(_sig_raw.get('models', {}).keys())[0]
sigma_cal = _sig_raw['models'][_sig_key]
print(f'Sigma model: {_sig_key}')
print(f'Sigma buckets: {list(sigma_cal.keys())}')

def get_sigma(avg_minutes):
    edges   = mod_cfg.get('minutes_bucket_edges',  [15, 28])
    labels  = mod_cfg.get('minutes_bucket_labels', ['<15','15-28','28+'])
    fallback= mod_cfg.get('sigma_fallback_label',  'fallback_overall')
    label   = labels[0] if avg_minutes < edges[0] else (labels[1] if avg_minutes < edges[1] else labels[2])
    bucket  = sigma_cal.get(label, sigma_cal.get(fallback, {}))
    return bucket.get('sigma', 5.0)

# ── Calibrator ───────────────────────────────────────────────────────────────
calibrator = None
CAL_PATH = '/home/picks/calibration_v1.0.pkl'
if os.path.exists(CAL_PATH):
    try:
        calibrator = joblib.load(CAL_PATH)
        print(f'Calibrator loaded: {type(calibrator).__name__}')
        import numpy as np
        test = np.array([0.50, 0.60, 0.70, 0.80])
        print(f'Calibrator map: {dict(zip(test, calibrator.predict(test).round(3)))}')
    except Exception as e:
        print(f'WARNING: calibrator failed ({e})')
else:
    print(f'WARNING: calibrator not found at {CAL_PATH}')
print()

# ── Probability helpers ──────────────────────────────────────────────────────
def implied_prob(odds):
    if odds < 0: return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

def decimal_odds(odds):
    if odds > 0: return odds / 100 + 1.0
    return -100 / odds + 1.0

def normal_cdf_over(pred, line, sigma):
    if sigma <= 0: return 1.0 if pred >= line else 0.0
    z = (line - pred) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))

def calibrate(raw_prob):
    if calibrator is None: return raw_prob
    try:
        return float(calibrator.predict([raw_prob])[0])
    except Exception:
        return raw_prob

def market_expected_points(lines_odds):
    """Survival integral over the milestone ladder. Mirrors predict_wnba exactly."""
    if len(lines_odds) < 2: return None
    pts = sorted(lines_odds, key=lambda x: x[0])
    ss = []
    for line, odds in pts:
        p = implied_prob(int(odds))
        if ss: p = min(p, ss[-1])
        ss.append(p)
    total = prev_t = 0.0
    for (line, _), p in zip(pts, ss):
        total += p * (line - prev_t)
        prev_t = line
    return total if total > 0 else None

# ── Sports DB (box scores) ───────────────────────────────────────────────────
sports = sqlite3.connect('/home/picks/sports.db')

# Confirm columns exist
bs_cols = [r[1] for r in sports.execute('PRAGMA table_info(wnba_player_box_scores)').fetchall()]
assert 'points'  in bs_cols, 'Column "points" not in wnba_player_box_scores!'
assert 'minutes' in bs_cols, 'Column "minutes" not in wnba_player_box_scores!'
print(f'Box score table: {len(bs_cols)} columns, "points" and "minutes" confirmed')

def get_rolling_stats(internal_id, before_date):
    rows = sports.execute('''
        SELECT points, minutes FROM wnba_player_box_scores
        WHERE internal_player_id = ?
        AND game_date < ?
        AND game_date >= '2026-01-01'
        AND did_not_play = 0
        ORDER BY game_date DESC
        LIMIT ?
    ''', (internal_id, before_date, max(N, 20))).fetchall()

    if not rows or len(rows) < MIN_PRIOR:
        return None

    season_rows = sports.execute('''
        SELECT points FROM wnba_player_box_scores
        WHERE internal_player_id = ?
        AND game_date >= '2026-01-01' AND game_date < ?
        AND did_not_play = 0
    ''', (internal_id, before_date)).fetchall()
    season_avg = (sum(r[0] for r in season_rows) / len(season_rows)) if season_rows else None

    recent = rows[:N]
    pts    = [r[0] for r in recent]
    mins   = [r[1] for r in recent if r[1] is not None]
    weights = [ALPHA * (DECAY ** i) for i in range(len(pts))]
    total_w = sum(weights)
    rolling  = sum(p*w for p,w in zip(pts, weights)) / total_w if total_w else pts[0]
    avg_mins = sum(mins)/len(mins) if mins else 25.0

    blended = BLEND_ROLL * rolling + BLEND_SEASON * season_avg if season_avg else rolling
    return blended, season_avg, avg_mins, len(recent)

# ── Alias map ────────────────────────────────────────────────────────────────
alias_map = {r[0].lower(): r[1] for r in sports.execute(
    "SELECT source_id, internal_id FROM id_aliases WHERE source='kambi'"
).fetchall()}
print(f'Alias map: {len(alias_map)} Kambi→internal_id entries')

# ── Load Kambi props from Postgres ───────────────────────────────────────────
pg  = psycopg2.connect(host='localhost', dbname='picksdb',
                       user='picksuser', password='password')
cur = pg.cursor()

cur.execute("""
    WITH latest AS (
        SELECT DISTINCT ON (p.event_id, p.player_name, p.line)
            p.event_id, p.player_name, p.line, p.over_odds,
            p.snapshot_time, g.game_time::date AS game_date
        FROM props_snapshots p
        JOIN games g ON p.event_id = g.event_id
        WHERE g.league = 'WNBA'
        AND   p.market_type = 'Player Points'
        AND   p.over_odds IS NOT NULL
        ORDER BY p.event_id, p.player_name, p.line,
                 p.snapshot_time DESC, p.over_odds DESC
    )
    SELECT event_id, game_date, player_name, line, over_odds
    FROM latest ORDER BY game_date, player_name, line
""")
props = cur.fetchall()
print(f'\nWNBA Player Points props (latest per event/player/line): {len(props)}')

# ── Replay loop ───────────────────────────────────────────────────────────────
qualifying = []
skipped    = defaultdict(int)
no_alias   = set()

def groupkey(r): return (r[0], r[1], r[2])   # event_id, game_date, player_name
props.sort(key=groupkey)

for (event_id, game_date, player_name), lines_iter in groupby(props, key=groupkey):
    lines_list = list(lines_iter)
    gd = str(game_date)

    key = player_name.lower()
    if key not in alias_map:
        skipped['no_alias'] += 1
        no_alias.add(player_name)
        continue

    internal_id = alias_map[key]
    stats = get_rolling_stats(internal_id, gd)
    if stats is None:
        skipped['insufficient_history'] += 1
        continue

    blended_pred, season_avg, avg_mins, n_games = stats
    sigma      = get_sigma(avg_mins)
    ladder     = [(float(r[3]), int(r[4])) for r in lines_list]
    mkt_exp    = market_expected_points(ladder)
    final_pred = 0.5 * blended_pred + 0.5 * mkt_exp if mkt_exp is not None else blended_pred

    agreeing = []
    for (ev_id, g_date, p_name, line, over_odds) in lines_list:
        line, over_odds = float(line), int(over_odds)
        raw_prob  = normal_cdf_over(final_pred, line, sigma)
        cal_prob  = calibrate(raw_prob)
        fair_prob = implied_prob(over_odds)
        edge      = cal_prob - fair_prob
        shrunken  = edge * EDGE_SHRINK
        if over_odds < MAX_ODDS:          continue
        if cal_prob  < MIN_MODEL_PROB:    continue
        if shrunken  < MIN_EDGE_PROB:     continue
        ev_val = cal_prob * decimal_odds(over_odds) - 1.0
        if ev_val <= 0:                   continue
        agreeing.append({'line': line, 'over_odds': over_odds, 'cal_prob': cal_prob,
                         'edge': shrunken, 'ev': ev_val})

    if len(agreeing) < MIN_AGREEMENT:
        skipped['below_agreement'] += 1
        continue

    best = max(agreeing, key=lambda x: x['ev'])

    # Actual result
    actual_row = sports.execute('''
        SELECT points FROM wnba_player_box_scores
        WHERE internal_player_id = ? AND game_date = ? AND did_not_play = 0
    ''', (internal_id, gd)).fetchone()

    if actual_row is None:
        dnp = sports.execute(
            'SELECT did_not_play FROM wnba_player_box_scores WHERE internal_player_id=? AND game_date=?',
            (internal_id, gd)
        ).fetchone()
        if dnp and dnp[0]:
            skipped['dnp'] += 1; continue
        skipped['no_result'] += 1
        qualifying.append({'game_date': gd, 'player': player_name,
                           'line': best['line'], 'over_odds': best['over_odds'],
                           'pred': final_pred, 'cal_prob': best['cal_prob'],
                           'edge': best['edge'], 'n_lines': len(agreeing),
                           'result': 'PENDING', 'profit': None, 'actual': None})
        continue

    actual_pts = actual_row[0]
    line_val   = best['line']
    result = ('WIN' if actual_pts > line_val else
              'LOSS' if actual_pts < line_val else 'PUSH')
    odds   = best['over_odds']
    profit = (decimal_odds(odds) - 1.0 if result == 'WIN' else
              -1.0 if result == 'LOSS' else 0.0)
    qualifying.append({'game_date': gd, 'player': player_name,
                       'line': line_val, 'over_odds': odds,
                       'pred': final_pred, 'cal_prob': best['cal_prob'],
                       'edge': best['edge'], 'n_lines': len(agreeing),
                       'result': result, 'profit': profit, 'actual': actual_pts})

# ── Results ───────────────────────────────────────────────────────────────────
print(f'\nSkip breakdown:')
for k, v in sorted(skipped.items()): print(f'  {k}: {v}')
print(f'No-alias players (sample): {list(no_alias)[:8]}')
print()

graded  = [b for b in qualifying if b['result'] != 'PENDING']
pending = [b for b in qualifying if b['result'] == 'PENDING']
wins    = sum(1 for b in graded if b['result'] == 'WIN')
losses  = sum(1 for b in graded if b['result'] == 'LOSS')
pushes  = sum(1 for b in graded if b['result'] == 'PUSH')
staked  = wins + losses
profit  = sum(b['profit'] for b in graded)
roi     = (profit / staked * 100) if staked else 0.0

print('=== WNBA FRESH REPLAY RESULTS ===')
print(f'Qualifying bets: {len(qualifying)} ({staked} graded, {len(pending)} pending)')
print(f'Record:  {wins}W-{losses}L-{pushes}P')
print(f'Profit:  {profit:+.2f}u')
print(f'ROI:     {roi:+.1f}%')
print(f'$20 flat: wagered ${staked*20} | net ${profit*20:+.2f}')

if staked:
    always_over = sum(1 for b in graded if b['actual'] is not None and b['actual'] > b['line'])
    print(f'\nBaseline "always OVER":  {always_over}/{staked} = {always_over/staked:.1%}')
    print(f'Break-even at -110 odds: 52.4%')

# By month
print('\n=== BY MONTH ===')
monthly = defaultdict(lambda: {'W':0,'L':0,'profit':0.0,'pending':0})
for b in qualifying:
    m = b['game_date'][:7]
    if b['result'] == 'PENDING': monthly[m]['pending'] += 1
    elif b['result'] == 'WIN':   monthly[m]['W'] += 1; monthly[m]['profit'] += b['profit']
    elif b['result'] == 'LOSS':  monthly[m]['L'] += 1; monthly[m]['profit'] += b['profit']
for month in sorted(monthly):
    m = monthly[month]
    t = m['W'] + m['L']
    r = m['profit']/t*100 if t else 0
    print(f'  {month}: {m["W"]}W-{m["L"]}L-{m["pending"]}P | {m["profit"]:+.2f}u | {r:+.1f}%')

# By player tier
print('\n=== BY PREDICTED POINTS TIER ===')
for lo, hi, lbl in [(0,15,'<15pts'),(15,20,'15-20pts'),(20,99,'>20pts')]:
    bucket = [b for b in graded if lo <= b['pred'] < hi]
    if not bucket: continue
    bw = sum(1 for b in bucket if b['result']=='WIN')
    bp = sum(b['profit'] for b in bucket)
    print(f'  {lbl}: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u, {bp/len(bucket)*100:+.1f}% ROI')

# By line height
print('\n=== BY LINE HEIGHT ===')
for lo, hi in [(0,15),(15,20),(20,25),(25,99)]:
    bucket = [b for b in graded if lo <= b['line'] < hi]
    if not bucket: continue
    bw = sum(1 for b in bucket if b['result']=='WIN')
    bp = sum(b['profit'] for b in bucket)
    print(f'  Line {lo}-{hi}: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u, {bp/len(bucket)*100:+.1f}% ROI')

# By agreement count
print('\n=== BY AGREEMENT COUNT ===')
for n in [2, 3, 4, 5]:
    bucket = [b for b in graded if b['n_lines'] == n]
    if not bucket: continue
    bw = sum(1 for b in bucket if b['result']=='WIN')
    bp = sum(b['profit'] for b in bucket)
    print(f'  {n} agreeing lines: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u')

# Best and worst
if graded:
    best  = max(graded, key=lambda b: b.get('profit') or -99)
    worst = min(graded, key=lambda b: b.get('profit') or  99)
    print(f'\nBest:  {best["game_date"]} {best["player"]} o{best["line"]} | pred={best["pred"]:.1f} act={best["actual"]} | {best["profit"]:+.2f}u')
    print(f'Worst: {worst["game_date"]} {worst["player"]} o{worst["line"]} | pred={worst["pred"]:.1f} act={worst["actual"]} | {worst["profit"]:+.2f}u')
