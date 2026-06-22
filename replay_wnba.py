"""
replay_wnba.py -- Clean historical replay of WNBA model.

Rebuilds predictions from raw ingredients:
  - sports.db wnba_player_box_scores (rolling stats, point-in-time)
  - Postgres props_snapshots (Kambi lines and odds)
  - sigma_calibration_v1.0.json (sigma by minutes bucket)
  - calibration_v1.0.pkl (isotonic calibrator)
  - Current wnba_config.json thresholds

This answers: "If today's model had run on every WNBA game since May 27,
what would it have done?"

No alerts.db.
"""
import json, sqlite3, psycopg2, math, os
import joblib
from collections import defaultdict
from datetime import datetime, timezone

# ── config ───────────────────────────────────────────────────────────────────
cfg      = json.load(open('/home/picks/wnba_config.json'))
bet_cfg  = cfg['betting']
mod_cfg  = cfg['model']

MIN_EDGE_PROB   = bet_cfg.get('min_edge_prob',          0.06)
MIN_MODEL_PROB  = bet_cfg.get('min_model_prob',         0.60)
EDGE_SHRINK     = bet_cfg.get('edge_shrink',            0.60)
MIN_AGREEMENT   = bet_cfg.get('min_agreement_thresholds', 2)
MAX_ODDS        = bet_cfg.get('max_odds_american',      -300)  # negative = max fav odds

N_LOOKBACK      = mod_cfg.get('n_lookback',     5)
ALPHA           = mod_cfg.get('alpha',          0.4)
DECAY           = mod_cfg.get('decay',          0.6)
BLEND_ROLL      = mod_cfg.get('blend_rolling_weight', 0.4)
BLEND_SEASON    = mod_cfg.get('blend_season_weight',  0.6)
MIN_PRIOR_GAMES = cfg['prediction'].get('min_prior_games', 1)

SIGMA_PATH = '/home/picks/sigma_calibration_v1.0.json'
CAL_PATH   = '/home/picks/calibration_v1.0.pkl'

print('=== WNBA CLEAN REPLAY ===')
print(f'Config: min_edge_prob={MIN_EDGE_PROB}, min_model_prob={MIN_MODEL_PROB}')
print(f'        edge_shrink={EDGE_SHRINK}, min_agreement={MIN_AGREEMENT}')
print(f'        rolling: alpha={ALPHA}, decay={DECAY}, n={N_LOOKBACK}')
print(f'        blend: {BLEND_ROLL*100:.0f}% rolling + {BLEND_SEASON*100:.0f}% season')
print()

# ── load sigma calibration ────────────────────────────────────────────────────
_sigma_raw = json.load(open(SIGMA_PATH))
# Structure: {"models": {"baseline_weighted_rolling_5__1.0": {"<15": {"sigma":...}, ...}}}
_sigma_model_key = list(_sigma_raw.get('models', {}).keys())[0]
sigma_cal = _sigma_raw['models'][_sigma_model_key]
print(f'Sigma cal buckets: {list(sigma_cal.keys())}')

def get_sigma(avg_minutes: float) -> float:
    edges   = mod_cfg.get('minutes_bucket_edges',  [15, 28])
    labels  = mod_cfg.get('minutes_bucket_labels', ['<15','15-28','28+'])
    fallback= mod_cfg.get('sigma_fallback_label',  'fallback_overall')
    if avg_minutes < edges[0]:
        label = labels[0]
    elif avg_minutes < edges[1]:
        label = labels[1]
    else:
        label = labels[2]
    bucket = sigma_cal.get(label, sigma_cal.get(fallback, {}))
    return bucket.get('sigma', 5.0)

# ── load isotonic calibrator ──────────────────────────────────────────────────
calibrator = None
if os.path.exists(CAL_PATH):
    try:
        calibrator = joblib.load(CAL_PATH)
        print(f'Calibrator loaded from {CAL_PATH} ({type(calibrator).__name__})')
    except Exception as e:
        print(f'WARNING: calibrator failed to load ({e}) — using raw Normal-CDF probs')
else:
    print(f'WARNING: calibrator not found at {CAL_PATH} — using raw Normal-CDF probs')

# ── probability helpers ───────────────────────────────────────────────────────
def normal_cdf_over(predicted: float, line: float, sigma: float) -> float:
    """P(score >= line) using Normal CDF."""
    if sigma <= 0:
        return 1.0 if predicted >= line else 0.0
    z = (line - predicted) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))

def calibrate(raw_prob: float) -> float:
    if calibrator is None:
        return raw_prob
    try:
        return float(calibrator.predict([raw_prob])[0])
    except Exception:
        return raw_prob

def implied_prob(odds: int) -> float:
    """Vig-included probability from American odds."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

def decimal_from_american(odds: int) -> float:
    if odds > 0:
        return odds / 100 + 1.0
    return -100 / odds + 1.0

def ev(model_prob: float, over_odds: int) -> float:
    return model_prob * decimal_from_american(over_odds) - 1.0

# ── survival integral (market expected points) ────────────────────────────────
def market_expected_points(lines_odds: list) -> float | None:
    """
    Mirrors predict_wnba.market_expected_points exactly.
    lines_odds: list of (line_float, over_odds_int) — vig-included, single-sided.
    Uses raw implied probability (no de-vig), enforces monotone non-increasing.
    Requires >= 2 ladder points.
    """
    if len(lines_odds) < 2:
        return None
    pts = sorted(lines_odds, key=lambda x: x[0])
    # Build vig-included survival curve, enforce monotone non-increasing
    ss = []
    for line, odds in pts:
        p = implied_prob(int(odds))
        if ss:
            p = min(p, ss[-1])
        ss.append(p)
    # Survival integral: E[X] ≈ sum over milestones of p_i * delta_i
    total = 0.0
    prev_t = 0.0
    for (line, _), p in zip(pts, ss):
        total += p * (line - prev_t)
        prev_t = line
    return total if total > 0 else None

# ── box score helpers ─────────────────────────────────────────────────────────
sports = sqlite3.connect('/home/picks/sports.db')

def get_rolling_stats(internal_id: int, before_date: str):
    """
    Compute rolling-5 and season average for player's points BEFORE before_date.
    Returns (rolling_pred, season_avg, avg_minutes, n_games) or None.
    """
    rows = sports.execute('''
        SELECT points, minutes FROM wnba_player_box_scores
        WHERE internal_player_id = ?
        AND game_date < ?
        AND did_not_play = 0
        ORDER BY game_date DESC
        LIMIT ?
    ''', (internal_id, before_date, max(N_LOOKBACK, 20))).fetchall()

    if not rows or len(rows) < MIN_PRIOR_GAMES:
        return None

    # Season (all games this season before date)
    season_rows = sports.execute('''
        SELECT points FROM wnba_player_box_scores
        WHERE internal_player_id = ?
        AND game_date >= '2026-01-01' AND game_date < ?
        AND did_not_play = 0
    ''', (internal_id, before_date)).fetchall()
    season_avg = sum(r[0] for r in season_rows) / len(season_rows) if season_rows else None

    # Rolling-5 exponentially weighted
    recent = rows[:N_LOOKBACK]
    pts = [r[0] for r in recent]
    mins = [r[1] for r in recent if r[1] is not None]

    # Exponential weights: most recent = weight 1, oldest = weight DECAY^(n-1)
    weights = [ALPHA * (DECAY ** i) for i in range(len(pts))]
    total_w = sum(weights)
    rolling_pred = sum(p * w for p, w in zip(pts, weights)) / total_w if total_w else pts[0]

    avg_mins = sum(mins) / len(mins) if mins else 25.0

    # Blend
    if season_avg is not None:
        blended = BLEND_ROLL * rolling_pred + BLEND_SEASON * season_avg
    else:
        blended = rolling_pred

    return blended, season_avg, avg_mins, len(recent)

# ── alias lookup (Kambi name → internal_id) ──────────────────────────────────
alias_map: dict[str, int] = {}
for row in sports.execute("SELECT source_id, internal_id FROM id_aliases WHERE source='kambi'").fetchall():
    alias_map[row[0].lower()] = row[1]

print(f'Alias map loaded: {len(alias_map)} Kambi→internal mappings')

# ── load props from Postgres ──────────────────────────────────────────────────
pg = psycopg2.connect(host='localhost', dbname='picksdb', user='picksuser', password='password')
cur = pg.cursor()

# Get all WNBA Player Points props — mirrors production _PROPS_SQL exactly.
# under_odds is always NULL (single-sided market); we use vig-included over_odds only.
# For each (event_id, player_name, line): take latest snapshot; for same-time ties, highest over_odds.
cur.execute("""
    WITH latest AS (
        SELECT DISTINCT ON (p.event_id, p.player_name, p.line)
            p.event_id,
            p.player_name,
            p.line,
            p.over_odds,
            p.snapshot_time,
            g.game_time::date AS game_date
        FROM props_snapshots p
        JOIN games g ON p.event_id = g.event_id
        WHERE g.league = 'WNBA'
        AND p.market_type = 'Player Points'
        AND p.over_odds IS NOT NULL
        ORDER BY p.event_id, p.player_name, p.line,
                 p.snapshot_time DESC, p.over_odds DESC
    )
    SELECT event_id, game_date, player_name, line, over_odds, snapshot_time
    FROM latest
    ORDER BY game_date, player_name, line
""")
props = cur.fetchall()
print(f'Total WNBA Player Points snapshots (latest per game/player/line): {len(props)}')

# ── replay loop ───────────────────────────────────────────────────────────────
qualifying = []
skipped = defaultdict(int)
no_alias  = set()

from itertools import groupby

def groupkey(r): return (r[0], r[1], r[2])  # event_id, game_date, player_name

props.sort(key=groupkey)

for (event_id, game_date, player_name), lines_iter in groupby(props, key=groupkey):
    lines_list = list(lines_iter)

    # Look up internal player ID
    key = player_name.lower()
    if key not in alias_map:
        skipped['no_alias'] += 1
        no_alias.add(player_name)
        continue
    internal_id = alias_map[key]

    # Compute rolling stats as of game_date (point-in-time)
    stats = get_rolling_stats(internal_id, str(game_date))
    if stats is None:
        skipped['insufficient_history'] += 1
        continue

    blended_pred, season_avg, avg_mins, n_games = stats

    # Get sigma
    sigma = get_sigma(avg_mins)

    # Build the milestone ladder: (line, over_odds) — vig-included, single-sided
    # Mirrors predict_wnba: _ladder[(player, event)].append((line, over_odds))
    ladder = [(float(r[3]), int(r[4])) for r in lines_list]
    mkt_expected = market_expected_points(ladder)

    # Market anchoring: 50% model + 50% market (mirrors predict_wnba market anchor)
    if mkt_expected is not None:
        final_pred = 0.5 * blended_pred + 0.5 * mkt_expected
    else:
        final_pred = blended_pred

    # Evaluate each line — collect qualifying (agreeing) lines
    agreeing_lines = []
    for (ev_id, g_date, p_name, line, over_odds, snap_time) in lines_list:
        line = float(line)
        over_odds = int(over_odds)
        raw_prob = normal_cdf_over(final_pred, line, sigma)
        cal_prob = calibrate(raw_prob)
        # Fair implied prob: vig-included (no under_odds; single-sided market)
        fair_prob = implied_prob(over_odds)
        edge = cal_prob - fair_prob
        shrunken_edge = edge * EDGE_SHRINK

        # Max odds: skip if too short a favorite (e.g., -400 is worse than -300)
        if over_odds < MAX_ODDS:
            continue

        if cal_prob >= MIN_MODEL_PROB and shrunken_edge >= MIN_EDGE_PROB:
            ev_val = ev(cal_prob, over_odds)
            if ev_val > 0:
                agreeing_lines.append({
                    'line': line, 'over_odds': over_odds,
                    'cal_prob': cal_prob, 'edge': shrunken_edge, 'ev': ev_val
                })

    if len(agreeing_lines) < MIN_AGREEMENT:
        skipped['below_agreement'] += 1
        continue

    # Use the primary line (highest EV among qualifying)
    best_line = max(agreeing_lines, key=lambda x: x['ev'])

    # Look up actual score
    game_date_str = str(game_date)
    actual_row = sports.execute('''
        SELECT points FROM wnba_player_box_scores
        WHERE internal_player_id = ?
        AND game_date = ?
        AND did_not_play = 0
    ''', (internal_id, game_date_str)).fetchone()

    if actual_row is None:
        # Check if DNP
        dnp_row = sports.execute('''
            SELECT did_not_play FROM wnba_player_box_scores
            WHERE internal_player_id = ?
            AND game_date = ?
        ''', (internal_id, game_date_str)).fetchone()
        if dnp_row and dnp_row[0]:
            skipped['dnp'] += 1
            continue
        # No box score at all — game may not be in sports.db yet
        skipped['no_result'] += 1
        qualifying.append({
            'game_date': game_date_str, 'player': player_name,
            'line': best_line['line'], 'over_odds': best_line['over_odds'],
            'pred': final_pred, 'cal_prob': best_line['cal_prob'],
            'edge': best_line['edge'], 'n_lines': len(agreeing_lines),
            'result': 'PENDING', 'profit': None, 'actual': None,
        })
        continue

    actual_pts = actual_row[0]
    line_val   = best_line['line']
    if actual_pts > line_val:   result = 'WIN'
    elif actual_pts < line_val: result = 'LOSS'
    else:                       result = 'PUSH'

    odds = best_line['over_odds']
    if result == 'WIN':
        profit = decimal_from_american(odds) - 1.0
    elif result == 'LOSS':
        profit = -1.0
    else:
        profit = 0.0

    qualifying.append({
        'game_date': game_date_str, 'player': player_name,
        'line': line_val, 'over_odds': odds,
        'pred': final_pred, 'cal_prob': best_line['cal_prob'],
        'edge': best_line['edge'], 'n_lines': len(agreeing_lines),
        'result': result, 'profit': profit, 'actual': actual_pts,
    })

# ── results ──────────────────────────────────────────────────────────────────
print(f'\nSkip breakdown:')
for k, v in sorted(skipped.items()): print(f'  {k}: {v}')
print(f'Players without alias (sample): {list(no_alias)[:10]}')
print()

graded  = [b for b in qualifying if b['result'] != 'PENDING']
pending = [b for b in qualifying if b['result'] == 'PENDING']
wins    = sum(1 for b in graded if b['result'] == 'WIN')
losses  = sum(1 for b in graded if b['result'] == 'LOSS')
pushes  = sum(1 for b in graded if b['result'] == 'PUSH')
staked  = wins + losses
profit  = sum(b['profit'] for b in graded)
roi     = (profit / staked * 100) if staked else 0.0

print('=== WNBA CLEAN REPLAY RESULTS ===')
print(f'Qualifying bets: {len(qualifying)} ({staked} graded, {len(pending)} pending)')
print(f'Record: {wins}W-{losses}L-{pushes}P')
print(f'Total profit: {profit:+.2f}u')
print(f'ROI: {roi:+.1f}%')
print(f'$20 flat stake — wagered: ${staked*20} | net: ${profit*20:+.2f}')

# Baseline: "always bet OVER on every prop that meets line/odds criteria (ignore model)"
# This tests if the model adds anything over a naive OVER strategy
always_over = sum(1 for b in graded if b['actual'] is not None and b['actual'] > b['line'])
if staked:
    print(f'\nBaseline "always OVER": {always_over}/{staked} = {always_over/staked:.1%} win rate')
    print(f'  Baseline break-even at -110 odds: ~52.4%')
else:
    print('\nNo graded bets — cannot compute baseline.')

# By month
monthly = defaultdict(lambda: {'W':0,'L':0,'profit':0.0,'pending':0})
for b in qualifying:
    m = b['game_date'][:7]
    if b['result'] == 'PENDING': monthly[m]['pending'] += 1
    elif b['result'] == 'WIN':   monthly[m]['W'] += 1; monthly[m]['profit'] += b['profit']
    elif b['result'] == 'LOSS':  monthly[m]['L'] += 1; monthly[m]['profit'] += b['profit']

print('\n=== BY MONTH ===')
for month in sorted(monthly):
    m = monthly[month]
    t = m['W'] + m['L']
    roi_m = m['profit'] / t * 100 if t else 0
    print(f"  {month}: {m['W']}W-{m['L']}L-{m['pending']}P | {m['profit']:+.2f}u | {roi_m:+.1f}%")

# By predicted points tier
print('\n=== BY PLAYER TIER (predicted points) ===')
for lo, hi, label in [(0,15,'<15pts'),(15,20,'15-20pts'),(20,99,'>20pts')]:
    bucket = [b for b in graded if lo <= b['pred'] < hi]
    if not bucket: continue
    bw = sum(1 for b in bucket if b['result'] == 'WIN')
    bp = sum(b['profit'] for b in bucket)
    print(f'  {label}: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u, {bp/len(bucket)*100:+.1f}% ROI')

# By line height
print('\n=== BY LINE HEIGHT ===')
for lo, hi in [(0,15),(15,20),(20,25),(25,99)]:
    bucket = [b for b in graded if lo <= b['line'] < hi]
    if not bucket: continue
    bw = sum(1 for b in bucket if b['result'] == 'WIN')
    bp = sum(b['profit'] for b in bucket)
    print(f'  Line {lo}-{hi}: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u')

# By agreement count
print('\n=== BY AGREEMENT THRESHOLD COUNT ===')
for n in [2, 3, 4]:
    bucket = [b for b in graded if b['n_lines'] == n]
    if not bucket: continue
    bw = sum(1 for b in bucket if b['result'] == 'WIN')
    bp = sum(b['profit'] for b in bucket)
    print(f'  {n} agreeing lines: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u')

# By edge bucket
print('\n=== BY EDGE SIZE ===')
for lo, hi in [(0.06,0.08),(0.08,0.10),(0.10,0.15),(0.15,1.0)]:
    bucket = [b for b in graded if lo <= b['edge'] < hi]
    if not bucket: continue
    bw = sum(1 for b in bucket if b['result'] == 'WIN')
    bp = sum(b['profit'] for b in bucket)
    print(f'  Edge {lo:.0%}-{hi:.0%}: n={len(bucket)}, {bw}W-{len(bucket)-bw}L, {bp:+.2f}u, {bp/len(bucket)*100:+.1f}% ROI')

# Best / worst
if graded:
    best  = max(graded, key=lambda b: b.get('profit') or -99)
    worst = min(graded, key=lambda b: b.get('profit') or 99)
    print(f'\nBest:  {best["game_date"]} {best["player"]} {best["line"]} | pred={best["pred"]:.1f} actual={best["actual"]} | {best["profit"]:+.2f}u')
    print(f'Worst: {worst["game_date"]} {worst["player"]} {worst["line"]} | pred={worst["pred"]:.1f} actual={worst["actual"]} | {worst["profit"]:+.2f}u')
