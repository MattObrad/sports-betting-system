"""
rebuild_calibration.py -- Rebuild WNBA isotonic calibrator on VPS Python/sklearn.

Mirrors fit_calibration.py logic but uses hardcoded VPS paths.
Run once: python3 rebuild_calibration.py
Output: /home/picks/calibration_v1.0.pkl (compatible with VPS sklearn 1.8.0)
"""
import json, sqlite3
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
import joblib

SPORTS_DB  = '/home/picks/sports.db'
SIGMA_PATH = '/home/picks/sigma_calibration_v1.0.json'
OUT_PATH   = '/home/picks/calibration_v1.0.pkl'
MODEL_KEY  = 'baseline_weighted_rolling_5__1.0'

N     = 5
ALPHA = 0.4
DECAY = 0.6
RW    = 0.4   # blend_rolling_weight
SW    = 0.6   # blend_season_weight
EDGES   = [15, 28]
LABELS  = ['<15', '15-28', '28+']
FALLBACK = 'fallback_overall'

WRAW = np.array([ALPHA * DECAY ** i for i in range(N)])

sig_raw = json.load(open(SIGMA_PATH))
SIG = sig_raw['models'][MODEL_KEY]
print(f'Sigma buckets: {list(SIG.keys())}')

FIT_LINES = range(2, 46)
VER_LINES = range(8, 41)
BINS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]


def wpred(window: np.ndarray) -> float:
    valid = window[~np.isnan(window)]
    if valid.size == 0:
        return np.nan
    k = valid.size
    w = WRAW[:k].copy()
    w = w / w.sum()
    return float(np.dot(valid[::-1], w))


def sigma_for(m) -> float:
    if pd.isna(m):
        return SIG[FALLBACK]['sigma']
    lab = LABELS[0] if m < EDGES[0] else LABELS[1] if m < EDGES[1] else LABELS[2]
    s = SIG.get(lab, {}).get('sigma')
    return s if s is not None else SIG[FALLBACK]['sigma']


def build_frame() -> pd.DataFrame:
    con = sqlite3.connect(SPORTS_DB)
    df = pd.read_sql('''
        SELECT internal_player_id, internal_game_id, game_date, points, minutes
        FROM wnba_player_box_scores
        WHERE game_date >= '2021-01-01' AND game_date < '2026-01-01'
          AND did_not_play = 0 AND minutes > 0 AND points IS NOT NULL
        ORDER BY internal_player_id, game_date, internal_game_id
    ''', con)
    con.close()
    print(f'Loaded {len(df):,} game rows (2021-2025)')
    df['year'] = df['game_date'].str[:4]
    g = df.groupby('internal_player_id', sort=False)
    df['r5'] = g['points'].transform(
        lambda s: s.shift(1).rolling(N, min_periods=1).apply(wpred, raw=True))
    df['m5'] = g['minutes'].transform(
        lambda s: s.shift(1).rolling(N, min_periods=N).mean())
    df['savg'] = df.groupby(['internal_player_id', 'year'], sort=False)['points'].transform(
        lambda s: s.shift(1).expanding().mean())
    df['pred'] = np.where(df['savg'].notna(), RW * df['r5'] + SW * df['savg'], df['r5'])
    df['sigma'] = df['m5'].map(sigma_for)
    return df[df['pred'].notna()].copy()


def stack(rows: pd.DataFrame, lines):
    pred = rows['pred'].to_numpy()
    sig  = rows['sigma'].to_numpy()
    act  = rows['points'].to_numpy()
    savg = rows['savg'].to_numpy()
    P, H, V = [], [], []
    for L in lines:
        p = norm.sf(L, loc=pred, scale=sig)
        P.append(p); H.append((act >= L).astype(float)); V.append(act - savg)
    return np.concatenate(P), np.concatenate(H), np.concatenate(V)


def table(probs, hits, vsavg, title):
    print(f'\n{title}')
    print(f"{'bin':<10}{'n':>8}{'avg pred':>10}{'realized':>10}{'gap':>8}")
    print('-' * 46)
    for lo, hi in BINS:
        m = (probs >= lo) & (probs < hi)
        n = int(m.sum())
        if n == 0:
            print(f'{int(lo*100)}-{int(min(hi,1)*100)}%   n=0')
            continue
        ap, rz = probs[m].mean(), hits[m].mean()
        print(f'{int(lo*100)}-{int(min(hi,1)*100)}%   {n:>8}{ap*100:>9.1f}%{rz*100:>9.1f}%{(rz-ap)*100:>+7.1f}')


df = build_frame()
fit = df[df['year'].isin(['2021', '2022', '2023', '2024'])]
ver = df[df['year'] == '2025']
print(f'Fit set (2021-2024): {len(fit):,}   Verify (2025): {len(ver):,}')

Xf, yf, _ = stack(fit, FIT_LINES)
print(f'Fit (pred, line) pairs: {len(Xf):,}')

iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
iso.fit(Xf, yf)

joblib.dump(iso, OUT_PATH)
print(f'\nSaved calibrator -> {OUT_PATH}')

print('\nFitted map (raw -> calibrated):')
for r in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
    print(f'  {r:.2f} -> {float(iso.predict([r])[0]):.3f}')

rawv, hitv, vav = stack(ver, VER_LINES)
calv = iso.predict(rawv)

table(rawv[rawv >= 0.50], hitv[rawv >= 0.50], vav[rawv >= 0.50], 'BEFORE — raw probs (2025 verify):')
table(calv[calv >= 0.50], hitv[calv >= 0.50], vav[calv >= 0.50], 'AFTER — calibrated (2025 verify):')

print('\nGoal: gap near 0 = on the diagonal.')
print('Done. Run replay_wnba.py to re-evaluate with working calibrator.')
