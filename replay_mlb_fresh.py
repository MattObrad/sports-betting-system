"""
replay_mlb_fresh.py -- Fresh MLB replay using current deployed model.

Loads the ensemble pkl via the models module (requires /home/picks on sys.path),
then scores ALL historical game_features rows that have final scores.
Reports results at multiple edge thresholds.

Golden rule: no alerts.db. No DB writes. Read-only.
"""
import sys, json, sqlite3
sys.path.insert(0, '/home/picks')

import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import defaultdict

print('=== MLB FRESH REPLAY — Current Model vs Full Historical Dataset ===')
print()

# ── Load model (requires models module) ──────────────────────────────────────
try:
    from models.ensemble import EnsembleTotalsModel
    import joblib
    MODEL_PATH = '/home/picks/models/saved/ensemble_v1.0.pkl'
    model = joblib.load(MODEL_PATH)
    print(f'Model loaded: {type(model).__name__} from {MODEL_PATH}')
    MODEL_OK = True
except Exception as e:
    print(f'WARNING: Model load failed ({e})')
    print('Falling back to existing monte_carlo_results...')
    MODEL_OK = False

# ── Config ──────────────────────────────────────────────────────────────────
con     = sqlite3.connect('/home/picks/mlb_data.db')
cfg     = json.load(open('/home/picks/config.json'))
bet_cfg = cfg.get('betting', {})
min_confidence = bet_cfg.get('min_confidence', 0.55)
SIGMA = 3.5  # effective sigma for win probability calculation

print(f'Config: min_confidence={min_confidence}, sigma={SIGMA}')
print()

# ════════════════════════════════════════════════════════════════════════════
# PATH A: Fresh predictions from current model + full game_features history
# ════════════════════════════════════════════════════════════════════════════
if MODEL_OK:

    # Columns added by JOIN that were NOT in training features (strip before predict)
    _STRIP = {'game_id', 'feature_version', 'computed_at',
              'game_date', 'market_line', 'juice', 'opening_line',
              'current_line', 'doubleheader', 'actual_total',
              'over_juice', 'under_juice', 'line_movement', 'current_total'}

    all_gf_cols = [r[1] for r in con.execute('PRAGMA table_info(game_features)').fetchall()]
    feature_cols = [c for c in all_gf_cols if c not in _STRIP]

    # Try to get model's exact feature list (EnsembleTotalsModel may expose it)
    for attr in ('feature_names_', 'feature_names_in_', '_feature_names'):
        if hasattr(model, attr):
            model_feats = list(getattr(model, attr))
            available   = [f for f in model_feats if f in all_gf_cols]
            if available:
                feature_cols = available
                print(f'Using model feature list ({attr}): {len(feature_cols)} features')
            break
    else:
        # Try via sub-model (lgbm or xgb booster)
        for sub in ('lgbm_', 'lgbm', '_lgbm'):
            sub_model = getattr(model, sub, None)
            if sub_model is not None:
                try:
                    fnames = sub_model.booster_.feature_name()
                    available = [f for f in fnames if f in all_gf_cols]
                    if available:
                        feature_cols = available
                        print(f'Using lgbm booster feature list: {len(feature_cols)} features')
                    break
                except Exception:
                    pass
        else:
            print(f'Using all game_features columns: {len(feature_cols)} features')

    # Load games: features + final scores, no doubleheaders
    games_df = pd.read_sql('''
        SELECT
            g.game_id, g.game_date,
            CAST(g.home_score + g.away_score AS REAL) AS actual_total,
            gf.current_total   AS market_line,
            gf.over_juice, gf.under_juice,
            gf.line_movement,  gf.current_total
        FROM games g
        JOIN game_features gf ON g.game_id = gf.game_id
        WHERE g.home_score IS NOT NULL
        AND   g.away_score IS NOT NULL
        AND   gf.current_total IS NOT NULL
        AND   gf.current_total BETWEEN 5.0 AND 13.0
        AND   COALESCE(g.doubleheader, 0) = 0
        ORDER BY g.game_date
    ''', con)

    print(f'Games with features + final scores: {len(games_df)}')
    print(f'Date range: {games_df.game_date.min()} to {games_df.game_date.max()}')

    # Hard-block: skip if BOTH market signals are null
    games_df = games_df[~(games_df.line_movement.isna() & games_df.current_total.isna())].copy()
    print(f'After null hard-block: {len(games_df)}')

    # Load feature matrix for these games
    ids_list = ','.join(str(g) for g in games_df.game_id.tolist())
    feat_df  = pd.read_sql(
        f'SELECT game_id, {", ".join(feature_cols)} FROM game_features WHERE game_id IN ({ids_list})',
        con
    )

    merged = games_df.merge(feat_df, on='game_id', how='inner').reset_index(drop=True)
    print(f'Merged: {len(merged)} rows')

    # Feature matrix — to_numeric coerces 'R'/'L' strings to NaN (LightGBM handles NaN natively)
    X = merged[feature_cols].apply(lambda c: pd.to_numeric(c, errors='coerce')).values

    # Run prediction
    try:
        residuals = model.predict(X)
        print(f'\nModel predictions: {len(residuals)} residuals')
        print(f'  Residual mean: {residuals.mean():+.3f}  std: {residuals.std():.3f}')
    except Exception as e:
        print(f'model.predict() failed: {e}')
        # Try prepare_X helper
        try:
            from utils.features import prepare_X
            X_prep = prepare_X(merged, model)
            residuals = model.predict(X_prep)
            print(f'prepare_X predictions: {len(residuals)}')
        except Exception as e2:
            print(f'prepare_X also failed: {e2}')
            print('Cannot generate fresh predictions. Check model interface.')
            MODEL_OK = False

if MODEL_OK:
    # Convert residuals → predicted totals → run edge → win probability
    market_lines   = merged['market_line'].values
    actual_totals  = merged['actual_total'].values
    predicted_totals = residuals + market_lines
    run_edges      = predicted_totals - market_lines
    over_probs     = 1 - norm.cdf(market_lines, predicted_totals, SIGMA)
    is_over        = run_edges > 0
    confidences    = np.where(is_over, over_probs, 1 - over_probs)
    over_juices    = merged['over_juice'].fillna(-110).astype(float).values
    under_juices   = merged['under_juice'].fillna(-110).astype(float).values
    game_dates     = merged['game_date'].values

    # Overall summary
    actual_over_rate = (actual_totals > market_lines).mean()
    print(f'\n=== PREDICTION SUMMARY ===')
    print(f'  Total games:          {len(run_edges)}')
    print(f'  OVER predictions:     {is_over.sum()} ({is_over.mean():.0%})')
    print(f'  UNDER predictions:    {(~is_over).sum()} ({(~is_over).mean():.0%})')
    print(f'  Avg predicted total:  {predicted_totals.mean():.2f}')
    print(f'  Avg market line:      {market_lines.mean():.2f}')
    print(f'  Avg signed edge:      {run_edges.mean():+.3f} runs')
    print(f'  Actual OVER rate:     {actual_over_rate:.1%}')
    print(f'  OVER win rate (all):  {(actual_totals[is_over] > market_lines[is_over]).mean():.1%}')

    # ── Threshold sweep ───────────────────────────────────────────────────
    print(f'\n=== MULTI-THRESHOLD REPLAY ===')
    print(f'Fixed: min_confidence={min_confidence}, extreme_block>3.0, null hard-block')
    print(f'{"Threshold":>10} | {"n":>6} | {"Record":>12} | {"ROI":>8} | {"OVER%":>6} | {"O-WR":>6} | {"U-WR":>6}')
    print('-' * 65)

    full_results = {}
    for min_edge_t in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        mask = (np.abs(run_edges) >= min_edge_t) & (confidences >= min_confidence) & (np.abs(run_edges) < 3.0)
        idx  = np.where(mask)[0]

        if len(idx) == 0:
            print(f'{min_edge_t:>10.1f} | {"0":>6} | {"—":>12} | {"—":>8} | {"—":>6} | {"—":>6} | {"—":>6}')
            full_results[min_edge_t] = None
            continue

        wins = losses = pushes = 0
        total_profit = 0.0
        over_count = over_wins = 0
        under_count = under_wins = 0

        for i in idx:
            if is_over[i]:
                over_count += 1
                odds = int(over_juices[i])
                if actual_totals[i] > market_lines[i]:
                    wins += 1; over_wins += 1
                    total_profit += odds/100 if odds > 0 else 100/abs(odds)
                elif actual_totals[i] < market_lines[i]:
                    losses += 1; total_profit -= 1.0
                else:
                    pushes += 1
            else:
                under_count += 1
                odds = int(under_juices[i])
                if actual_totals[i] < market_lines[i]:
                    wins += 1; under_wins += 1
                    total_profit += odds/100 if odds > 0 else 100/abs(odds)
                elif actual_totals[i] > market_lines[i]:
                    losses += 1; total_profit -= 1.0
                else:
                    pushes += 1

        staked   = wins + losses
        roi      = total_profit / staked * 100 if staked else 0
        over_pct = over_count / len(idx) * 100
        owr      = over_wins  / over_count  * 100 if over_count  else 0
        uwr      = under_wins / under_count * 100 if under_count else 0
        record   = f'{wins}W-{losses}L-{pushes}P'
        print(f'{min_edge_t:>10.1f} | {len(idx):>6} | {record:>12} | {roi:>+7.1f}% | {over_pct:>5.0f}% | {owr:>5.1f}% | {uwr:>5.1f}%')
        full_results[min_edge_t] = {'n': len(idx), 'wins': wins, 'losses': losses,
                                     'profit': total_profit, 'roi': roi}

    print()
    print('Break-even win rate: 52.4% at -110 | 53.5% at -115')

    # ── Monthly breakdown at 1.0-run threshold ────────────────────────────
    print(f'\n=== MONTHLY BREAKDOWN (threshold=1.0 run, min_confidence={min_confidence}) ===')
    mask = (np.abs(run_edges) >= 1.0) & (confidences >= min_confidence) & (np.abs(run_edges) < 3.0)
    monthly = defaultdict(lambda: {'W':0,'L':0,'P':0,'profit':0.0,'over':0,'total':0})

    for i in np.where(mask)[0]:
        month = str(game_dates[i])[:7]
        monthly[month]['total'] += 1
        if is_over[i]:
            monthly[month]['over'] += 1
            odds = int(over_juices[i])
            if actual_totals[i] > market_lines[i]:
                monthly[month]['W'] += 1
                monthly[month]['profit'] += odds/100 if odds > 0 else 100/abs(odds)
            elif actual_totals[i] < market_lines[i]:
                monthly[month]['L'] += 1; monthly[month]['profit'] -= 1.0
            else:
                monthly[month]['P'] += 1
        else:
            odds = int(under_juices[i])
            if actual_totals[i] < market_lines[i]:
                monthly[month]['W'] += 1
                monthly[month]['profit'] += odds/100 if odds > 0 else 100/abs(odds)
            elif actual_totals[i] > market_lines[i]:
                monthly[month]['L'] += 1; monthly[month]['profit'] -= 1.0
            else:
                monthly[month]['P'] += 1

    if monthly:
        for month in sorted(monthly.keys()):
            m = monthly[month]
            staked = m['W'] + m['L']
            roi = m['profit'] / staked * 100 if staked else 0
            over_pct = m['over'] / m['total'] * 100 if m['total'] else 0
            print(f'  {month}: {m["W"]}W-{m["L"]}L-{m["P"]}P | {m["profit"]:+.2f}u | {roi:+.1f}% | {over_pct:.0f}% OVER')
    else:
        print('  No qualifying bets at 1.0-run threshold.')

# ════════════════════════════════════════════════════════════════════════════
# PATH B: Fallback — use existing monte_carlo_results
# ════════════════════════════════════════════════════════════════════════════
if not MODEL_OK:
    print()
    print('=== FALLBACK: EXISTING MC RESULTS ===')
    rows = con.execute('''
        SELECT mcr.run_date, mcr.game_id, mcr.market_line, mcr.predicted_total,
               mcr.ensemble_over_prob, mcr.juice,
               g.game_date, g.home_score + g.away_score AS actual_total,
               gf.line_movement, gf.current_total, gf.over_juice, gf.under_juice,
               COALESCE(g.doubleheader, 0) AS doubleheader
        FROM monte_carlo_results mcr
        JOIN games g  ON mcr.game_id = g.game_id
        JOIN game_features gf ON mcr.game_id = gf.game_id
        WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        AND mcr.predicted_total IS NOT NULL AND mcr.market_line IS NOT NULL
        AND mcr.market_line <= 13.0 AND COALESCE(g.doubleheader, 0) = 0
        ORDER BY mcr.run_date
    ''').fetchall()
    print(f'MC results with grades: {len(rows)}')

    for thresh in [0.8, 1.0, 1.2, 1.5]:
        qualifying = []
        for row in rows:
            (run_date, game_id, market_line, predicted, over_prob, juice,
             game_date, actual, line_movement, current_total, over_j, under_j, dh) = row
            run_edge  = predicted - market_line
            abs_edge  = abs(run_edge)
            direction = 'OVER' if run_edge > 0 else 'UNDER'
            confidence = over_prob if direction == 'OVER' else 1.0 - over_prob
            if line_movement is None and current_total is None: continue
            if abs_edge > 3.0: continue
            if abs_edge < thresh: continue
            if confidence < min_confidence: continue
            if direction == 'OVER':
                result = 'WIN' if actual > market_line else ('LOSS' if actual < market_line else 'PUSH')
                odds = int(over_j or juice or -110)
            else:
                result = 'WIN' if actual < market_line else ('LOSS' if actual > market_line else 'PUSH')
                odds = int(under_j or -110)
            profit = (odds/100 if odds > 0 else 100/abs(odds)) if result=='WIN' else (-1.0 if result=='LOSS' else 0.0)
            qualifying.append({'result': result, 'profit': profit, 'direction': direction})
        wins   = sum(1 for q in qualifying if q['result']=='WIN')
        losses = sum(1 for q in qualifying if q['result']=='LOSS')
        staked = wins + losses
        profit = sum(q['profit'] for q in qualifying)
        roi    = profit/staked*100 if staked else 0
        print(f'  thresh={thresh}: n={len(qualifying)} | {wins}W-{losses}L | {roi:+.1f}% ROI')
