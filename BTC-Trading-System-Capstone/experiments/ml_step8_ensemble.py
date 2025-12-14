# 파일명: ml_step8_ensemble.py
# Phase 9: 앙상블 모델 (XGBoost + CatBoost + LightGBM)
import sys
import subprocess
import argparse
import warnings
warnings.filterwarnings('ignore')

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
except:
    pip_install("pandas")
    import pandas as pd

try:
    import numpy as np
except:
    pip_install("numpy")
    import numpy as np

try:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
except:
    pip_install("scikit-learn")
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

try:
    import xgboost as xgb
except:
    pip_install("xgboost")
    import xgboost as xgb

try:
    import lightgbm as lgb
except:
    pip_install("lightgbm")
    import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
except:
    pip_install("catboost")
    from catboost import CatBoostClassifier

#################################
import os
FILE_PATH = os.environ.get("BTC_FILE_PATH", r"C:\Users\User\binance_data\1m_history.csv")
USE_ROWS = None
#################################

def load_data(path, use_rows=None):
    df = pd.read_csv(path, usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    if use_rows is not None and len(df) > use_rows:
        df = df.tail(use_rows)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def resample_ohlcv(df, timeframe='15min'):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['datetime'], unit='ms')
    df = df.set_index('timestamp')
    
    resampled = df.resample(timeframe).agg({
        'datetime': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index(drop=True)

# ============ 기술적 지표 ============

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def calc_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return (upper - lower) / sma, (series - lower) / (upper - lower)

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ============ 특성 공학 ============

def make_features(df):
    # 기본 특성
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['vol_chg'] = df['volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df['ret_mean_10'] = df['ret_1'].rolling(10, min_periods=10).mean()
    df['ret_std_10'] = df['ret_1'].rolling(10, min_periods=10).std()
    
    # RSI
    df['rsi_14'] = calc_rsi(df['close'], period=14)
    df['rsi_7'] = calc_rsi(df['close'], period=7)
    
    # EMA
    df['ema_5'] = calc_ema(df['close'], span=5)
    df['ema_10'] = calc_ema(df['close'], span=10)
    df['ema_20'] = calc_ema(df['close'], span=20)
    df['ema_5_ratio'] = df['close'] / df['ema_5'] - 1
    df['ema_10_ratio'] = df['close'] / df['ema_10'] - 1
    df['ema_20_ratio'] = df['close'] / df['ema_20'] - 1
    df['ema_cross_5_10'] = (df['ema_5'] > df['ema_10']).astype(int)
    df['ema_cross_5_20'] = (df['ema_5'] > df['ema_20']).astype(int)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(df['close'])
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # Bollinger Bands
    df['bb_width'], df['bb_position'] = calc_bollinger_bands(df['close'])
    
    # ATR
    df['atr_14'] = calc_atr(df['high'], df['low'], df['close'], period=14)
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    # 시간 특성
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'], unit='ms')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['session_asia'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['session_europe'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['session_us'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
    
    # 거래량/모멘텀
    df['vol_ma_10'] = df['volume'].rolling(10).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma_10']
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # 타깃
    df['y'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df = df.dropna().reset_index(drop=True)
    
    features = [
        'ret_1', 'ret_3', 'ret_5', 'hl_range', 'vol_chg', 'ret_mean_10', 'ret_std_10',
        'rsi_14', 'rsi_7',
        'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio', 'ema_cross_5_10', 'ema_cross_5_20',
        'macd_hist', 'macd_cross',
        'bb_width', 'bb_position',
        'atr_ratio',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'session_asia', 'session_europe', 'session_us',
        'vol_ratio', 'momentum_10', 'momentum_20'
    ]
    
    return df, features

def time_split(df, test_ratio=0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def evaluate_predictions(y_true, y_pred, proba, test_df, model_name):
    """예측 결과 평가"""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    thr = 0.55
    signal = np.where(proba >= thr, 1, np.where(proba <= 1 - thr, -1, 0))
    ret = test_df['close'].pct_change().fillna(0).values
    pnl = (signal[:-1] * ret[1:]).sum()
    trades = (signal != 0).sum()
    
    return {
        'name': model_name,
        'accuracy': acc,
        'balanced_acc': bal_acc,
        'trades': trades,
        'pnl': pnl
    }

def main():
    parser = argparse.ArgumentParser(description='Phase 9: 앙상블 모델')
    parser.add_argument('--timeframe', type=str, default='15m',
                        choices=['1m', '5m', '15m', '30m'],
                        help='사용할 타임프레임 (기본: 15m)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 9: 앙상블 모델 (XGBoost + CatBoost + LightGBM)")
    print("=" * 70)
    
    print(f"\n[1] 데이터 로딩 (타임프레임: {args.timeframe})...")
    df_1m = load_data(FILE_PATH, USE_ROWS)
    
    if args.timeframe == '1m':
        df = df_1m
    else:
        tf_map = {'5m': '5min', '15m': '15min', '30m': '30min'}
        df = resample_ohlcv(df_1m, tf_map[args.timeframe])
    
    print(f"    데이터 크기: {len(df):,}행")
    
    print("\n[2] 특성 공학 적용...")
    df, feats = make_features(df)
    print(f"    특성 수: {len(feats)}개")
    
    print("\n[3] 데이터 분할...")
    train, test = time_split(df, test_ratio=0.2)
    X_tr, y_tr = train[feats].values, train['y'].values
    X_te, y_te = test[feats].values, test['y'].values
    print(f"    Train: {len(train):,}행, Test: {len(test):,}행")
    
    scale_pos_weight = len(y_tr[y_tr == 0]) / len(y_tr[y_tr == 1])
    
    results = []
    models = {}
    probas = {}
    
    # ============ 개별 모델 학습 ============
    
    # XGBoost
    print("\n[4] XGBoost 학습...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', n_jobs=-1, random_state=42
    )
    model_xgb.fit(X_tr, y_tr)
    pred_xgb = model_xgb.predict(X_te)
    proba_xgb = model_xgb.predict_proba(X_te)[:, 1]
    
    result_xgb = evaluate_predictions(y_te, pred_xgb, proba_xgb, test, "XGBoost")
    results.append(result_xgb)
    models['xgb'] = model_xgb
    probas['xgb'] = proba_xgb
    print(f"    Accuracy: {result_xgb['accuracy']:.6f}, PnL: {result_xgb['pnl']:.5f}")
    
    # LightGBM
    print("\n[5] LightGBM 학습...")
    model_lgb = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced',
        n_jobs=-1, random_state=42, verbose=-1
    )
    model_lgb.fit(X_tr, y_tr)
    pred_lgb = model_lgb.predict(X_te)
    proba_lgb = model_lgb.predict_proba(X_te)[:, 1]
    
    result_lgb = evaluate_predictions(y_te, pred_lgb, proba_lgb, test, "LightGBM")
    results.append(result_lgb)
    models['lgb'] = model_lgb
    probas['lgb'] = proba_lgb
    print(f"    Accuracy: {result_lgb['accuracy']:.6f}, PnL: {result_lgb['pnl']:.5f}")
    
    # CatBoost
    print("\n[6] CatBoost 학습...")
    model_cat = CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1,
        auto_class_weights='Balanced',
        random_state=42, verbose=False
    )
    model_cat.fit(X_tr, y_tr)
    pred_cat = model_cat.predict(X_te)
    proba_cat = model_cat.predict_proba(X_te)[:, 1]
    
    result_cat = evaluate_predictions(y_te, pred_cat, proba_cat, test, "CatBoost")
    results.append(result_cat)
    models['cat'] = model_cat
    probas['cat'] = proba_cat
    print(f"    Accuracy: {result_cat['accuracy']:.6f}, PnL: {result_cat['pnl']:.5f}")
    
    # ============ 앙상블 방법 ============
    
    print("\n[7] 앙상블 모델 구성...")
    
    # 1. 단순 평균 (Simple Average)
    print("\n    [7-1] 단순 평균 앙상블...")
    proba_avg = (proba_xgb + proba_lgb + proba_cat) / 3
    pred_avg = (proba_avg >= 0.5).astype(int)
    result_avg = evaluate_predictions(y_te, pred_avg, proba_avg, test, "앙상블(단순평균)")
    results.append(result_avg)
    print(f"    Accuracy: {result_avg['accuracy']:.6f}, PnL: {result_avg['pnl']:.5f}")
    
    # 2. 가중 평균 (Weighted Average) - 개별 모델 성능 기반
    print("\n    [7-2] 가중 평균 앙상블...")
    # PnL 기반 가중치 (양수만 사용)
    pnls = np.array([max(result_xgb['pnl'], 0.01), 
                     max(result_lgb['pnl'], 0.01), 
                     max(result_cat['pnl'], 0.01)])
    weights = pnls / pnls.sum()
    print(f"    가중치: XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CAT={weights[2]:.3f}")
    
    proba_weighted = weights[0] * proba_xgb + weights[1] * proba_lgb + weights[2] * proba_cat
    pred_weighted = (proba_weighted >= 0.5).astype(int)
    result_weighted = evaluate_predictions(y_te, pred_weighted, proba_weighted, test, "앙상블(가중평균)")
    results.append(result_weighted)
    print(f"    Accuracy: {result_weighted['accuracy']:.6f}, PnL: {result_weighted['pnl']:.5f}")
    
    # 3. Soft Voting (확률 기반)
    print("\n    [7-3] Soft Voting 앙상블...")
    # 각 모델의 확률 평균
    proba_soft = (proba_xgb + proba_lgb + proba_cat) / 3
    pred_soft = (proba_soft >= 0.5).astype(int)
    result_soft = evaluate_predictions(y_te, pred_soft, proba_soft, test, "앙상블(Soft Voting)")
    results.append(result_soft)
    print(f"    Accuracy: {result_soft['accuracy']:.6f}, PnL: {result_soft['pnl']:.5f}")
    
    # 4. Hard Voting (다수결)
    print("\n    [7-4] Hard Voting 앙상블...")
    pred_hard = ((pred_xgb + pred_lgb + pred_cat) >= 2).astype(int)
    proba_hard = proba_avg  # PnL 계산용
    result_hard = evaluate_predictions(y_te, pred_hard, proba_hard, test, "앙상블(Hard Voting)")
    results.append(result_hard)
    print(f"    Accuracy: {result_hard['accuracy']:.6f}, PnL: {result_hard['pnl']:.5f}")
    
    # 5. 최고 성능 2개 모델 앙상블
    print("\n    [7-5] Top-2 모델 앙상블...")
    individual_results = [result_xgb, result_lgb, result_cat]
    sorted_by_pnl = sorted(individual_results, key=lambda x: x['pnl'], reverse=True)
    top2_names = [sorted_by_pnl[0]['name'], sorted_by_pnl[1]['name']]
    print(f"    Top-2 모델: {top2_names}")
    
    top2_probas = []
    if 'XGBoost' in top2_names:
        top2_probas.append(proba_xgb)
    if 'LightGBM' in top2_names:
        top2_probas.append(proba_lgb)
    if 'CatBoost' in top2_names:
        top2_probas.append(proba_cat)
    
    proba_top2 = np.mean(top2_probas, axis=0)
    pred_top2 = (proba_top2 >= 0.5).astype(int)
    result_top2 = evaluate_predictions(y_te, pred_top2, proba_top2, test, "앙상블(Top-2)")
    results.append(result_top2)
    print(f"    Accuracy: {result_top2['accuracy']:.6f}, PnL: {result_top2['pnl']:.5f}")
    
    # ============ 결과 요약 ============
    print("\n" + "=" * 70)
    print("앙상블 모델 성능 비교")
    print("=" * 70)
    print(f"{'모델':<25} {'Accuracy':<12} {'Bal.Acc':<12} {'PnL':<12} {'Trades':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<25} {r['accuracy']:<12.6f} {r['balanced_acc']:<12.6f} "
              f"{r['pnl']:<12.5f} {r['trades']:<10}")
    
    print("-" * 70)
    
    # 최고 성능 모델
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_pnl = max(results, key=lambda x: x['pnl'])
    
    print(f"\n[최고 Accuracy]: {best_acc['name']} ({best_acc['accuracy']:.6f})")
    print(f"[최고 PnL]: {best_pnl['name']} ({best_pnl['pnl']:.5f})")
    
    # 개별 vs 앙상블 비교
    individual_best = max(individual_results, key=lambda x: x['pnl'])
    ensemble_results = [r for r in results if '앙상블' in r['name']]
    ensemble_best = max(ensemble_results, key=lambda x: x['pnl'])
    
    print(f"\n[개별 최고]: {individual_best['name']} (PnL: {individual_best['pnl']:.5f})")
    print(f"[앙상블 최고]: {ensemble_best['name']} (PnL: {ensemble_best['pnl']:.5f})")
    
    improvement = (ensemble_best['pnl'] / individual_best['pnl'] - 1) * 100 if individual_best['pnl'] > 0 else 0
    print(f"[앙상블 개선율]: {improvement:+.1f}%")
    
    print("\n" + "=" * 70)
    print("Phase 9 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()

