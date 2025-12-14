# 파일명: ml_step7_regime.py
# Phase 8: 레짐 분류 (횡보/트렌드 구분) 모델
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

def calc_adx(high, low, close, period=14):
    """ADX (Average Directional Index) - 추세 강도 측정"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = calc_atr(high, low, close, period)
    
    plus_di = 100 * calc_ema(plus_dm, period) / atr
    minus_di = 100 * calc_ema(minus_dm, period) / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = calc_ema(dx, period)
    
    return adx, plus_di, minus_di

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
    
    # ADX (레짐 분류용)
    df['adx'], df['plus_di'], df['minus_di'] = calc_adx(df['high'], df['low'], df['close'])
    df['di_cross'] = (df['plus_di'] > df['minus_di']).astype(int)
    
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
    
    # 레짐 분류 (ADX 기반)
    # ADX < 25: 횡보장, ADX >= 25: 추세장
    df['regime'] = (df['adx'] >= 25).astype(int)  # 0: 횡보, 1: 추세
    
    df = df.dropna().reset_index(drop=True)
    
    features = [
        'ret_1', 'ret_3', 'ret_5', 'hl_range', 'vol_chg', 'ret_mean_10', 'ret_std_10',
        'rsi_14', 'rsi_7',
        'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio', 'ema_cross_5_10', 'ema_cross_5_20',
        'macd_hist', 'macd_cross',
        'bb_width', 'bb_position',
        'atr_ratio',
        'adx', 'plus_di', 'minus_di', 'di_cross',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'session_asia', 'session_europe', 'session_us',
        'vol_ratio', 'momentum_10', 'momentum_20'
    ]
    
    return df, features

def time_split(df, test_ratio=0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def train_xgboost(X_tr, y_tr):
    """XGBoost 모델 학습"""
    scale_pos_weight = len(y_tr[y_tr == 0]) / (len(y_tr[y_tr == 1]) + 1e-10)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    return model

def evaluate_model(model, X_te, y_te, test_df, model_name):
    """모델 평가"""
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_te, pred)
    bal_acc = balanced_accuracy_score(y_te, pred)
    
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
        'pnl': pnl,
        'predictions': pred,
        'probabilities': proba
    }

def main():
    parser = argparse.ArgumentParser(description='Phase 8: 레짐 분류 모델')
    parser.add_argument('--timeframe', type=str, default='15m',
                        choices=['1m', '5m', '15m', '30m'],
                        help='사용할 타임프레임 (기본: 15m)')
    parser.add_argument('--adx_threshold', type=float, default=25,
                        help='ADX 임계값 (기본: 25)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 8: 레짐 분류 (횡보/트렌드 구분) 모델")
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
    
    # ADX 임계값 조정
    df['regime'] = (df['adx'] >= args.adx_threshold).astype(int)
    
    # 레짐 분포 확인
    regime_counts = df['regime'].value_counts()
    print(f"\n[3] 레짐 분포 (ADX 임계값: {args.adx_threshold})")
    print(f"    횡보장 (ADX < {args.adx_threshold}): {regime_counts.get(0, 0):,}행 ({regime_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"    추세장 (ADX >= {args.adx_threshold}): {regime_counts.get(1, 0):,}행 ({regime_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    print("\n[4] 데이터 분할...")
    train, test = time_split(df, test_ratio=0.2)
    print(f"    Train: {len(train):,}행, Test: {len(test):,}행")
    
    # 테스트셋 레짐 분포
    test_regime_counts = test['regime'].value_counts()
    print(f"    Test 횡보장: {test_regime_counts.get(0, 0):,}행")
    print(f"    Test 추세장: {test_regime_counts.get(1, 0):,}행")
    
    # ============ 단일 모델 (기준선) ============
    print("\n[5] 단일 모델 (전체 데이터) 학습...")
    X_tr_all = train[feats].values
    y_tr_all = train['y'].values
    X_te_all = test[feats].values
    y_te_all = test['y'].values
    
    model_single = train_xgboost(X_tr_all, y_tr_all)
    result_single = evaluate_model(model_single, X_te_all, y_te_all, test, "단일 모델")
    
    print(f"    Accuracy: {result_single['accuracy']:.6f}")
    print(f"    Balanced Acc: {result_single['balanced_acc']:.6f}")
    print(f"    Trades: {result_single['trades']:,}, PnL: {result_single['pnl']:.5f}")
    
    # ============ 레짐별 모델 ============
    print("\n[6] 레짐별 전용 모델 학습...")
    
    # 횡보장 전용 모델
    train_sideways = train[train['regime'] == 0]
    test_sideways = test[test['regime'] == 0]
    
    if len(train_sideways) > 100 and len(test_sideways) > 10:
        print(f"\n    [횡보장 모델] Train: {len(train_sideways):,}행, Test: {len(test_sideways):,}행")
        X_tr_sw = train_sideways[feats].values
        y_tr_sw = train_sideways['y'].values
        X_te_sw = test_sideways[feats].values
        y_te_sw = test_sideways['y'].values
        
        model_sideways = train_xgboost(X_tr_sw, y_tr_sw)
        result_sideways = evaluate_model(model_sideways, X_te_sw, y_te_sw, test_sideways, "횡보장 모델")
        
        print(f"    Accuracy: {result_sideways['accuracy']:.6f}")
        print(f"    Balanced Acc: {result_sideways['balanced_acc']:.6f}")
        print(f"    Trades: {result_sideways['trades']:,}, PnL: {result_sideways['pnl']:.5f}")
    else:
        result_sideways = None
        print("    [횡보장 모델] 데이터 부족으로 스킵")
    
    # 추세장 전용 모델
    train_trend = train[train['regime'] == 1]
    test_trend = test[test['regime'] == 1]
    
    if len(train_trend) > 100 and len(test_trend) > 10:
        print(f"\n    [추세장 모델] Train: {len(train_trend):,}행, Test: {len(test_trend):,}행")
        X_tr_tr = train_trend[feats].values
        y_tr_tr = train_trend['y'].values
        X_te_tr = test_trend[feats].values
        y_te_tr = test_trend['y'].values
        
        model_trend = train_xgboost(X_tr_tr, y_tr_tr)
        result_trend = evaluate_model(model_trend, X_te_tr, y_te_tr, test_trend, "추세장 모델")
        
        print(f"    Accuracy: {result_trend['accuracy']:.6f}")
        print(f"    Balanced Acc: {result_trend['balanced_acc']:.6f}")
        print(f"    Trades: {result_trend['trades']:,}, PnL: {result_trend['pnl']:.5f}")
    else:
        result_trend = None
        print("    [추세장 모델] 데이터 부족으로 스킵")
    
    # ============ 메타 모델 (레짐에 따라 예측 선택) ============
    print("\n[7] 메타 모델 (레짐 기반 선택) 평가...")
    
    if result_sideways and result_trend:
        # 테스트셋에서 레짐에 따라 다른 모델의 예측 사용
        test_indices_sw = test[test['regime'] == 0].index
        test_indices_tr = test[test['regime'] == 1].index
        
        # 메타 예측 결합
        meta_pred = np.zeros(len(test))
        meta_proba = np.zeros(len(test))
        
        # 횡보장 구간은 횡보장 모델 사용
        sw_mask = test['regime'] == 0
        tr_mask = test['regime'] == 1
        
        meta_pred[sw_mask.values] = model_sideways.predict(test[sw_mask][feats].values)
        meta_proba[sw_mask.values] = model_sideways.predict_proba(test[sw_mask][feats].values)[:, 1]
        
        meta_pred[tr_mask.values] = model_trend.predict(test[tr_mask][feats].values)
        meta_proba[tr_mask.values] = model_trend.predict_proba(test[tr_mask][feats].values)[:, 1]
        
        # 메타 모델 평가
        acc_meta = accuracy_score(y_te_all, meta_pred)
        bal_acc_meta = balanced_accuracy_score(y_te_all, meta_pred)
        
        thr = 0.55
        signal_meta = np.where(meta_proba >= thr, 1, np.where(meta_proba <= 1 - thr, -1, 0))
        ret = test['close'].pct_change().fillna(0).values
        pnl_meta = (signal_meta[:-1] * ret[1:]).sum()
        trades_meta = (signal_meta != 0).sum()
        
        print(f"    Accuracy: {acc_meta:.6f}")
        print(f"    Balanced Acc: {bal_acc_meta:.6f}")
        print(f"    Trades: {trades_meta:,}, PnL: {pnl_meta:.5f}")
        
        result_meta = {
            'accuracy': acc_meta,
            'balanced_acc': bal_acc_meta,
            'trades': trades_meta,
            'pnl': pnl_meta
        }
    else:
        result_meta = None
        print("    메타 모델 구성 불가 (레짐별 모델 부족)")
    
    # ============ 결과 요약 ============
    print("\n" + "=" * 70)
    print("레짐 분류 모델 비교 요약")
    print("=" * 70)
    print(f"{'모델':<15} {'Accuracy':<12} {'Bal.Acc':<12} {'PnL':<12} {'Trades':<10}")
    print("-" * 70)
    
    print(f"{'단일 모델':<15} {result_single['accuracy']:<12.6f} {result_single['balanced_acc']:<12.6f} "
          f"{result_single['pnl']:<12.5f} {result_single['trades']:<10}")
    
    if result_sideways:
        print(f"{'횡보장 전용':<15} {result_sideways['accuracy']:<12.6f} {result_sideways['balanced_acc']:<12.6f} "
              f"{result_sideways['pnl']:<12.5f} {result_sideways['trades']:<10}")
    
    if result_trend:
        print(f"{'추세장 전용':<15} {result_trend['accuracy']:<12.6f} {result_trend['balanced_acc']:<12.6f} "
              f"{result_trend['pnl']:<12.5f} {result_trend['trades']:<10}")
    
    if result_meta:
        print(f"{'메타 모델':<15} {result_meta['accuracy']:<12.6f} {result_meta['balanced_acc']:<12.6f} "
              f"{result_meta['pnl']:<12.5f} {result_meta['trades']:<10}")
    
    # 최고 성능 모델 선택
    print("-" * 70)
    all_results = [('단일 모델', result_single['pnl'])]
    if result_sideways:
        all_results.append(('횡보장 전용', result_sideways['pnl']))
    if result_trend:
        all_results.append(('추세장 전용', result_trend['pnl']))
    if result_meta:
        all_results.append(('메타 모델', result_meta['pnl']))
    
    best_model = max(all_results, key=lambda x: x[1])
    print(f"[최고 PnL 모델]: {best_model[0]} (PnL: {best_model[1]:.5f})")
    
    print("\n" + "=" * 70)
    print("Phase 8 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()

