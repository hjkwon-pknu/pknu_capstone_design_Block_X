# 파일명: ml_step6_advanced_features.py
# Phase 7: 특징 엔지니어링 강화 (MFI, CCI, OBV, VWAP, Stochastic, 캔들패턴)
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

import os
FILE_PATH = os.environ.get("BTC_FILE_PATH", r"C:\Users\User\binance_data\1m_history.csv")

USE_ROWS = 1_000_000

def load_data(path, use_rows=None):
    df = pd.read_csv(path, usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    if use_rows is not None and len(df) > use_rows:
        df = df.tail(use_rows)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def resample_ohlcv(df, timeframe='15min'):
    """1분봉 데이터를 다른 타임프레임으로 리샘플링"""
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
    
    resampled = resampled.reset_index(drop=True)
    return resampled

# ============ 기본 기술적 지표 ============

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_sma(series, period):
    return series.rolling(window=period).mean()

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

# ============ 고급 기술적 지표 (새로 추가) ============

def calc_mfi(high, low, close, volume, period=14):
    """MFI (Money Flow Index) - 거래량 가중 RSI"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0).rolling(period).sum()
    negative_flow = money_flow.where(delta < 0, 0).rolling(period).sum()
    
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    return mfi

def calc_cci(high, low, close, period=20):
    """CCI (Commodity Channel Index)"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def calc_obv(close, volume):
    """OBV (On Balance Volume)"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calc_obv_change(close, volume, period=10):
    """OBV 변화율"""
    obv = calc_obv(close, volume)
    return obv.pct_change(period).replace([np.inf, -np.inf], 0).fillna(0)

def calc_vwap(high, low, close, volume):
    """VWAP (Volume Weighted Average Price)"""
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    vwap = cumulative_tp_vol / cumulative_vol
    return vwap

def calc_vwap_deviation(close, vwap):
    """VWAP 이탈 정도"""
    return (close - vwap) / vwap

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d

def calc_williams_r(high, low, close, period=14):
    """Williams %R"""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def calc_adx(high, low, close, period=14):
    """ADX (Average Directional Index) - 추세 강도"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = calc_atr(high, low, close, period)
    
    plus_di = 100 * calc_ema(plus_dm, period) / atr
    minus_di = 100 * calc_ema(minus_dm, period) / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = calc_ema(dx, period)
    
    return adx, plus_di, minus_di

# ============ 캔들 패턴 ============

def detect_doji(open_p, high, low, close, threshold=0.1):
    """도지 캔들 (몸통이 매우 작음)"""
    body = abs(close - open_p)
    total_range = high - low
    return (body / total_range < threshold).astype(int)

def detect_hammer(open_p, high, low, close):
    """해머 패턴 (긴 아래꼬리, 작은 몸통, 짧은 윗꼬리)"""
    body = abs(close - open_p)
    total_range = high - low
    lower_shadow = np.minimum(open_p, close) - low
    upper_shadow = high - np.maximum(open_p, close)
    
    is_hammer = (
        (lower_shadow > 2 * body) &
        (upper_shadow < body) &
        (body < total_range * 0.3)
    )
    return is_hammer.astype(int)

def detect_engulfing(open_p, high, low, close):
    """엔갈핑 패턴"""
    prev_body = (close.shift(1) - open_p.shift(1))
    curr_body = (close - open_p)
    
    # Bullish engulfing
    bullish = (
        (prev_body < 0) &  # 이전 캔들 음봉
        (curr_body > 0) &  # 현재 캔들 양봉
        (open_p < close.shift(1)) &  # 현재 시가가 이전 종가보다 낮음
        (close > open_p.shift(1))    # 현재 종가가 이전 시가보다 높음
    )
    
    # Bearish engulfing
    bearish = (
        (prev_body > 0) &  # 이전 캔들 양봉
        (curr_body < 0) &  # 현재 캔들 음봉
        (open_p > close.shift(1)) &  # 현재 시가가 이전 종가보다 높음
        (close < open_p.shift(1))    # 현재 종가가 이전 시가보다 낮음
    )
    
    # 1: bullish, -1: bearish, 0: none
    return bullish.astype(int) - bearish.astype(int)

# ============ Lag Features ============

def add_lag_features(df, columns, lags=[1, 2, 3, 5]):
    """과거 N기간 값을 특성으로 추가"""
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

# ============ 특성 공학 (강화 버전) ============

def make_advanced_features(df):
    """기존 29개 + 고급 특성 추가 = 50개+ 특성"""
    
    # === 기본 특성 (기존) ===
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
    
    # === 고급 특성 (새로 추가) ===
    
    # MFI (Money Flow Index)
    df['mfi_14'] = calc_mfi(df['high'], df['low'], df['close'], df['volume'], period=14)
    
    # CCI (Commodity Channel Index)
    df['cci_20'] = calc_cci(df['high'], df['low'], df['close'], period=20)
    
    # OBV 변화율
    df['obv_change_10'] = calc_obv_change(df['close'], df['volume'], period=10)
    
    # VWAP
    df['vwap'] = calc_vwap(df['high'], df['low'], df['close'], df['volume'])
    df['vwap_deviation'] = calc_vwap_deviation(df['close'], df['vwap'])
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = calc_stochastic(df['high'], df['low'], df['close'])
    df['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
    
    # Williams %R
    df['williams_r'] = calc_williams_r(df['high'], df['low'], df['close'])
    
    # ADX (추세 강도)
    df['adx'], df['plus_di'], df['minus_di'] = calc_adx(df['high'], df['low'], df['close'])
    df['di_cross'] = (df['plus_di'] > df['minus_di']).astype(int)
    
    # 캔들 패턴
    df['doji'] = detect_doji(df['open'], df['high'], df['low'], df['close'])
    df['hammer'] = detect_hammer(df['open'], df['high'], df['low'], df['close'])
    df['engulfing'] = detect_engulfing(df['open'], df['high'], df['low'], df['close'])
    
    # 캔들 특성
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Lag Features (수익률과 거래량의 과거 값)
    df = add_lag_features(df, ['ret_1', 'vol_chg'], lags=[1, 2, 3])
    
    # 타깃
    df['y'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 결측치 제거
    df = df.dropna().reset_index(drop=True)
    
    # 특성 목록 (기존 29개 + 새로운 특성들)
    features = [
        # 기본 특성 (29개)
        'ret_1', 'ret_3', 'ret_5', 'hl_range', 'vol_chg', 'ret_mean_10', 'ret_std_10',
        'rsi_14', 'rsi_7',
        'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio', 'ema_cross_5_10', 'ema_cross_5_20',
        'macd_hist', 'macd_cross',
        'bb_width', 'bb_position',
        'atr_ratio',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'session_asia', 'session_europe', 'session_us',
        'vol_ratio', 'momentum_10', 'momentum_20',
        # 고급 특성 (새로 추가)
        'mfi_14', 'cci_20', 'obv_change_10',
        'vwap_deviation',
        'stoch_k', 'stoch_d', 'stoch_cross',
        'williams_r',
        'adx', 'plus_di', 'minus_di', 'di_cross',
        'doji', 'hammer', 'engulfing',
        'body_ratio', 'upper_shadow', 'lower_shadow',
        'ret_1_lag_1', 'ret_1_lag_2', 'ret_1_lag_3',
        'vol_chg_lag_1', 'vol_chg_lag_2', 'vol_chg_lag_3'
    ]
    
    return df, features

def time_split(df, test_ratio=0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def main():
    parser = argparse.ArgumentParser(description='Phase 7: 고급 특성 엔지니어링')
    parser.add_argument('--timeframe', type=str, default='15m',
                        choices=['1m', '5m', '15m', '30m'],
                        help='사용할 타임프레임 (기본: 15m)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 7: 특징 엔지니어링 강화")
    print("=" * 70)
    
    print(f"\n[1] 데이터 로딩 (타임프레임: {args.timeframe})...")
    df_1m = load_data(FILE_PATH, USE_ROWS)
    
    # 리샘플링
    if args.timeframe == '1m':
        df = df_1m
    else:
        tf_map = {'5m': '5min', '15m': '15min', '30m': '30min'}
        df = resample_ohlcv(df_1m, tf_map[args.timeframe])
    
    print(f"    데이터 크기: {len(df):,}행")
    
    print("\n[2] 고급 특성 공학 적용...")
    df, feats = make_advanced_features(df)
    print(f"    총 특성 수: {len(feats)}개")
    
    # 특성 그룹별 분류
    basic_feats = feats[:29]
    advanced_feats = feats[29:]
    print(f"    - 기본 특성: {len(basic_feats)}개")
    print(f"    - 고급 특성: {len(advanced_feats)}개")
    print(f"    고급 특성 목록: {advanced_feats}")
    
    print("\n[3] 데이터 분할...")
    train, test = time_split(df, test_ratio=0.2)
    X_tr, y_tr = train[feats].values, train['y'].values
    X_te, y_te = test[feats].values, test['y'].values
    print(f"    Train: {len(train):,}행, Test: {len(test):,}행")
    
    # 기존 특성으로 학습 (비교용)
    print("\n[4] 기본 특성(29개)으로 XGBoost 학습...")
    scale_pos_weight = len(y_tr[y_tr == 0]) / len(y_tr[y_tr == 1])
    
    model_basic = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', n_jobs=-1, random_state=42
    )
    X_tr_basic = train[basic_feats].values
    X_te_basic = test[basic_feats].values
    model_basic.fit(X_tr_basic, y_tr)
    
    pred_basic = model_basic.predict(X_te_basic)
    proba_basic = model_basic.predict_proba(X_te_basic)[:, 1]
    
    acc_basic = accuracy_score(y_te, pred_basic)
    bal_acc_basic = balanced_accuracy_score(y_te, pred_basic)
    
    thr = 0.55
    signal_basic = np.where(proba_basic >= thr, 1, np.where(proba_basic <= 1 - thr, -1, 0))
    ret = test['close'].pct_change().fillna(0).values
    pnl_basic = (signal_basic[:-1] * ret[1:]).sum()
    
    print(f"    Accuracy: {acc_basic:.6f}")
    print(f"    Balanced Acc: {bal_acc_basic:.6f}")
    print(f"    PnL: {pnl_basic:.5f}")
    
    # 고급 특성 포함하여 학습
    print(f"\n[5] 전체 특성({len(feats)}개)으로 XGBoost 학습...")
    model_advanced = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', n_jobs=-1, random_state=42
    )
    model_advanced.fit(X_tr, y_tr)
    
    pred_adv = model_advanced.predict(X_te)
    proba_adv = model_advanced.predict_proba(X_te)[:, 1]
    
    acc_adv = accuracy_score(y_te, pred_adv)
    bal_acc_adv = balanced_accuracy_score(y_te, pred_adv)
    
    signal_adv = np.where(proba_adv >= thr, 1, np.where(proba_adv <= 1 - thr, -1, 0))
    pnl_adv = (signal_adv[:-1] * ret[1:]).sum()
    
    print(f"    Accuracy: {acc_adv:.6f}")
    print(f"    Balanced Acc: {bal_acc_adv:.6f}")
    print(classification_report(y_te, pred_adv, digits=4))
    print(f"    Trades: {(signal_adv != 0).sum()}, PnL: {pnl_adv:.5f}")
    
    # Feature Importance
    print("\n[6] Feature Importance (Top 15)")
    print("-" * 50)
    importance = model_advanced.feature_importances_
    feat_imp = sorted(zip(feats, importance), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(feat_imp[:15]):
        marker = " (신규)" if feat in advanced_feats else ""
        print(f"    {i+1:2d}. {feat:<20}: {imp:.4f}{marker}")
    
    # 비교 요약
    print("\n" + "=" * 70)
    print("기본 vs 고급 특성 비교")
    print("=" * 70)
    print(f"{'항목':<20} {'기본 (29개)':<15} {'고급 ({0}개)':<15} {'변화':<10}".format(len(feats)))
    print("-" * 70)
    print(f"{'Accuracy':<20} {acc_basic:<15.6f} {acc_adv:<15.6f} {(acc_adv-acc_basic)*100:+.2f}%p")
    print(f"{'Balanced Acc':<20} {bal_acc_basic:<15.6f} {bal_acc_adv:<15.6f} {(bal_acc_adv-bal_acc_basic)*100:+.2f}%p")
    print(f"{'PnL':<20} {pnl_basic:<15.5f} {pnl_adv:<15.5f} {(pnl_adv/pnl_basic-1)*100 if pnl_basic != 0 else 0:+.1f}%")
    
    print("\n" + "=" * 70)
    print("Phase 7 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()

