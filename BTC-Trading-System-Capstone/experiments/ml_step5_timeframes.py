# 파일명: ml_step5_timeframes.py
# Phase 6: 시간 프레임 확장 비교 (1m, 5m, 15m, 30m)
import sys
import subprocess
import argparse
import warnings
warnings.filterwarnings('ignore')

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# pandas
try:
    import pandas as pd
except:
    pip_install("pandas")
    import pandas as pd

# numpy
try:
    import numpy as np
except:
    pip_install("numpy")
    import numpy as np

# scikit-learn
try:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
except:
    pip_install("scikit-learn")
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

# xgboost
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

def resample_ohlcv(df, timeframe):
    """1분봉 데이터를 다른 타임프레임으로 리샘플링"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['datetime'], unit='ms')
    df = df.set_index('timestamp')
    
    # 리샘플링 규칙
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

# ============ 기술적 지표 함수 ============

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calc_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    bb_width = (upper_band - lower_band) / sma
    bb_position = (series - lower_band) / (upper_band - lower_band)
    return bb_width, bb_position

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

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
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test

def run_xgboost(X_tr, y_tr, X_te, y_te, test_df, timeframe_name):
    """XGBoost 모델 실행 및 평가"""
    scale_pos_weight = len(y_tr[y_tr == 0]) / len(y_tr[y_tr == 1])
    
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
    
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_te, pred)
    bal_acc = balanced_accuracy_score(y_te, pred)
    
    # PnL
    thr = 0.55
    signal = np.where(proba >= thr, 1, np.where(proba <= 1 - thr, -1, 0))
    ret = test_df['close'].pct_change().fillna(0).values
    pnl = (signal[:-1] * ret[1:])
    
    # 추가 메트릭: 승률, 샤프 비율
    winning_trades = pnl[pnl > 0]
    losing_trades = pnl[pnl < 0]
    win_rate = len(winning_trades) / len(pnl[pnl != 0]) if len(pnl[pnl != 0]) > 0 else 0
    
    # 단순 샤프 비율 (연간화 X)
    sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
    
    return {
        'timeframe': timeframe_name,
        'accuracy': acc,
        'balanced_acc': bal_acc,
        'trades': (signal != 0).sum(),
        'pnl': pnl.sum(),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'data_size': len(test_df)
    }

def main():
    parser = argparse.ArgumentParser(description='Phase 6: 시간 프레임 비교')
    parser.add_argument('--timeframe', type=str, default='all',
                        choices=['1m', '5m', '15m', '30m', 'all'],
                        help='테스트할 타임프레임')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 6: 시간 프레임 확장 비교")
    print("=" * 70)
    
    print("\n[1] 1분봉 데이터 로딩...")
    df_1m = load_data(FILE_PATH, USE_ROWS)
    print(f"    로드된 데이터: {len(df_1m):,}행")
    
    # 타임프레임 설정
    timeframes = {
        '1m': ('1min', df_1m),
        '5m': ('5min', None),
        '15m': ('15min', None),
        '30m': ('30min', None)
    }
    
    if args.timeframe != 'all':
        timeframes = {args.timeframe: timeframes[args.timeframe]}
    
    results = []
    
    for tf_name, (resample_rule, data) in timeframes.items():
        print(f"\n{'='*70}")
        print(f"[{tf_name}] 타임프레임 처리 중...")
        print("=" * 70)
        
        # 리샘플링
        if tf_name == '1m':
            df = df_1m.copy()
        else:
            print(f"    리샘플링: 1m → {tf_name}")
            df = resample_ohlcv(df_1m, resample_rule)
        
        print(f"    데이터 크기: {len(df):,}행")
        
        # 특성 공학
        print(f"    특성 공학 적용 중...")
        df, feats = make_features(df)
        
        # 데이터 분할
        train, test = time_split(df, test_ratio=0.2)
        X_tr, y_tr = train[feats].values, train['y'].values
        X_te, y_te = test[feats].values, test['y'].values
        print(f"    Train: {len(train):,}행, Test: {len(test):,}행")
        
        # XGBoost 실행
        print(f"    XGBoost 학습 중...")
        result = run_xgboost(X_tr, y_tr, X_te, y_te, test, tf_name)
        results.append(result)
        
        # 결과 출력
        print(f"\n    [결과]")
        print(f"    Accuracy: {result['accuracy']:.6f}")
        print(f"    Balanced Acc: {result['balanced_acc']:.6f}")
        print(f"    Trades: {result['trades']:,}")
        print(f"    PnL: {result['pnl']:.5f}")
        print(f"    Win Rate: {result['win_rate']:.2%}")
        print(f"    Sharpe Ratio: {result['sharpe']:.4f}")
    
    # 종합 비교
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("타임프레임별 성능 비교 요약")
        print("=" * 70)
        print(f"{'타임프레임':<10} {'Accuracy':<12} {'Bal.Acc':<12} {'PnL':<12} {'Win Rate':<12} {'Sharpe':<10}")
        print("-" * 70)
        
        best_pnl = max(results, key=lambda x: x['pnl'])
        best_acc = max(results, key=lambda x: x['accuracy'])
        
        for r in results:
            pnl_mark = " ★" if r == best_pnl else ""
            acc_mark = " ◆" if r == best_acc else ""
            print(f"{r['timeframe']:<10} {r['accuracy']:<12.6f} {r['balanced_acc']:<12.6f} "
                  f"{r['pnl']:<12.5f}{pnl_mark} {r['win_rate']:<12.2%} {r['sharpe']:<10.4f}{acc_mark}")
        
        print("-" * 70)
        print("★ = 최고 PnL, ◆ = 최고 Accuracy")
        
        # 추천 타임프레임
        print(f"\n[추천] PnL 기준 최적 타임프레임: {best_pnl['timeframe']}")
        print(f"[추천] Accuracy 기준 최적 타임프레임: {best_acc['timeframe']}")
    
    print("\n" + "=" * 70)
    print("Phase 6 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()

