# 파일명: ml_step2_features.py
# Phase 2: 기술적 지표 및 시간 특성 추가
import sys
import subprocess

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# pandas 자동 설치
try:
    import pandas as pd
except:
    pip_install("pandas")
    import pandas as pd

# numpy 자동 설치
try:
    import numpy as np
except:
    pip_install("numpy")
    import numpy as np

# scikit-learn 자동 설치
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
except:
    pip_install("scikit-learn")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

FILE_PATH = r"C:\Users\User\binance_data\1m_history.csv"
USE_ROWS = 1_000_000

def load_data(path, use_rows=None):
    df = pd.read_csv(path, usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    if use_rows is not None and len(df) > use_rows:
        df = df.tail(use_rows)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# ============ 기술적 지표 함수 ============

def calc_rsi(series, period=14):
    """RSI (Relative Strength Index) 계산"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_ema(series, span):
    """EMA (Exponential Moving Average) 계산"""
    return series.ewm(span=span, adjust=False).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calc_bollinger_bands(series, period=20, std_dev=2):
    """Bollinger Bands 계산"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    bb_width = (upper_band - lower_band) / sma  # 밴드 폭 비율
    bb_position = (series - lower_band) / (upper_band - lower_band)  # 밴드 내 위치 (0~1)
    return bb_width, bb_position

def calc_atr(high, low, close, period=14):
    """ATR (Average True Range) 계산"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# ============ 특성 공학 ============

def make_features(df):
    # === 기본 특성 (Step 1과 동일) ===
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['vol_chg'] = df['volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df['ret_mean_10'] = df['ret_1'].rolling(10, min_periods=10).mean()
    df['ret_std_10'] = df['ret_1'].rolling(10, min_periods=10).std()
    
    # === 기술적 지표 추가 ===
    
    # RSI
    df['rsi_14'] = calc_rsi(df['close'], period=14)
    df['rsi_7'] = calc_rsi(df['close'], period=7)
    
    # EMA
    df['ema_5'] = calc_ema(df['close'], span=5)
    df['ema_10'] = calc_ema(df['close'], span=10)
    df['ema_20'] = calc_ema(df['close'], span=20)
    
    # EMA 비율 (현재가 대비)
    df['ema_5_ratio'] = df['close'] / df['ema_5'] - 1
    df['ema_10_ratio'] = df['close'] / df['ema_10'] - 1
    df['ema_20_ratio'] = df['close'] / df['ema_20'] - 1
    
    # EMA 크로스오버 신호
    df['ema_cross_5_10'] = (df['ema_5'] > df['ema_10']).astype(int)
    df['ema_cross_5_20'] = (df['ema_5'] > df['ema_20']).astype(int)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(df['close'])
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # Bollinger Bands
    df['bb_width'], df['bb_position'] = calc_bollinger_bands(df['close'])
    
    # ATR (변동성 지표)
    df['atr_14'] = calc_atr(df['high'], df['low'], df['close'], period=14)
    df['atr_ratio'] = df['atr_14'] / df['close']  # 가격 대비 ATR 비율
    
    # === 시간 특성 추가 ===
    # datetime을 timestamp로 변환 후 시간 추출
    df['timestamp'] = pd.to_datetime(df['datetime'], unit='ms')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # 주기적 시간 특성 (sin/cos 인코딩)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 거래 시간대 구분 (아시아/유럽/미국)
    df['session_asia'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['session_europe'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['session_us'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
    
    # === 추가 파생 특성 ===
    # 거래량 이동평균 대비
    df['vol_ma_10'] = df['volume'].rolling(10).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma_10']
    
    # 가격 모멘텀
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # 타깃: 다음 1분이 오르면 1, 내리면 0
    df['y'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 결측치 제거
    df = df.dropna().reset_index(drop=True)
    
    # 특성 목록
    features = [
        # 기본 특성
        'ret_1', 'ret_3', 'ret_5', 'hl_range', 'vol_chg', 'ret_mean_10', 'ret_std_10',
        # RSI
        'rsi_14', 'rsi_7',
        # EMA
        'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio', 'ema_cross_5_10', 'ema_cross_5_20',
        # MACD
        'macd_hist', 'macd_cross',
        # Bollinger Bands
        'bb_width', 'bb_position',
        # ATR
        'atr_ratio',
        # 시간 특성
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'session_asia', 'session_europe', 'session_us',
        # 거래량/모멘텀
        'vol_ratio', 'momentum_10', 'momentum_20'
    ]
    
    return df, features

def time_split(df, test_ratio=0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test

def main():
    print("=" * 60)
    print("Phase 2: 기술적 지표 및 시간 특성 추가")
    print("=" * 60)
    
    print("\n[1] 데이터 로딩...")
    df = load_data(FILE_PATH, USE_ROWS)
    print(f"    로드된 데이터: {len(df):,}행")
    
    print("\n[2] 특성 공학...")
    df, feats = make_features(df)
    print(f"    사용 특성 수: {len(feats)}개")
    print(f"    특성 목록: {feats}")
    
    print("\n[3] 데이터 분할...")
    train, test = time_split(df, test_ratio=0.2)
    X_tr, y_tr = train[feats].values, train['y'].values
    X_te, y_te = test[feats].values, test['y'].values
    print(f"    Train: {len(train):,}행, Test: {len(test):,}행")
    
    print("\n[4] Logistic Regression 학습...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=-1))
    ])
    pipe.fit(X_tr, y_tr)
    
    print("\n[5] 평가 결과")
    print("-" * 40)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_te, pred):.6f}")
    print(f"Balanced Acc: {balanced_accuracy_score(y_te, pred):.6f}")
    print(classification_report(y_te, pred, digits=4))
    
    # 간단 PnL
    thr = 0.55
    signal = np.where(proba >= thr, 1, np.where(proba <= 1 - thr, -1, 0))
    ret = test['close'].pct_change().fillna(0).values
    pnl = (signal[:-1] * ret[1:])
    print(f"Trades: {(signal != 0).sum()}, PnL(단순 합): {pnl.sum():.5f}")
    
    print("\n" + "=" * 60)
    print("Phase 2 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()

