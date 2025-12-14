# 파일명: ml_step4_tuning.py
# Phase 4: XGBoost 하이퍼파라미터 튜닝 (Time Series Cross Validation)
import sys
import subprocess
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
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, make_scorer
except:
    pip_install("scikit-learn")
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, make_scorer

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

def main():
    print("=" * 60)
    print("Phase 4: XGBoost 하이퍼파라미터 튜닝")
    print("=" * 60)
    
    print("\n[1] 데이터 로딩...")
    df = load_data(FILE_PATH, USE_ROWS)
    print(f"    로드된 데이터: {len(df):,}행")
    
    print("\n[2] 특성 공학...")
    df, feats = make_features(df)
    print(f"    사용 특성 수: {len(feats)}개")
    
    print("\n[3] 데이터 분할...")
    train, test = time_split(df, test_ratio=0.2)
    X_tr, y_tr = train[feats].values, train['y'].values
    X_te, y_te = test[feats].values, test['y'].values
    print(f"    Train: {len(train):,}행, Test: {len(test):,}행")
    
    # 클래스 가중치 계산
    scale_pos_weight = len(y_tr[y_tr == 0]) / len(y_tr[y_tr == 1])
    
    print("\n[4] 하이퍼파라미터 탐색 공간 정의...")
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [1, 1.5, 2, 3]
    }
    print(f"    탐색할 파라미터: {list(param_dist.keys())}")
    
    print("\n[5] Time Series Cross Validation 설정...")
    tscv = TimeSeriesSplit(n_splits=3)
    print("    CV Splits: 3 (Time Series Split)")
    
    print("\n[6] RandomizedSearchCV 실행 중... (약 2-5분 소요)")
    
    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,  # 30번의 랜덤 조합 시도
        scoring='balanced_accuracy',
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_tr, y_tr)
    
    print("\n[7] 최적 파라미터 발견!")
    print("-" * 40)
    print("Best Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"\nBest CV Score (Balanced Acc): {random_search.best_score_:.6f}")
    
    print("\n[8] 최적 모델로 Test Set 평가...")
    print("-" * 40)
    best_model = random_search.best_estimator_
    pred = best_model.predict(X_te)
    proba = best_model.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_te, pred)
    bal_acc = balanced_accuracy_score(y_te, pred)
    
    print(f"Accuracy: {acc:.6f}")
    print(f"Balanced Acc: {bal_acc:.6f}")
    print(classification_report(y_te, pred, digits=4))
    
    # PnL 계산
    thr = 0.55
    signal = np.where(proba >= thr, 1, np.where(proba <= 1 - thr, -1, 0))
    ret = test['close'].pct_change().fillna(0).values
    pnl = (signal[:-1] * ret[1:])
    print(f"Trades: {(signal != 0).sum()}, PnL(단순 합): {pnl.sum():.5f}")
    
    # Feature Importance
    print("\n[9] Feature Importance (Top 10)")
    print("-" * 40)
    importance = best_model.feature_importances_
    feat_imp = sorted(zip(feats, importance), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(feat_imp[:10]):
        print(f"    {i+1}. {feat}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("Phase 4 완료!")
    print("=" * 60)
    
    # 튜닝 전후 비교
    print("\n[튜닝 전후 비교]")
    print("-" * 40)
    print(f"{'항목':<20} {'튜닝 전':<15} {'튜닝 후':<15} {'개선':<10}")
    print("-" * 40)
    print(f"{'Accuracy':<20} {'0.5161':<15} {acc:<15.4f} {(acc-0.5161)*100:+.2f}%p")
    print(f"{'Balanced Acc':<20} {'0.5151':<15} {bal_acc:<15.4f} {(bal_acc-0.5151)*100:+.2f}%p")
    print(f"{'PnL':<20} {'0.44343':<15} {pnl.sum():<15.5f} {(pnl.sum()/0.44343-1)*100:+.1f}%")

if __name__ == "__main__":
    main()

