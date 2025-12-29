import sys
import subprocess
import argparse
import warnings

warnings.filterwarnings("ignore")


def pip_install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# pandas
try:
    import pandas as pd
except Exception:
    pip_install("pandas")
    import pandas as pd

# numpy
try:
    import numpy as np
except Exception:
    pip_install("numpy")
    import numpy as np

# xgboost (옵션: 간단 성능 체크용)
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None  # 나중에 필요 시 설치 안내만 출력

#################################
import os
FILE_PATH = os.environ.get("BTC_FILE_PATH", r"C:\Users\User\binance_data\1m_history.csv")
USE_ROWS = None
#################################

def load_data(path: str, use_rows: int | None = None) -> pd.DataFrame:
    """1분봉 OHLCV 데이터 로드"""
    df = pd.read_csv(path, usecols=["datetime", "open", "high", "low", "close", "volume"])
    if use_rows is not None and len(df) > use_rows:
        df = df.tail(use_rows)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    - label='right', closed='right'로 '캔들 종료 시각'에 라벨링
    - resample index(종료시각)를 timestamp로 보존
    - datetime(ms)도 종료시각 기준으로 동기화
    """
    df = df_1m.copy()

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_localize(None)
    else:
        ts = pd.to_datetime(df["timestamp"])
        if getattr(ts.dt, "tz", None) is not None:
            df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df["timestamp"] = ts

    df = df.sort_values("timestamp").set_index("timestamp")

    resampled = (
        df.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()  # timestamp 컬럼 유지
    )

    # 종료시각 기준 ms로 datetime 동기화
    resampled["datetime"] = (resampled["timestamp"].astype("int64") // 10**6).astype("int64")
    return resampled

#######################################

def make_base_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """기존 29개 수준의 기본 특성 생성 (15분봉 기준과 동일 구조)"""
    df = df.copy()

    # 수익률/변동성/거래량
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["vol_chg"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df["ret_mean_10"] = df["ret_1"].rolling(10, min_periods=10).mean()
    df["ret_std_10"] = df["ret_1"].rolling(10, min_periods=10).std()

    # RSI
    df["rsi_14"] = calc_rsi(df["close"], period=14)
    df["rsi_7"] = calc_rsi(df["close"], period=7)

    # EMA
    df["ema_5"] = calc_ema(df["close"], span=5)
    df["ema_10"] = calc_ema(df["close"], span=10)
    df["ema_20"] = calc_ema(df["close"], span=20)
    df["ema_5_ratio"] = df["close"] / df["ema_5"] - 1
    df["ema_10_ratio"] = df["close"] / df["ema_10"] - 1
    df["ema_20_ratio"] = df["close"] / df["ema_20"] - 1
    df["ema_cross_5_10"] = (df["ema_5"] > df["ema_10"]).astype(int)
    df["ema_cross_5_20"] = (df["ema_5"] > df["ema_20"]).astype(int)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(df["close"])
    df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

    # Bollinger Bands
    df["bb_width"], df["bb_position"] = calc_bollinger_bands(df["close"])

    # ATR
    df["atr_14"] = calc_atr(df["high"], df["low"], df["close"], period=14)
    df["atr_ratio"] = df["atr_14"] / df["close"]

    # 시간 특성
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["session_asia"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
    df["session_europe"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
    df["session_us"] = ((df["hour"] >= 16) & (df["hour"] < 24)).astype(int)

    # 거래량/모멘텀
    df["vol_ma_10"] = df["volume"].rolling(10).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma_10"]
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    df = df.dropna().reset_index(drop=True)

    features = [
        "ret_1",
        "ret_3",
        "ret_5",
        "hl_range",
        "vol_chg",
        "ret_mean_10",
        "ret_std_10",
        "rsi_14",
        "rsi_7",
        "ema_5_ratio",
        "ema_10_ratio",
        "ema_20_ratio",
        "ema_cross_5_10",
        "ema_cross_5_20",
        "macd_hist",
        "macd_cross",
        "bb_width",
        "bb_position",
        "atr_ratio",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "session_asia",
        "session_europe",
        "session_us",
        "vol_ratio",
        "momentum_10",
        "momentum_20",
    ]

    return df, features


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series]:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    bb_width = (upper_band - lower_band) / sma
    bb_position = (series - lower_band) / (upper_band - lower_band)
    return bb_width, bb_position


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def make_directional_label(
    df: pd.DataFrame,
    horizon: int = 4,
    target_return: float = 0.004,
    cost: float = 0.002,
    mode: str = "filter",
) -> pd.DataFrame:
    """
    거래 비용을 고려한 방향성 라벨 생성.

    - horizon: 몇 개 캔들 뒤의 종가 기준으로 수익률 계산 (예: 4 → 약 1시간 후, 15분봉 기준)
    - target_return: 최소 목표 수익률 (예: 0.004 → 0.4%)
    - cost: 왕복 거래 비용(수수료+슬리피지) 가정치 (예: 0.002 → 0.2%)
    - mode:
        - 'filter': |ret| >= cost + target_return 인 구간만 남기고, y ∈ {0,1} (Up/Down) 이진 라벨
        - 'triple': y3 ∈ {-1,0,1} (Down/Flat/Up) 3클래스 라벨 생성 (기존 y는 그대로 두고 y3 추가)
    """
    df = df.copy()

    # 미래 horizon 캔들의 종가 기준 수익률
    future_close = df["close"].shift(-horizon)
    fwd_ret = future_close / df["close"] - 1
    df["fwd_ret_h{}".format(horizon)] = fwd_ret

    # 비용 + 목표 수익률 합산 임계값
    threshold = cost + target_return

    if mode == "filter":
        mask = fwd_ret.abs() >= threshold
        before = len(df)
        df = df[mask].copy()
        df["y"] = (df["fwd_ret_h{}".format(horizon)] > 0).astype(int)
        df = df.dropna().reset_index(drop=True)

        print(f"[라벨링] horizon={horizon}, target={target_return:.4f}, cost={cost:.4f}")
        print(f"  - 전체 샘플: {before:,} → 필터 후: {len(df):,} ({len(df)/before*100:.1f}%)")
        print("  - y=0(Down) / y=1(Up) 분포:")
        print(df["y"].value_counts().sort_index())
        print(df["y"].value_counts(normalize=True).sort_index().round(4))

    elif mode == "triple":
        labels = np.zeros(len(df), dtype=int)
        labels[fwd_ret >= threshold] = 1
        labels[fwd_ret <= -threshold] = -1
        df["y3"] = labels
        df = df.dropna().reset_index(drop=True)

        print(f"[라벨링] horizon={horizon}, target={target_return:.4f}, cost={cost:.4f} (3클래스)")
        print("  - y3=-1(Down), 0(Flat), 1(Up) 분포:")
        print(df["y3"].value_counts().sort_index())
        print(df["y3"].value_counts(normalize=True).sort_index().round(4))

    else:
        raise ValueError("mode must be 'filter' or 'triple'")

    return df


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * (1 - test_ratio))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test

def quick_xgb_check(df: pd.DataFrame, features: list[str], label_col: str = "y") -> None:
    """새 라벨로 간단한 XGBoost 성능을 체크하는 유틸 (선택 실행)."""
    if xgb is None:
        print("\n[XGBoost 미설치] xgboost 패키지가 없어 간단 성능 체크는 건너뜁니다.")
        print("필요하다면: pip install xgboost 후 다시 실행하세요.")
        return

    train, test = time_split(df, test_ratio=0.2)
    X_tr, y_tr = train[features].values, train[label_col].values
    X_te, y_te = test[features].values, test[label_col].values

    scale_pos_weight = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, pred)
    bal_acc = balanced_accuracy_score(y_te, pred)

    print("\n[Quick XGBoost] 새 라벨 기준 성능")
    print(f"  Accuracy: {acc:.6f}")
    print(f"  Balanced Acc: {bal_acc:.6f}")
    print(classification_report(y_te, pred, digits=4))

    # 거래 비용을 반영한 간단 PnL (라벨 설계에서 이미 비용을 반영했으므로 보수적 해석)
    thr = 0.55
    signal = np.where(proba >= thr, 1, np.where(proba <= 1 - thr, -1, 0))
    ret = test["close"].pct_change().fillna(0).values
    pnl = (signal[:-1] * ret[1:])
    print(f"  Trades: {(signal != 0).sum():,}, PnL(단순 합): {pnl.sum():.5f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 10: 타깃/라벨 재설계 실험")
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        choices=["1m", "5m", "15m", "30m"],
        help="사용할 타임프레임 (기본: 15m)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="앞으로 몇 개 캔들의 누적 수익률을 볼지 (기본: 4 → 15m 기준 약 1시간)",
    )
    parser.add_argument(
        "--target-ret",
        type=float,
        default=0.004,
        help="목표 수익률 (예: 0.004 → 0.4%)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.002,
        help="왕복 거래 비용(수수료+슬리피지) 가정치 (예: 0.002 → 0.2%)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="filter",
        choices=["filter", "triple"],
        help="라벨 방식: filter(고정 이진, Flat 구간 제거) / triple(3클래스)",
    )
    parser.add_argument(
        "--quick-xgb",
        action="store_true",
        help="새 라벨로 간단한 XGBoost 성능 체크를 진행할지 여부",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Phase 10: 타깃/라벨 재설계 (거래 비용 반영)")
    print("=" * 80)

    print(f"\n[1] 1분봉 데이터 로딩...")
    df_1m = load_data(FILE_PATH, USE_ROWS)
    print(f"    로드된 데이터: {len(df_1m):,}행")

    # 타임프레임 변환
    if args.timeframe == "1m":
        df_tf = df_1m
        rule_str = "1T"
    else:
        tf_map = {"5m": "5T", "15m": "15T", "30m": "30T"}
        rule_str = tf_map[args.timeframe]
        print(f"\n[2] 리샘플링: 1m → {args.timeframe}")
        df_tf = resample_ohlcv(df_1m, rule_str)

    print(f"    타임프레임 변환 후 데이터: {len(df_tf):,}행 ({args.timeframe})")

    print("\n[3] 기본 특성 생성...")
    df_feat, feats = make_base_features(df_tf)
    print(f"    사용 특성 수: {len(feats)}개")

    print("\n[4] 거래 비용/목표 수익률을 반영한 라벨 생성...")
    df_labeled = make_directional_label(
        df_feat,
        horizon=args.horizon,
        target_return=args.target_ret,
        cost=args.cost,
        mode=args.mode,
    )

    # 라벨 품질 추가 통계
    print("\n[5] 라벨 품질 추가 통계 (forward return 기준)")
    col_ret = f"fwd_ret_h{args.horizon}"
    print(
        df_labeled[col_ret].describe(
            percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        )
    )

    # 간단 모델 체크 옵션
    if args.mode == "filter" and args.quick_xgb:
        print("\n[6] 새 라벨로 간단 XGBoost 성능 체크...")
        quick_xgb_check(df_labeled, feats, label_col="y")

    print("\n" + "=" * 80)
    print("Phase 10 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()


