import sys
import subprocess
import argparse
import warnings
#########
from pnl_utils import calculate_pnl_from_forward_return

warnings.filterwarnings("ignore")


def pip_install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# pandas / numpy
try:
    import pandas as pd
except Exception:
    pip_install("pandas")
    import pandas as pd

try:
    import numpy as np
except Exception:
    pip_install("numpy")
    import numpy as np

# xgboost
try:
    import xgboost as xgb  # type: ignore
except Exception:
    pip_install("xgboost")
    import xgboost as xgb  # type: ignore

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

# 기존 라벨/기본 피처 생성 유틸 재사용
from ml_step10_label_redesign import (  # type: ignore
    load_data,
    resample_ohlcv,
    make_base_features,
    make_directional_label,
    time_split,
)

#################################
import os
FILE_PATH = os.environ.get("BTC_FILE_PATH", r"C:\Users\User\binance_data\1m_history.csv")
USE_ROWS = None
#################################

def make_tf_features(df_tf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    상위 타임프레임용 간단 피처 (EMA20 비율, RSI, ATR 비율, 모멘텀 등) 생성.
    prefix 예: '1h', '4h'
    """
    df = df_tf.copy()

    # 기본 수익률
    df[f"ret_1_{prefix}"] = df["close"].pct_change(1)
    df[f"ret_3_{prefix}"] = df["close"].pct_change(3)

    # EMA / RSI / ATR
    df[f"ema_20_{prefix}"] = df["close"].ewm(span=20, adjust=False).mean()
    df[f"ema_20_ratio_{prefix}"] = df["close"] / df[f"ema_20_{prefix}"] - 1

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f"rsi_14_{prefix}"] = 100 - (100 / (1 + rs))

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    df[f"atr_ratio_{prefix}"] = atr / df["close"]

    # 간단 모멘텀
    df[f"momentum_4_{prefix}"] = df["close"] / df["close"].shift(4) - 1

    # 타임스탬프 (merge_asof 용)
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms")

    cols = [
        "timestamp",
        f"ret_1_{prefix}",
        f"ret_3_{prefix}",
        f"ema_20_ratio_{prefix}",
        f"rsi_14_{prefix}",
        f"atr_ratio_{prefix}",
        f"momentum_4_{prefix}",
    ]
    df = df[cols].dropna().reset_index(drop=True)
    return df


def build_multitimeframe_dataset(
    timeframe: str = "15m",
    horizon: int = 4,
    target_ret: float = 0.004,
    cost: float = 0.002,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    15m(또는 선택한 타임프레임) 기준으로:
      - 기존 29개 피처
      - 상위 타임프레임(1H, 4H) 피처를 merge_asof로 붙인 데이터셋 생성.
    """
    print(f"[1] 1분봉 데이터 로딩...")
    df_1m = load_data(FILE_PATH, USE_ROWS)
    print(f"    로드된 데이터: {len(df_1m):,}행")

    # 베이스 타임프레임 리샘플링
    if timeframe == "1m":
        df_base_raw = df_1m
        base_rule = "1T"
    else:
        tf_map = {"5m": "5T", "15m": "15T", "30m": "30T"}
        base_rule = tf_map[timeframe]
        print(f"\n[2] 리샘플링: 1m → {timeframe}")
        df_base_raw = resample_ohlcv(df_1m, base_rule)

    print(f"    베이스 타임프레임 데이터: {len(df_base_raw):,}행 ({timeframe})")

    # 베이스 피처 + 라벨
    print("\n[3] 베이스 피처 생성...")
    df_base_feat, base_feats = make_base_features(df_base_raw)

    print("\n[4] 거래 비용/목표 수익률 라벨 생성 (베이스 타임프레임 기준)...")
    df_labeled = make_directional_label(
        df_base_feat,
        horizon=horizon,
        target_return=target_ret,
        cost=cost,
        mode="filter",
    )

    # timestamp 준비
    if "timestamp" not in df_labeled.columns:
        df_labeled["timestamp"] = pd.to_datetime(df_labeled["datetime"], unit="ms")

    # 상위 타임프레임 1H, 4H 리샘플링 및 피처
    print("\n[5] 상위 타임프레임 피처 생성 (1H, 4H)...")
    df_1h_raw = resample_ohlcv(df_1m, "1H")
    df_4h_raw = resample_ohlcv(df_1m, "4H")

    df_1h_feat = make_tf_features(df_1h_raw, prefix="1h")
    df_4h_feat = make_tf_features(df_4h_raw, prefix="4h")

    # merge_asof로 상위 타임프레임 피처를 베이스에 붙이기
    df_labeled = df_labeled.sort_values("timestamp").reset_index(drop=True)
    df_1h_feat = df_1h_feat.sort_values("timestamp").reset_index(drop=True)
    df_4h_feat = df_4h_feat.sort_values("timestamp").reset_index(drop=True)


    print("    1H 피처 merge_asof...")
    df_mt = pd.merge_asof(
        df_labeled,
        df_1h_feat,
        on="timestamp",
        direction="backward",
    )

    print("    4H 피처 merge_asof...")
    df_mt = pd.merge_asof(
        df_mt,
        df_4h_feat,
        on="timestamp",
        direction="backward",
        suffixes=("", "_dup"),
    )

    # 필요 없는 dup 컬럼 제거
    dup_cols = [c for c in df_mt.columns if c.endswith("_dup")]
    if dup_cols:
        df_mt = df_mt.drop(columns=dup_cols)

    # 결측 제거
    before = len(df_mt)
    df_mt = df_mt.dropna().reset_index(drop=True)
    print(f"\n[6] merge 후 결측 제거: {before:,} → {len(df_mt):,}")

    higher_feats = [
        c
        for c in df_mt.columns
        if c.endswith("_1h") or c.endswith("_4h")
    ]
    print(f"    상위 타임프레임 피처 수: {len(higher_feats)}개")

    return df_mt, base_feats, higher_feats


def run_xgb_experiments(
    df: pd.DataFrame,
    base_feats: list[str],
    higher_feats: list[str],
    label_col: str = "y",  horizon=4, cost=0.002
) -> None:
    """
    - 기본 29개 피처만 사용
    - 기본 + 상위 타임프레임 피처 모두 사용
    두 설정에 대해 XGBoost 성능/간단 PnL 비교.
    """
    train, test = time_split(df, test_ratio=0.2)
    y_tr = train[label_col].values
    y_te = test[label_col].values

    # 공통 설정
    scale_pos_weight = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)

    def fit_and_eval(feats: list[str], name: str) -> None:
        X_tr = train[feats].values
        X_te = test[feats].values

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

        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_te, pred)
        bal_acc = balanced_accuracy_score(y_te, pred)

        print("\n" + "=" * 70)
        print(f"[{name}] 결과")
        print("=" * 70)
        print(f"Accuracy: {acc:.6f}")
        print(f"Balanced Acc: {bal_acc:.6f}")
        print(classification_report(y_te, pred, digits=4))

        thr = 0.55
        signal = np.where(proba >= thr, 1, np.where(proba <= 1 - thr, -1, 0))
        total_pnl, trades = calculate_pnl_from_forward_return(
            test_df=test,
            signal=signal,
            horizon= horizon,   # step11에 horizon 변수가 있으면 그대로
            cost= cost
            )          # step11에 cost 변수가 있으면 그대로
        print(f"Trades: {trades:,}, PnL(단순 합): {total_pnl:.5f}")

    # 1) 기본 피처만
    fit_and_eval(base_feats, "기본 피처(베이스 타임프레임만)")

    # 2) 기본 + 상위 타임프레임
    full_feats = base_feats + higher_feats
    fit_and_eval(full_feats, "기본 + 상위 타임프레임 피처")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 11: 다중 타임프레임 피처 + 피처 재구성 실험"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        choices=["1m", "5m", "15m", "30m"],
        help="베이스 타임프레임 (기본: 15m)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="앞으로 몇 개 캔들의 누적 수익률을 볼지 (기본: 4)",
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
    args = parser.parse_args()

    print("=" * 80)
    print("Phase 11: 다중 타임프레임 피처 + 피처 재구성 실험")
    print("=" * 80)

    df_mt, base_feats, higher_feats = build_multitimeframe_dataset(
        timeframe=args.timeframe,
        horizon=args.horizon,
        target_ret=args.target_ret,
        cost=args.cost,
    )

    print("\n[7] XGBoost로 기본 vs 다중 타임프레임 피처 비교...")
    run_xgb_experiments(df_mt, base_feats, higher_feats, label_col="y",horizon=args.horizon, cost=args.cost)

    print("\n" + "=" * 80)
    print("Phase 11 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()


