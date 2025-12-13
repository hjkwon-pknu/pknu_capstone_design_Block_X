import sys
import subprocess
import argparse
import warnings
from pnl_utils import calculate_pnl_from_forward_return

warnings.filterwarnings("ignore")


def pip_install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


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

try:
    import xgboost as xgb  # type: ignore
except Exception:
    pip_install("xgboost")
    import xgboost as xgb  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    pip_install("lightgbm")
    import lightgbm as lgb  # type: ignore

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:
    pip_install("catboost")
    from catboost import CatBoostClassifier  # type: ignore

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

from ml_step11_multitimeframe_features import (  # type: ignore
    build_multitimeframe_dataset,
)
from ml_step10_label_redesign import time_split  # type: ignore


def get_xgb_param_dist():
    return {
        "n_estimators": [80, 120, 160],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
    }


def get_lgb_param_dist():
    return {
        "n_estimators": [80, 120, 160],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 40],
    }


def get_cat_param_dist():
    return {
        "iterations": [80, 120, 160],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1.0, 3.0, 5.0],
        "border_count": [32, 64, 128],
    }


def build_models(scale_pos_weight: float):
    models = {}

    models["xgb"] = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    models["lgbm"] = lgb.LGBMClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    models["cat"] = CatBoostClassifier(
        iterations=120,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_state=42,
        verbose=False,
    )

    return models


def tune_model(name, model, X_tr, y_tr, n_iter: int = 12):
    if name == "xgb":
        param_dist = get_xgb_param_dist()
    elif name == "lgbm":
        param_dist = get_lgb_param_dist()
    elif name == "cat":
        param_dist = get_cat_param_dist()
    else:
        return model, None

    print(f"\n[튜닝] {name} RandomizedSearchCV 진행 (n_iter={n_iter})...")
    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    search.fit(X_tr, y_tr)

    print(f"[튜닝 완료] {name} Best Balanced Acc (CV): {search.best_score_:.6f}")
    print("Best Params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    return search.best_estimator_, search.best_score_


def eval_on_test(name, model, X_te, y_te, test_df,horizon=4, cost=0.002):
    pred = model.predict(X_te)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)[:, 1]
    else:
        proba = pred.astype(float)

    acc = accuracy_score(y_te, pred)
    bal_acc = balanced_accuracy_score(y_te, pred)

    print("\n" + "=" * 80)
    print(f"[{name}] 테스트셋 결과")
    print("=" * 80)
    print(f"Accuracy: {acc:.6f}")
    print(f"Balanced Acc: {bal_acc:.6f}")
    print(classification_report(y_te, pred, digits=4))

    thr = 0.55
    signal = np.where(proba >= thr, 1, np.where(proba <= 1 - thr, -1, 0))
    total_pnl, trades = calculate_pnl_from_forward_return(
    test_df=test_df,
    signal=signal,
    horizon= horizon,   # step12는 args를 쓰는 경우가 대부분
    cost= cost
    )

    print(f"Trades: {(signal != 0).sum():,}, PnL(단순 합): {total_pnl:.5f}")

    return acc, bal_acc, total_pnl


def main():
    parser = argparse.ArgumentParser(
        description="Phase 12: 트리 계열 모델 고도화 (시계열 CV + 튜닝)"
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
    parser.add_argument(
        "--use-mtf",
        action="store_true",
        help="1H/4H 상위 타임프레임 피처를 함께 사용할지 여부",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["xgb", "lgbm", "cat", "all"],
        help="튜닝/평가할 모델 선택",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=12,
        help="RandomizedSearchCV 반복 횟수 (기본 12)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Phase 12: 트리 계열 모델 고도화 (시계열 CV + 튜닝)")
    print("=" * 80)

    # 데이터셋/피처 구성 (Phase 11 유틸 재사용)
    df_mt, base_feats, higher_feats = build_multitimeframe_dataset(
        timeframe=args.timeframe,
        horizon=args.horizon,
        target_ret=args.target_ret,
        cost=args.cost,
    )

    if args.use_mtf:
        feats = base_feats + higher_feats
        print(f"\n[피처 구성] 베이스 + 상위 타임프레임 피처 사용 (총 {len(feats)}개)")
    else:
        feats = base_feats
        print(f"\n[피처 구성] 베이스 피처만 사용 (총 {len(feats)}개)")

    train, test = time_split(df_mt, test_ratio=0.2)
    X_tr, y_tr = train[feats].values, train["y"].values
    X_te, y_te = test[feats].values, test["y"].values

    print(f"\n[데이터 분할] Train: {len(train):,}행, Test: {len(test):,}행")

    scale_pos_weight = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)
    base_models = build_models(scale_pos_weight)

    targets = ["xgb", "lgbm", "cat"] if args.model == "all" else [args.model]
    results = {}

    for name in targets:
        print("\n" + "-" * 80)
        print(f"[{name}] 모델 학습/튜닝 시작")
        print("-" * 80)
        tuned_model, cv_score = tune_model(
            name, base_models[name], X_tr, y_tr, n_iter=args.n_iter
        )
        acc, bal_acc, pnl = eval_on_test(
            f"{name.upper()}(튜닝)", tuned_model, X_te, y_te, test,horizon=args.horizon,cost=args.cost
        )
        results[name] = {
            "cv_bal_acc": cv_score,
            "test_acc": acc,
            "test_bal_acc": bal_acc,
            "test_pnl": pnl,
        }

    # 요약 출력
    print("\n" + "=" * 80)
    print("튜닝된 트리 계열 모델 요약")
    print("=" * 80)
    print(f"{'모델':<8} {'CV BalAcc':<12} {'Test Acc':<12} {'Test BalAcc':<14} {'Test PnL':<10}")
    print("-" * 80)
    for name, r in results.items():
        cv_b = r["cv_bal_acc"] if r["cv_bal_acc"] is not None else 0.0
        print(
            f"{name.upper():<8} {cv_b:<12.6f} {r['test_acc']:<12.6f} "
            f"{r['test_bal_acc']:<14.6f} {r['test_pnl']:<10.5f}"
        )

    print("\n" + "=" * 80)
    print("Phase 12 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()


