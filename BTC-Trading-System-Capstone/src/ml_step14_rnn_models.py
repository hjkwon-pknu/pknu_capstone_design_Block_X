import sys
import subprocess
import argparse
import warnings
import random

#################################
import os
FILE_PATH = os.environ.get("BTC_FILE_PATH", r"C:\Users\User\binance_data\1m_history.csv")
USE_ROWS = None
#################################

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

# torch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise ImportError(
        "PyTorch is required for Phase 14. "
        "In Colab: Runtime > Change runtime type > GPU."
    ) from e

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

import ml_step11_multitimeframe_features as step11  # type: ignore
from ml_step10_label_redesign import time_split  # type: ignore

# Ensure FILE_PATH uses Colab env var if provided
step11.FILE_PATH = os.environ.get("BTC_FILE_PATH", getattr(step11, "FILE_PATH", ""))

from pnl_utils import make_signal_from_proba, sweep_thresholds, pnl_details

def buy_and_hold_baseline(timeframe, start_ts, end_ts, cost):
    import numpy as np
    import pandas as pd

    # 1) 필터링 없는 base 15m 시계열 구성
    df_1m = step11.load_data(step11.FILE_PATH, step11.USE_ROWS)

    if timeframe == "1m":
        base = df_1m.copy()
        base["timestamp"] = pd.to_datetime(base["datetime"], unit="ms")
    else:
        rule_map = {"5m": "5T", "15m": "15T", "30m": "30T"}
        base = step11.resample_ohlcv(df_1m, rule_map[timeframe])

    base = base.sort_values("timestamp")
    base = base[(base["timestamp"] >= start_ts) & (base["timestamp"] <= end_ts)].reset_index(drop=True)

    # 2) Buy&Hold 성과(복리 기준)
    close = base["close"].to_numpy()
    if len(close) < 2:
        return {"name": "Buy&Hold", "best_thr": np.nan, "test_acc": np.nan, "test_bal_acc": np.nan,
                "test_pnl": 0.0, "test_trades": 1, "test_sharpe_trades": 0.0, "test_mdd": 0.0}

    rets = close[1:] / close[:-1] - 1.0
    equity = np.cumprod(1.0 + rets)  # 시작 1.0
    total_return = float(equity[-1] - 1.0 - cost)  # 진입+청산 비용을 1회로 단순 반영

    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    mdd = float(dd.min())

    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(len(rets)))  # “구간 내” 샤프(연환산 아님)

    return {
        "name": "Buy&Hold",
        "best_thr": np.nan,
        "test_acc": np.nan,
        "test_bal_acc": np.nan,
        "test_pnl": total_return,
        "test_trades": 1,
        "test_sharpe_trades": sharpe,
        "test_mdd": mdd,
    }


def make_sequences(df: pd.DataFrame, feature_cols: list[str], window: int):
    X_list, y_list = [], []
    Xv = df[feature_cols].values.astype(np.float32)
    yv = df["y"].values.astype(np.int64)
    for i in range(window - 1, len(df)):
        X_list.append(Xv[i - window + 1 : i + 1])
        y_list.append(yv[i])
    return np.asarray(X_list), np.asarray(y_list)


def build_sequence_datasets_with_windows(
    timeframe: str,
    horizon: int,
    target_ret: float,
    cost: float,
    use_mtf: bool,
    window: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
):
    df_mt, base_feats, higher_feats = step11.build_multitimeframe_dataset(
        timeframe=timeframe,
        horizon=horizon,
        target_ret=target_ret,
        cost=cost,
    )

    feats = base_feats + higher_feats if use_mtf else base_feats

    train_df, test_df = time_split(df_mt, test_ratio=test_ratio)

    mu = train_df[feats].mean()
    sigma = train_df[feats].std().replace(0, 1.0)

    train_norm = train_df.copy()
    test_norm = test_df.copy()
    train_norm[feats] = (train_norm[feats] - mu) / sigma
    test_norm[feats] = (test_norm[feats] - mu) / sigma

    train_in, val_df = time_split(train_norm, test_ratio=val_ratio)

    X_tr, y_tr = make_sequences(train_in, feats, window)
    X_val, y_val = make_sequences(val_df, feats, window)
    X_te, y_te = make_sequences(test_norm, feats, window)

    val_df_win = val_df.iloc[window - 1 :].reset_index(drop=True)
    test_df_win = test_df.iloc[window - 1 :].reset_index(drop=True)

    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te), val_df_win, test_df_win, feats


class RNNClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, rnn_type: str = "lstm", num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        logits = self.head(last).squeeze(-1)
        return logits


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def train_rnn(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 3,
):
    model.to(device)

    n_pos = max(int((y_tr == 1).sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_bal = -1.0
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        proba_val = predict_proba(model, X_val, device=device)
        pred_val = (proba_val >= 0.5).astype(int)
        val_bal = balanced_accuracy_score(y_val, pred_val)
        val_acc = accuracy_score(y_val, pred_val)

        print(f"[Epoch {ep:02d}] TrainLoss={np.mean(losses):.4f}, ValAcc={val_acc:.4f}, ValBalAcc={val_bal:.4f}")

        if val_bal > best_val_bal + 1e-6:
            best_val_bal = val_bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_bal


def eval_and_thr_sweep(
    name: str,
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_df_win: pd.DataFrame,
    X_te: np.ndarray,
    y_te: np.ndarray,
    test_df_win: pd.DataFrame,
    horizon: int,
    cost: float,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    non_overlapping: bool,
    device: torch.device,
    out_dir: str,
    min_trades : int,
):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    proba_te = predict_proba(model, X_te, device=device)
    pred_te = (proba_te >= 0.5).astype(int)
    acc = accuracy_score(y_te, pred_te)
    bal = balanced_accuracy_score(y_te, pred_te)

    print("\n" + "=" * 80)
    print(f"[{name}] 테스트셋 분류 성능 (threshold=0.5)")
    print("=" * 80)
    print(f"Accuracy: {acc:.6f}")
    print(f"Balanced Acc: {bal:.6f}")
    print(classification_report(y_te, pred_te, digits=4))

    thr_grid = np.arange(thr_min, thr_max + 1e-12, thr_step).tolist()

    proba_val = predict_proba(model, X_val, device=device)
    # 1) Raw sweep (제약 없음) - 그래프/참고용
    sweep_val_raw = sweep_thresholds(
        val_df_win,
        proba_val,
        thr_grid,
        horizon=horizon,
        cost=cost,
        non_overlapping=non_overlapping,
        min_trades=0,
    )

    # 2) Constrained sweep (min_trades 적용) - 선택용
    sweep_val = sweep_thresholds(
        val_df_win,
        proba_val,
        thr_grid,
        horizon=horizon,
        cost=cost,
        non_overlapping=non_overlapping,
        min_trades=min_trades,
    )

    sweep_val_raw.to_csv(os.path.join(out_dir, f"{name}_thr_sweep_val_raw.csv"), index=False)
    sweep_val.to_csv(os.path.join(out_dir, f"{name}_thr_sweep_val.csv"), index=False)

    # 3) Best thr 선택 (no-trade 규칙 포함)
    best_thr = None
    if len(sweep_val) > 0:
        top_pnl = float(sweep_val.loc[0, "pnl"])
        if top_pnl > 0:
            best_thr = float(sweep_val.loc[0, "thr"])

      ##############################
    if best_thr is None:
        # No-Trade (검증에서 기대수익<=0 이거나, min_trades 조건을 만족하는 thr이 없음)
        sig_te = np.zeros_like(proba_te, dtype=int)
        det_te = pnl_details(
            test_df_win,
            sig_te,
            horizon=horizon,
            cost=cost,
            non_overlapping=non_overlapping,
        )
        print("\n" + "-" * 80)
        print(f"[{name}] Thr Sweep 결과: NO-TRADE 선택 (min_trades={min_trades})")
        print(f"Test    : pnl={det_te['equity'][-1]:.5f}, trades={det_te['trades']}, win_rate={det_te['win_rate']:.3f}, sharpe={det_te['sharpe_trades']:.3f}, mdd={det_te['mdd']:.5f}")
        print("-" * 80)
    else:
        sig_te = make_signal_from_proba(proba_te, best_thr)
        det_te = pnl_details(
            test_df_win,
            sig_te,
            horizon=horizon,
            cost=cost,
            non_overlapping=non_overlapping,
        )
        print("\n" + "-" * 80)
        print(f"[{name}] Thr Sweep (Val 기준 BestThr={best_thr:.2f}, min_trades={min_trades})")
        print(f"Val Top-1: pnl={float(sweep_val.loc[0,'pnl']):.5f}, trades={int(sweep_val.loc[0,'trades'])}, sharpe={float(sweep_val.loc[0,'sharpe_trades']):.3f}, mdd={float(sweep_val.loc[0,'mdd']):.5f}")
        print(f"Test    : pnl={det_te['equity'][-1]:.5f}, trades={det_te['trades']}, win_rate={det_te['win_rate']:.3f}, sharpe={det_te['sharpe_trades']:.3f}, mdd={det_te['mdd']:.5f}")
        print("-" * 80)

    # sig_te = make_signal_from_proba(proba_te, best_thr)
    # det_te = pnl_details(
    #     test_df_win,
    #     sig_te,
    #     horizon=horizon,
    #     cost=cost,
    #     non_overlapping=non_overlapping,
    # )

    # print("\n" + "-" * 80)
    # print(f"[{name}] Thr Sweep (Val 기준 BestThr={best_thr:.2f})")
    # print(f"Val Top-1: pnl={sweep_val.loc[0,'pnl']:.5f}, trades={int(sweep_val.loc[0,'trades'])}, sharpe={sweep_val.loc[0,'sharpe_trades']:.3f}, mdd={sweep_val.loc[0,'mdd']:.5f}")
    # print(f"Test    : pnl={det_te['equity'][-1]:.5f}, trades={det_te['trades']}, win_rate={det_te['win_rate']:.3f}, sharpe={det_te['sharpe_trades']:.3f}, mdd={det_te['mdd']:.5f}")
    # print("-" * 80)

######
    plot_df = sweep_val_raw.sort_values("thr")
    plt.figure()
    plt.plot(plot_df["thr"].values, plot_df["pnl"].values)
#####
    # plt.figure()
    # plt.plot(sweep_val["thr"].values, sweep_val["pnl"].values)
    plt.xlabel("thr")
    plt.ylabel("PnL (val)")
    plt.title(f"{name} - Threshold Sweep (Validation)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_thr_sweep_val.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(det_te["equity"])
    plt.xlabel("time index (test window)")
    plt.ylabel("equity (cumsum pnl)")

    thr_label = "NO-TRADE" if best_thr is None else f"{best_thr:.2f}"
    plt.title(f"{name} - Equity Curve (Test, thr={thr_label})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_equity_test.png"), dpi=150)
    plt.close()


    return {
        "name": name,
        "best_thr": best_thr,
        "test_acc": float(acc),
        "test_bal_acc": float(bal),
        "test_pnl": float(det_te["equity"][-1]) if len(det_te["equity"]) else 0.0,
        "test_trades": int(det_te["trades"]),
        "test_sharpe_trades": float(det_te["sharpe_trades"]),
        "test_mdd": float(det_te["mdd"]),
    }

def buyhold_baseline(test_df_win: pd.DataFrame, cost: float, out_dir: str):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    close = test_df_win["close"].astype(float).values
    # 바-바(bar-to-bar) 수익률(단순) 누적: 전략 equity(cumsum pnl)와 비교 목적
    ret = pd.Series(close).pct_change().fillna(0.0).values
    equity = np.cumsum(ret)

    # cost는 step10에서 "왕복 거래 비용" 가정치입니다. :contentReference[oaicite:4]{index=4}
    # Buy&Hold는 진입/청산 1회라고 보고, 전체 equity를 cost만큼 한 번 아래로 shift
    equity = equity - cost

    # MDD
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    mdd = float(dd.min()) if len(dd) else 0.0

    # Sharpe(바 단위) - 표준편차 0 방지
    rstd = float(np.std(ret))
    sharpe = float(np.mean(ret) / rstd * np.sqrt(len(ret))) if rstd > 0 else 0.0

    pnl = float(equity[-1]) if len(equity) else 0.0

    plt.figure()
    plt.plot(equity)
    plt.xlabel("time index (test window)")
    plt.ylabel("equity (cumsum pnl)")
    plt.title(f"Buy&Hold - Equity Curve (Test, cost={cost})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "BuyHold_equity_test.png"), dpi=150)
    plt.close()

    return {
        "name": "Buy&Hold",
        "best_thr": None,
        "test_acc": np.nan,
        "test_bal_acc": np.nan,
        "test_pnl": pnl,
        "test_trades": 1,
        "test_sharpe_trades": sharpe,
        "test_mdd": mdd,
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 14: LSTM/GRU 시퀀스 모델 + Thr Sweep 자동화")
    parser.add_argument("--timeframe", type=str, default="15m", choices=["1m", "5m", "15m", "30m"])
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--target-ret", type=float, default=0.004)
    parser.add_argument("--cost", type=float, default=0.002)
    parser.add_argument("--use-mtf", action="store_true")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--thr-min", type=float, default=0.50)
    parser.add_argument("--thr-max", type=float, default=0.70)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--non-overlapping", action="store_true", help="horizon 보유기간 겹침 방지 PnL (권장)")
    parser.add_argument("--out-dir", type=str, default="reports")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ## 시드 재현성
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    print("=" * 80)
    print("Phase 14: LSTM/GRU 시퀀스 모델 + Thr Sweep 자동화")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[학습 장치] {device}")

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), val_df_win, test_df_win, feats = build_sequence_datasets_with_windows(
        timeframe=args.timeframe,
        horizon=args.horizon,
        target_ret=args.target_ret,
        cost=args.cost,
        use_mtf=args.use_mtf,
        window=args.window,
    )

    n_features = len(feats)
    print(f"[시퀀스 피처] use_mtf={args.use_mtf}, 총 {n_features}개")
    print(f"[시퀀스 크기] Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    results.append(buyhold_baseline(test_df_win, cost=args.cost, out_dir=args.out_dir))

    start_ts = test_df_win["timestamp"].iloc[0]
    end_ts   = test_df_win["timestamp"].iloc[-1]
    results.append(buy_and_hold_baseline(args.timeframe, start_ts, end_ts, args.cost))


    lstm = RNNClassifier(n_features=n_features, hidden=args.hidden, rnn_type="lstm", num_layers=args.layers, dropout=args.dropout)
    lstm, best_val_bal = train_rnn(lstm, X_tr, y_tr, X_val, y_val, device=device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, patience=args.patience)
    print(f"[LSTM] 최고 Val Balanced Acc: {best_val_bal:.6f}")
    results.append(
        eval_and_thr_sweep(
            name="LSTM",
            model=lstm,
            X_val=X_val,
            y_val=y_val,
            val_df_win=val_df_win,
            X_te=X_te,
            y_te=y_te,
            test_df_win=test_df_win,
            horizon=args.horizon,
            cost=args.cost,
            thr_min=args.thr_min,
            thr_max=args.thr_max,
            thr_step=args.thr_step,
            non_overlapping=args.non_overlapping,
            device=device,
            out_dir=args.out_dir,
            min_trades=args.min_trades,

        )
    )

    gru = RNNClassifier(n_features=n_features, hidden=args.hidden, rnn_type="gru", num_layers=args.layers, dropout=args.dropout)
    gru, best_val_bal = train_rnn(gru, X_tr, y_tr, X_val, y_val, device=device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, patience=args.patience)
    print(f"[GRU] 최고 Val Balanced Acc: {best_val_bal:.6f}")
    results.append(
        eval_and_thr_sweep(
            name="GRU",
            model=gru,
            X_val=X_val,
            y_val=y_val,
            val_df_win=val_df_win,
            X_te=X_te,
            y_te=y_te,
            test_df_win=test_df_win,
            horizon=args.horizon,
            cost=args.cost,
            thr_min=args.thr_min,
            thr_max=args.thr_max,
            thr_step=args.thr_step,
            non_overlapping=args.non_overlapping,
            device=device,
            out_dir=args.out_dir,
            min_trades=args.min_trades,
        )
    )

    df_res = pd.DataFrame(results)
    os.makedirs(args.out_dir, exist_ok=True)
    df_res.to_csv(f"{args.out_dir}/phase14_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("Phase 14 요약 (CSV 저장: reports/phase14_summary.csv)")
    print("=" * 80)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
