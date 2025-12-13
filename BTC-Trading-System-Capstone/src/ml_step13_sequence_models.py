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
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    pip_install("torch")
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from ml_step11_multitimeframe_features import (  # type: ignore
    build_multitimeframe_dataset,
)
from ml_step10_label_redesign import time_split  # type: ignore


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1DClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)  # (B, hidden)
        out = self.fc(x).squeeze(-1)  # (B,)
        return out


def make_sequences(df: pd.DataFrame, feats, window: int):
    """
    시계열 구간에서 슬라이딩 윈도우 시퀀스를 생성.
    마지막 시점의 라벨(y)을 시퀀스 라벨로 사용.
    """
    X_list = []
    y_list = []
    for i in range(window - 1, len(df)):
        window_slice = df.iloc[i - window + 1 : i + 1]
        X_list.append(window_slice[feats].values.astype("float32"))
        y_list.append(df["y"].iloc[i])
    X = np.stack(X_list)  # (N, window, F)
    y = np.array(y_list, dtype="float32")
    return X, y


def build_sequence_datasets(
    timeframe: str = "15m",
    horizon: int = 4,
    target_ret: float = 0.004,
    cost: float = 0.002,
    use_mtf: bool = True,
    window: int = 32,
):
    """
    Phase 11의 다중 타임프레임 데이터셋을 기반으로
    슬라이딩 윈도우 시퀀스(3D 텐서)를 생성.
    """
    df_mt, base_feats, higher_feats = build_multitimeframe_dataset(
        timeframe=timeframe,
        horizon=horizon,
        target_ret=target_ret,
        cost=cost,
    )

    feats = base_feats + higher_feats if use_mtf else base_feats
    print(f"[시퀀스 피처] {'베이스+상위TF' if use_mtf else '베이스만'} 사용, 총 {len(feats)}개")

    # 시계열 분할 (먼저 시간순으로 쪼갠 후, 각 안에서 시퀀스 생성)
    train_df, test_df = time_split(df_mt, test_ratio=0.2)

    # 정규화(Train 기반) – 피처별 평균/표준편차
    mu = train_df[feats].mean()
    sigma = train_df[feats].std().replace(0, 1.0)

    train_df_norm = train_df.copy()
    test_df_norm = test_df.copy()
    train_df_norm[feats] = (train_df_norm[feats] - mu) / sigma
    test_df_norm[feats] = (test_df_norm[feats] - mu) / sigma

    # Train 안에서 다시 Train/Val 분할 (마지막 20%를 Val)
    train_inner_df, val_df = time_split(train_df_norm, test_ratio=0.2)

    print(f"[시퀀스 분할] Train: {len(train_inner_df):,}행, Val: {len(val_df):,}행, Test: {len(test_df_norm):,}행")

    X_tr, y_tr = make_sequences(train_inner_df, feats, window)
    X_val, y_val = make_sequences(val_df, feats, window)
    X_te, y_te = make_sequences(test_df_norm, feats, window)

    print(f"[시퀀스 크기] Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    return (
        (X_tr, y_tr),
        (X_val, y_val),
        (X_te, y_te),
        test_df.iloc[window - 1 :].reset_index(drop=True),  # PnL 계산용 정렬된 Test 구간
        feats,
    )


def train_cnn1d(
    X_tr,
    y_tr,
    X_val,
    y_val,
    n_features: int,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[학습 장치] {device}")

    train_ds = SeqDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = SeqDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CNN1DClassifier(n_features=n_features, hidden=64).to(device)

    # 클래스 비율 기반 pos_weight
    pos_ratio = y_tr.mean()
    pos_weight = (1.0 - pos_ratio) / max(pos_ratio, 1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_balacc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)

        # 검증
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(yb.numpy())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        proba = 1 / (1 + np.exp(-all_logits))
        pred = (proba >= 0.5).astype(int)

        val_acc = accuracy_score(all_labels, pred)
        val_bal = balanced_accuracy_score(all_labels, pred)

        print(f"[Epoch {epoch:02d}] TrainLoss={avg_loss:.4f}, ValAcc={val_acc:.4f}, ValBalAcc={val_bal:.4f}")

        if val_bal > best_val_balacc:
            best_val_balacc = val_bal
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_balacc


def eval_sequence_model(model, X_te, y_te, test_df_window: pd.DataFrame, horizon=4, cost=0.002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    ds = SeqDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(yb.numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    proba = 1 / (1 + np.exp(-all_logits))
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(all_labels, pred)
    bal_acc = balanced_accuracy_score(all_labels, pred)

    print("\n" + "=" * 80)
    print("[CNN1D 시퀀스 모델] 테스트셋 결과")
    print("=" * 80)
    print(f"Accuracy: {acc:.6f}")
    print(f"Balanced Acc: {bal_acc:.6f}")
    print(classification_report(all_labels, pred, digits=4))

    # 간단 PnL 계산 (이전과 동일 로직, thr=0.55)
    thr = 0.55
    signal = np.where(proba >= thr, 1, np.where(proba <= 1 - thr, -1, 0))

    total_pnl, trades = calculate_pnl_from_forward_return(
      test_df=test_df_window,
      signal=signal,
      horizon=horizon,
      cost=cost
    )

    print(f"Trades: {trades:,}, PnL(단순 합): {total_pnl:.5f}")
    return acc, bal_acc, total_pnl

def main():
    parser = argparse.ArgumentParser(
        description="Phase 13: 시퀀스 모델(CNN1D) 실험 (슬라이딩 윈도우)"
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
        "--window",
        type=int,
        default=32,
        help="슬라이딩 윈도우 길이 (기본: 32 시점)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="학습 epoch 수 (기본: 10)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Phase 13: 시퀀스 모델(CNN1D) 실험 (슬라이딩 윈도우)")
    print("=" * 80)

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), test_df_win, feats = build_sequence_datasets(
        timeframe=args.timeframe,
        horizon=args.horizon,
        target_ret=args.target_ret,
        cost=args.cost,
        use_mtf=args.use_mtf,
        window=args.window,
    )

    n_features = len(feats)
    model, best_val_bal = train_cnn1d(
        X_tr,
        y_tr,
        X_val,
        y_val,
        n_features=n_features,
        epochs=args.epochs,
    )
    print(f"\n[최종] 최고 Val Balanced Acc: {best_val_bal:.6f}")

    acc, bal_acc, pnl = eval_sequence_model(model, X_te, y_te, test_df_win, horizon=args.horizon,cost=args.cost)

    print("\n" + "=" * 80)
    print("시퀀스 모델 요약")
    print("=" * 80)
    print(f"Test Acc: {acc:.6f}, Test BalAcc: {bal_acc:.6f}, Test PnL: {pnl:.5f}")
    print("\n" + "=" * 80)
    print("Phase 13 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()


