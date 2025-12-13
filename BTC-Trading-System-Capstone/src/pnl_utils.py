import math
from typing import Iterable, Tuple, Dict, Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def make_signal_from_proba(proba, thr: float) -> np.ndarray:
    """Convert predicted probabilities into {-1, 0, 1} trading signals.

    Symmetric thresholding:
      - Long  : proba >= thr
      - Short : proba <= 1-thr
      - Flat  : otherwise
    """
    p = np.asarray(proba, dtype=float)
    if not (0.5 <= thr < 1.0):
        raise ValueError("thr must be in [0.5, 1.0)")
    return np.where(p >= thr, 1, np.where(p <= 1.0 - thr, -1, 0)).astype(int)


def _get_forward_return(test_df, horizon: int) -> np.ndarray:
    col = f"fwd_ret_h{horizon}"
    if hasattr(test_df, "columns") and col in getattr(test_df, "columns"):
        r = np.asarray(test_df[col].to_numpy(), dtype=float)
    else:
        close = np.asarray(test_df["close"].to_numpy(), dtype=float)
        future = np.roll(close, -horizon)
        r = future / close - 1.0
        r[-horizon:] = 0.0
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r


def calculate_pnl_from_forward_return(
    test_df,
    signal,
    horizon: int = 4,
    cost: float = 0.002,
    non_overlapping: bool = True,
) -> Tuple[float, int]:
    """PnL calculator consistent with Step10's fwd_ret_h{horizon}.

    - Uses forward (horizon-bar) return if available.
    - Applies a per-trade cost whenever |signal|>0.
    - If non_overlapping=True: after entering a trade at t, skips next (horizon-1) steps
      to avoid overlapping holding periods.

    Returns:
      (total_pnl, trades)
    """
    sig = np.asarray(signal, dtype=int)
    r = _get_forward_return(test_df, horizon=horizon)

    n = min(len(sig), len(r))
    sig = sig[:n]
    r = r[:n]

    pnl = np.zeros(n, dtype=float)
    trades = 0

    if not non_overlapping:
        trade = (sig != 0).astype(float)
        pnl = sig * r - cost * trade
        trades = int(trade.sum())
        return float(pnl.sum()), trades

    i = 0
    while i < n:
        s = int(sig[i])
        if s == 0:
            i += 1
            continue
        pnl[i] = s * r[i] - cost
        trades += 1
        i += max(1, horizon)

    return float(pnl.sum()), trades


def pnl_details(
    test_df,
    signal,
    horizon: int = 4,
    cost: float = 0.002,
    non_overlapping: bool = True,
) -> Dict[str, Any]:
    """Detailed PnL + equity curve + simple risk stats."""
    sig = np.asarray(signal, dtype=int)
    r = _get_forward_return(test_df, horizon=horizon)

    n = min(len(sig), len(r))
    sig = sig[:n]
    r = r[:n]

    pnl = np.zeros(n, dtype=float)
    trade_mask = np.zeros(n, dtype=bool)

    if not non_overlapping:
        trade_mask = sig != 0
        pnl = sig * r - cost * trade_mask.astype(float)
    else:
        i = 0
        while i < n:
            if sig[i] == 0:
                i += 1
                continue
            trade_mask[i] = True
            pnl[i] = sig[i] * r[i] - cost
            i += max(1, horizon)

    equity = np.cumsum(pnl)

    trade_pnl = pnl[trade_mask]
    trades = int(trade_mask.sum())
    wins = int((trade_pnl > 0).sum()) if trades else 0
    win_rate = wins / trades if trades else 0.0
    avg_trade = float(trade_pnl.mean()) if trades else 0.0
    std_trade = float(trade_pnl.std(ddof=1)) if trades > 1 else 0.0
    sharpe = (avg_trade / std_trade * math.sqrt(trades)) if std_trade > 0 else 0.0

    peak = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    drawdown = equity - peak
    mdd = float(drawdown.min()) if len(drawdown) else 0.0  # negative value

    return {
        "pnl": pnl,
        "equity": equity,
        "trades": trades,
        "win_rate": float(win_rate),
        "avg_trade": float(avg_trade),
        "sharpe_trades": float(sharpe),
        "mdd": float(mdd),
    }


def sweep_thresholds(
    test_df,
    proba,
    thr_list: Iterable[float],
    horizon: int = 4,
    cost: float = 0.002,
    non_overlapping: bool = True,
    min_trades: int = 0,
):
    """Evaluate multiple thresholds and return a table.

    Sorting:
      - pnl desc
      - trades asc
    """
    if pd is None:
        raise ImportError("pandas is required for sweep_thresholds()")

    rows = []
    for thr in thr_list:
        sig = make_signal_from_proba(proba, float(thr))
        det = pnl_details(
            test_df,
            sig,
            horizon=horizon,
            cost=cost,
            non_overlapping=non_overlapping,
        )
        rows.append(
            {
                "thr": float(thr),
                "pnl": float(det["equity"][-1]) if len(det["equity"]) else 0.0,
                "trades": int(det["trades"]),
                "win_rate": float(det["win_rate"]),
                "avg_trade": float(det["avg_trade"]),
                "sharpe_trades": float(det["sharpe_trades"]),
                "mdd": float(det["mdd"]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["pnl", "trades"], ascending=[False, True]).reset_index(drop=True)

    if min_trades and len(df):
      df = df[df["trades"] >= int(min_trades)].reset_index(drop=True)
    return df
