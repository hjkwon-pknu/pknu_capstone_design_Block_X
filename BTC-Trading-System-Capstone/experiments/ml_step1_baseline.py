# 파일명 예: ml_step1_baseline.py
import sys
import subprocess

# 의존성 자동 설치 함수
def pip_install(package):
  subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# pandas 자동 설치
try:
  import pandas as pd
except:
  print("pandas가 설치되지 않았습니다. 자동 설치 중...")
  pip_install("pandas")
  import pandas as pd

# numpy 자동 설치
try:
  import numpy as np
except:
  print("numpy가 설치되지 않았습니다. 자동 설치 중...")
  pip_install("numpy")
  import numpy as np

# scikit-learn 자동 설치
try:
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
except:
  print("scikit-learn이 설치되지 않았습니다. 자동 설치 중...")
  pip_install("scikit-learn")
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

FILE_PATH = r"C:\Users\User\binance_data\1m_history.csv"  # 필요 시 경로 수정
USE_ROWS = 1_000_000  # 처음엔 50만~100만 행으로 시작

def load_data(path, use_rows=None):
  df = pd.read_csv(path, usecols=['datetime','open','high','low','close','volume'])
  if use_rows is not None and len(df) > use_rows:
    df = df.tail(use_rows)
  df = df.sort_values('datetime').reset_index(drop=True)
  return df

def make_features(df):
  # 수익률/변동성/거래량 기반의 가벼운 특성
  df['ret_1'] = df['close'].pct_change(1)
  df['ret_3'] = df['close'].pct_change(3)
  df['ret_5'] = df['close'].pct_change(5)
  df['hl_range'] = (df['high'] - df['low']) / df['close']
  df['vol_chg'] = df['volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
  df['ret_mean_10'] = df['ret_1'].rolling(10, min_periods=10).mean()
  df['ret_std_10']  = df['ret_1'].rolling(10, min_periods=10).std()
  # 타깃: 다음 1분이 오르면 1, 내리면 0 (동일가는 0으로 처리)
  df['y'] = (df['close'].shift(-1) > df['close']).astype(int)
  df = df.dropna().reset_index(drop=True)
  features = ['ret_1','ret_3','ret_5','hl_range','vol_chg','ret_mean_10','ret_std_10']
  return df, features

def time_split(df, test_ratio=0.2):
  n = len(df)
  split = int(n * (1 - test_ratio))
  train = df.iloc[:split].copy()
  test  = df.iloc[split:].copy()
  return train, test

def main():
  df = load_data(FILE_PATH, USE_ROWS)
  df, feats = make_features(df)
  train, test = time_split(df, test_ratio=0.2)
  X_tr, y_tr = train[feats].values, train['y'].values
  X_te, y_te = test[feats].values,  test['y'].values

  pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=300, class_weight='balanced'))
  ])
  pipe.fit(X_tr, y_tr)

  pred = pipe.predict(X_te)
  proba = pipe.predict_proba(X_te)[:,1]

  print("Accuracy:", accuracy_score(y_te, pred))
  print("Balanced Acc:", balanced_accuracy_score(y_te, pred))
  print(classification_report(y_te, pred, digits=4))

  # 간단 PnL(확률 임계값 0.55 이상/이하일 때만 매매)
  thr = 0.55
  signal = np.where(proba >= thr, 1, np.where(proba <= 1-thr, -1, 0))
  ret = test['close'].pct_change().fillna(0).values
  pnl = (signal[:-1] * ret[1:])  # 한 틱 뒤 반영
  print(f"Trades: {(signal!=0).sum()}, PnL(단순 합): {pnl.sum():.5f}")

if __name__ == "__main__":
  main()
