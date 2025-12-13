# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys
import tempfile
import subprocess
from datetime import datetime, timedelta
from zipfile import ZipFile
from os import environ, getenv, makedirs, getcwd, walk, remove
from os.path import basename, join, exists, expanduser as home

def pip_install(package):
  subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def pip_install_requirements(requirements_dir):
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_dir.rstrip(".txt")+".txt"])

## GitPython으로 git을 사용할 수 있도록 한다. 없다면 pip로 GitPython을 설치 한다.
try:
  from git import Repo
except:
  pip_install("GitPython")
  from git import Repo

## 캔들 데이터를 사용하기 위해서는 Pandas가 필요하다.
try:
  import pandas as pd
except:
  pip_install("pandas")
  import pandas as pd

## Pandas 짝꿍 Numpy가 필요하다.
try:
  import numpy as np
except:
  pip_install("numpy")
  import numpy as np

## Binance REST API로 다운로드된 데이터에서 부족한 부분만 가져올 수 있도록 requests를 사용하자.
try:
  import requests
except:
  pip_install("requests")
  import requests

## 바이낸스 퍼블릭 데이터 다운로드 소스코드를 Temp(임시폴더) 다운로드 받아서 사용하도록 한다.
repo_url = "https://github.com/binance/binance-public-data.git"
temp_path = tempfile.mkdtemp(prefix='candle_download_')

## git으로 소스코드를 임시폴더에 클론(다운로드) 시키고 위치를 저장해두자.
repo_path = Repo.clone_from(repo_url, temp_path)
WORK_PATH = repo_path.working_dir

## STORE_DIRECTORY 환경변수가 없으면 사용자폴더에 binance_data를 사용하도록 설정한다.
STORE_PATH = join(home('~'), "binance_data") if not "STORE_DIRECTORY" in environ.keys() else getenv("STORE_DIRECTORY")
environ["STORE_DIRECTORY"] = STORE_PATH

## 캔들 데이터를 다운로드 받는 download-kline.py를 실행한다.
def download_klines(cmd, args):
  subprocess.check_call(cmd + args)

## 저장할 위치가 없으면 만들어주고, download-kline.py에 다운로드 받을 코인 정보등을 입력한다.
def download_binance_datas(symbol="BTCUSDT", interval="1m"):
  # environ["STORE_DIRECTORY"] = "/Users/name/binance_data/"
  if not exists(STORE_PATH):
    makedirs(STORE_PATH)
  # Install requirements library
  pip_install_requirements(join(WORK_PATH, "python", "requirements.txt"))
  # configure download command,
  kline_cmd = [sys.executable, join(WORK_PATH, "python", "download-kline.py")]
  monthly_args = ["-t", "um", "-s", symbol, "-i", interval, "-skip-daily", "1", "-startDate", "2020-01-01"]
  daily_args = ["-t", "um", "-s", symbol, "-i", interval, "-skip-monthly", "1", "-startDate", f"{datetime.now().strftime('%Y-%m')}-01"]
  # excute download kline
  download_klines(kline_cmd, monthly_args)
  download_klines(kline_cmd, daily_args)

def klines_unzip(search_directory=STORE_PATH):
  search_directory = join(search_directory, 'data')
  for root, dirs, files in walk(search_directory):
    for file in files:
      if file.endswith('.zip'):
        zip_file_path = join(root, file)
        # 압축을 풀 디렉토리 선택 (zip 파일이 있는 폴더와 동일한 위치)
        extract_directory = root
        # 압축 파일 열기
        with ZipFile(zip_file_path, 'r') as zip_ref:
          # 압축 해제
          zip_ref.extractall(extract_directory)
        print(f'압축 해제: {zip_file_path} -> {extract_directory}')

def klines_history(search_directory=STORE_PATH):
  search_directory = join(search_directory, 'data')
  # 모든 CSV 파일을 저장할 데이터 프레임 초기화
  history = pd.DataFrame()
  for root, dirs, files in walk(search_directory):
    for file in sorted(files):
      if file.endswith('.csv') and file != f'{basename(root)}.csv':
        csv_file_path = join(root, file)
        # 해더가 있는지 확인하고 넘어가야함. 첫 번째 라인 확인
        with open(csv_file_path, 'r') as file:
          first_line = file.readline()
          # 컬럼 이름이 있는 경우 읽을때
          if 'open_time' in first_line or 'Open' in first_line or 'open' in first_line:
            print(f'DataFrame(Header exist): {csv_file_path}')
            df = pd.read_csv(csv_file_path)  # header=0 (기본값)
          # 컬럼 이름이 없는 경우 읽을때
          else:
            print(f'DataFrame(Header empty): {csv_file_path}')
            df = pd.read_csv(csv_file_path, header=None)
            df.columns = ['open_time', 'open','high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        df = df.iloc[:, :6]
        df.columns = ['datetime', 'open','high', 'low', 'close', 'volume']
        history = pd.concat([history, df])
  history.index = pd.to_datetime(history['datetime'], unit='ms', utc=True)
  history = history.astype(float)
  history = history.tz_convert('Asia/Seoul')
  history = history.iloc[np.unique(history.index.values, return_index=True)[1]]
  history_file_path = join(STORE_PATH, f'{basename(root)}_history.csv')
  print("### Save History Klines/Candles (Download) : ", history_file_path)
  history.to_csv(history_file_path, index=False)
  return history, history_file_path, history['datetime'].iloc[-1]

if __name__ == "__main__":
  ## 바이낸스 비트코인 선물 1분지표 다운로드를 실행한다.
  download_binance_datas(symbol="BTCUSDT", interval="1m")
  ## 다운받은 zip파일들의 압축을 해제한다.
  klines_unzip(search_directory=STORE_PATH)
  ## csv파일들을 읽어 csv단일 파일로 저장한다.
  history_df, history_file_path, history_last_timestamp = klines_history(STORE_PATH)
  print(f"### Download Klines/Candles Count is: {len(history_df)}")
  print(f"### Last Klines/Candles Timestamp is: {history_last_timestamp}")