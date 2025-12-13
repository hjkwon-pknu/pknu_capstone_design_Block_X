# pknu_capstone_design_Block_X
# BTC-Trading-System-Capstone (Block_X)

BTCUSDT 시계열 데이터 기반으로,
(1) 거래비용을 반영한 라벨링(Forward Return)과
(2) Threshold(의사결정 임계값) 최적화(Thr Sweep),
(3) 시퀀스 모델(LSTM/GRU) 및 트리 계열 모델(XGB/LGBM/CAT)의 성능을
분류 지표(Accuracy/Balanced Acc) + 트레이딩 지표(PnL/Sharpe/MDD)로 함께 평가한 캡스톤 프로젝트입니다.

## Repository Structure
- `src/`
  - `download.py` : 데이터 준비(선택)
  - `pnl_utils.py` : 공통 PnL 계산 유틸
  - `ml_step10_label_redesign.py` : 거래비용 반영 라벨(Phase 10)
  - `ml_step11_multitimeframe_features.py` : MTF 피처 구성(Phase 11)
  - `ml_step12_tree_advanced.py` : 트리 계열 튜닝(Phase 12)
  - `ml_step13_sequence_models.py` : CNN1D 시퀀스 실험(Phase 13)
  - `ml_step14_rnn_models.py` : LSTM/GRU + Thr Sweep + (선택)Buy&Hold 비교(Phase 14)
- `experiments/` : 초기 베이스라인(step1~step9) 실험 코드 보관
- `reports/` : 그래프/CSV 등 실험 산출물 저장
- `data/` : 데이터 폴더(원본 CSV는 업로드하지 않음)

## Data Format
입력 데이터는 최소 다음 컬럼을 포함한 CSV를 사용합니다:
`datetime, open, high, low, close, volume`

프로젝트 코드에서는 데이터 경로를 환경변수로 맞춰 사용했습니다.
- (권장) `BTC_FILE_PATH` 환경변수에 1분봉 CSV 경로 지정
