# HUST weighted-MAML GRU SOH trajectory experiment

이 디렉터리는 기존 CALCE 코드나 `hust_direct_rul_boil`,
`hust_hybrid_anil_boil`을 수정하지 않는 독립 실험입니다.

## 정확한 실험 정의

- 데이터: `data/HUST/HUST_<protocol>-<replicate>.pkl`
- 입력 구간: cycle 1~100 (`L=100` 고정)
- 입력 채널: `[SOH, cycle 평균 voltage, cycle 평균 current]`
- 출력: cycle 101부터 해당 target cell의 **마지막 관측 cycle까지** 미래 SOH
- 모델: multivariate GRU encoder + scalar autoregressive GRU decoder
- 메타러닝: second-order MAML
- source 가중치: target 첫 100 cycle만 이용하는 RBF MMD-QP
- 기본 source: target과 같은 HUST protocol의 나머지 replicate
- adaptation: fast `[1, 3, 5, 10, 15, 20]`, full 최대 200 step

SOH는 각 cycle의 마지막 유효 `discharge_capacity_in_Ah`를 nominal capacity로
나눕니다. Voltage/current는 각 cycle에 저장된 전체 샘플의 유효값 평균입니다.
Voltage/current z-score의 평균과 표준편차는 각 cell의 첫 100 cycle에서만
계산하므로 target 미래 정보는 학습이나 adaptation에 들어가지 않습니다.
중간 누락 cycle은 cycle 축 선형 보간하고, cycle 1 이전 쪽의 선두 레코드가
누락된 셀은 가장 이른 유효값으로 경계를 채운 뒤 해당 cycle을
`is_interpolated=true`로 기록합니다.

주의: current는 CALCE 설정과 동일하게 **부호를 유지한 whole-cycle 평균**입니다.
한 cycle 안에 charge와 discharge가 모두 있으면 양수/음수가 일부 상쇄될 수
있습니다. 이번 실험은 CALCE와 동일한 입력 정의를 비교하기 위해 이 방식을
의도적으로 유지합니다.

## 준비

프로젝트 루트에서 올바른 가상환경을 활성화하고 editable install을 한 번 합니다.

```bash
cd ~/바탕화면/battery_gru
source .venv/bin/activate
python -m pip install -e .
```

아래 두 경로가 있어야 합니다.

```text
data/HUST/HUST_1-1.pkl ...
data/Life labels/HUST_labels.json
```

## 학습 + adaptation + 평가

다음 명령은 `HUST_1-1.pkl`을 target으로 하고 같은 protocol의 나머지 cell을
source로 사용합니다.

```bash
python -m hust_weighted_gru \
  --config hust_weighted_gru/config.yaml \
  --target HUST_1-1.pkl \
  --source-mode same_protocol \
  --device cuda
```

다른 source 구성도 선택할 수 있습니다.

- `same_protocol`: 같은 protocol의 다른 replicate만 사용 (권장, CALCE
  `same_family` 대응)
- `all_hust`: target 한 cell을 제외한 모든 HUST cell 사용
- `leave_protocol_out`: target protocol 전체를 제외한 다른 protocol만 사용

`all_hust`와 `leave_protocol_out`은 매 meta iteration마다 수십 개 source의 긴
미래 trajectory를 평가하므로 `same_protocol`보다 훨씬 오래 걸립니다.

## 중단 후 재개

Meta checkpoint는 100 iteration마다 `last.pt`에 저장됩니다. Meta-training에는
early stopping이 없고 설정의 2300 iteration까지 진행합니다.

```bash
RUN=$(ls -td outputs/hust_weighted_gru/runs/*same_protocol_HUST_1-1_L100_* | head -1)

python -m hust_weighted_gru \
  --config hust_weighted_gru/config.yaml \
  --resume "$RUN/checkpoints/last.pt" \
  --device cuda
```

예를 들어 202 iteration에서 중단했다면 마지막 저장 시점인 200에서 이어집니다.

## 현재 checkpoint로 바로 adaptation

2300 iteration을 모두 기다리지 않고 현재 `last.pt`로 fast/full adaptation과
평가만 할 수 있습니다.

```bash
RUN=$(ls -td outputs/hust_weighted_gru/runs/*same_protocol_HUST_1-1_L100_* | head -1)

python -m hust_weighted_gru \
  --config hust_weighted_gru/config.yaml \
  --resume "$RUN/checkpoints/last.pt" \
  --adapt-only \
  --device cuda
```

Full adaptation에는 support loss 기준 patience 20의 early stopping이 있습니다.
Fast adaptation은 지정 step까지 정확히 진행하므로 early stopping을 사용하지
않습니다.

## 결과 위치

각 run은 다음처럼 설정을 읽을 수 있는 이름으로 저장됩니다.

```text
outputs/hust_weighted_gru/runs/
  <date>_same_protocol_HUST_1-1_L100_soh-voltage-current_weighted-maml_seed42/
```

주요 파일:

- `figures/key_results_summary.png`: 관측/예측 곡선, fast/full MAE·RMSE·RUL 오류
- `figures/target_soh_fast_*.png`, `target_soh_full.png`: step별 상세 결과
- `figures/training_loss.png`: weighted meta loss와 EMA
- `figures/alpha_trajectory.png`: source weight 변화
- `metrics/adaptation_comparison.csv`: fast/full 전체 성능 비교
- `predictions/target_*_prediction.csv`: cycle별 실제/예측 SOH
- `weights/final_alpha.csv`: target-aware source 가중치
- `config_resolved.yaml`, `run_manifest.json`: 완전히 해석된 설정과 실행 정보
