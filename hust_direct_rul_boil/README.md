# HUST Hierarchical Direct Raw-RUL BOIL

기존 CALCE 프로젝트를 수정하지 않고 만든 독립 HUST 실험입니다. HUST의 첫 100 cycles만 사용하고, cycle 100 시점의 RUL을 **표준화하지 않은 실제 cycle 단위**로 직접 예측합니다.

## 실험 정의

- 데이터 경로: `data/HUST/*.pkl`
- label 경로: `data/Life labels/HUST_labels.json`
- 예상 파일명: `HUST_<protocol>-<replicate>.pkl`, 예: `HUST_1-1.pkl`
- 입력: cycle 1~100
- reference: cycle 10
- 정답: `raw RUL = EOL label - 100`
- 평가: protocol 번호 하나를 완전히 제외하는 leave-one-protocol-out
- target protocol의 cell, feature 통계, RUL label은 학습 및 checkpoint 선택에 사용하지 않음
- target adaptation 없음

BatteryLife/BatteryML 형식의 HUST는 77개 LFP cell과 10개 protocol group을 갖는 것을 전제로 합니다. 서버 데이터의 protocol 수가 다르면 `expected_protocol_count`를 실제 값으로 바꾸십시오.

## 입력 feature

각 cycle의 충전 구간을 상대 시간축 64점으로 보간합니다.

### 충전 파형 8채널

```text
1. charge voltage
2. charge current C-rate
3. cumulative charge capacity / nominal capacity
4. voltage × current C-rate (power proxy)
5. voltage - cycle-10 voltage profile
6. current - cycle-10 current profile
7. capacity fraction - cycle-10 profile
8. power proxy - cycle-10 profile
```

“10cycle과의 voltage/current”는 cycle 10 파형을 early reference로 두고 이후 각 cycle의 변화량을 계산한다는 의미로 구현했습니다.

### Cycle scalar 14개

```text
SOH
charge/discharge duration
mean charge/discharge voltage
mean charge/discharge C-rate
coulombic efficiency
charge/discharge energy per nominal capacity
cycle-10 대비 SOH 변화
cycle-10 대비 charge/discharge duration 변화
최근 10-cycle SOH slope
```

입력 feature는 source training cell 통계로만 표준화합니다. RUL label은 평균/표준편차 표준화를 하지 않습니다.

## 확장 모델

```text
cycle 내부 64×8 waveform
        │
Dilated residual 1-D CNN
+ attentive statistics pooling
        │ cycle embedding
        ├──────── cycle scalar MLP
        │
multi-scale cycle convolution (kernel 3/7/15)
        │
 ┌──────┴──────────────────┐
 │                         │
Invariant F1            Specific F2
Transformer-CBAM ×2     Transformer-CBAM ×2
 │                         │
ASP                       ASP
 └──────────┬──────────────┘
            │
positive raw-RUL prediction head
            │
      RUL in actual cycles
```

F1과 F2가 각각 waveform encoder부터 독립적으로 가지고 있어 BOIL inner loop에서 F2의 저수준 파형 표현까지 바뀔 수 있습니다.

## Raw-RUL loss

출력은 실제 cycle 단위입니다. Dataset mean/std로 label을 변환하거나 복원하지 않습니다.

```text
raw RUL task loss
  = SmoothL1(predicted cycles, actual cycles, beta=50) / 500
```

`/ 500`은 dataset 통계가 아닌 고정 물리 단위의 loss scaling입니다. Prediction head는 `500 × softplus(score)`로 양수 RUL을 출력하고, 초기 bias만 source training RUL 중앙값 근처로 설정합니다. Target label 통계는 사용하지 않습니다.

Joint loss:

```text
raw-RUL task
+ 0.10 × domain adversarial CE
+ 0.05 × domain fuzzy KL
+ 0.10 × F1/F2 orthogonality
```

BOIL inner loop는 F2 body만 업데이트하고 prediction head는 고정합니다. Outer loop에서는 F2와 prediction head가 업데이트됩니다.

## Leakage 방지와 모델 선택

각 outer target fold에서 다음처럼 나뉩니다.

```text
1 target protocol: 최종 평가 전까지 완전히 격리
9 source protocols:
  - protocol마다 1개 cell: source validation 전용
  - 나머지 cell: joint/BOIL training
```

Source validation cell은 입력 normalization과 학습에도 사용하지 않습니다. Checkpoint는 이 validation cell들의 raw-cycle MAE로 선택합니다.

각 meta iteration에서는 source protocol 하나를 가상 meta-test로 정하고, 그 protocol은 해당 iteration의 joint update에서도 제외합니다.

## 서버 데이터 검사

먼저 모델을 만들거나 학습하지 않고 pickle/label/schema/feature를 검사합니다.

```bash
cd ~/바탕화면/battery_gru
source .venv/bin/activate

python -m hust_direct_rul_boil.main \
  --config hust_direct_rul_boil/config.yaml \
  --inspect-data-only
```

77개 대용량 pickle을 읽고 첫 100 cycles의 파형을 추출하므로 검사 자체에는 시간이 걸릴 수 있습니다.

## 한 protocol 먼저 실행

확장된 계층형 모델과 second-order BOIL이므로 전체 10-fold 전에 한 fold를 먼저 권장합니다.

```bash
python -m hust_direct_rul_boil.main \
  --config hust_direct_rul_boil/config.yaml \
  --fold protocol_1 \
  --seed 42 \
  --device cuda
```

## 전체 protocol 실행

```bash
python -m hust_direct_rul_boil.main \
  --config hust_direct_rul_boil/config.yaml \
  --fold all \
  --seed 42 \
  --device cuda
```

## Resume

`last.pt`는 100 iterations마다 저장되며 early stopping 시에도 저장됩니다.

```bash
CKPT=$(ls -t outputs/hust_direct_rul_boil/*/fold_protocol_1_seed42/checkpoints/last.pt | head -1)

python -m hust_direct_rul_boil.main \
  --config hust_direct_rul_boil/config.yaml \
  --fold all \
  --seed 42 \
  --device cuda \
  --resume "$CKPT"
```

완료된 fold는 건너뛰고 중단된 fold와 나머지 fold를 같은 RUN 폴더에서 계속합니다.

## 출력

```text
outputs/hust_direct_rul_boil/<RUN>/
├── dataset_summary.json
├── resolved_config.yaml
├── fold_protocol_1_seed42/
│   ├── source_split.json
│   ├── checkpoints/best.pt
│   ├── checkpoints/last.pt
│   ├── training_history.csv
│   ├── training_diagnostics.png
│   ├── target_predictions.csv
│   ├── target_metrics.json
│   └── target_prediction_scatter.png
├── fold_metrics.csv
├── all_target_predictions.csv
├── aggregate_metrics.json
├── key_results_dashboard.png
└── aggregate_target_prediction_scatter.png
```

각 target metric에는 동일 source training cell의 평균 RUL만 예측하는 baseline MAE도 함께 저장됩니다.

`key_results_dashboard.png` 한 장에는 다음 핵심 결과가 함께 표시됩니다.

```text
전체 MAE / RMSE / MAPE / R² / ±15% 정확도
Actual vs predicted scatter
Protocol별 모델 MAE와 source-mean baseline 비교
Protocol별 과대·과소예측 bias
Protocol별 ±15% 이내 정확도
가장 성능이 나쁜 protocol과 baseline을 이긴 protocol 수
```
