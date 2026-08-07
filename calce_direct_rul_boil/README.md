# CALCE Direct-RUL BOIL

기존 `battery_weighted_maml` 및 `paper_reproduction` 코드를 수정하지 않고 만든 독립 실험입니다. 첫 100 cycle의 요약 시계열에서 미래 SOH trajectory가 아닌 **cycle 100 시점의 RUL을 직접 회귀**합니다.

## 확정한 실험 정의

- 데이터: life label이 있는 CALCE 13개 cell
- 입력 길이: 첫 100 cycle 고정
- cycle별 입력 7개:
  1. SOH
  2. charge duration (hour)
  3. discharge duration (hour)
  4. mean charge voltage (V)
  5. mean discharge voltage (V)
  6. mean charge C-rate
  7. mean discharge C-rate
- 정답: `RUL = CALCE EOL cycle - 100`
- 평가: 아래 네 조건을 하나씩 통째로 제외하는 4-fold leave-one-domain-out
  - `CS2_0.5C`
  - `CS2_1C`
  - `CX2_0.5C`
  - `CX2_1C`
- held-out target domain의 feature, RUL label, normalization 통계는 학습과 checkpoint 선택에 사용하지 않습니다.
- 최종 평가는 target fine-tuning/adaptation 없이 direct forward로 수행합니다.

## 모델

첨부 논문의 아이디어를 가져오되 direct-RUL 문제와 작은 CALCE 표본 수에 맞게 축소했습니다.

```text
first 100 cycles × 7 features
                │
       ┌────────┴────────┐
       │                 │
domain-invariant F1   domain-specific F2  ← BOIL inner update는 이 body만
Transformer-CBAM ×2   Transformer-CBAM ×2
       │                 │
       ASP               ASP
       └────────┬────────┘
                │ concatenate
         fixed-in-inner predictor P
                │
         standardized direct RUL
```

두 feature extractor 모두 명시적 multi-head self-attention, CBAM, residual connection, attentive statistics pooling(ASP)을 사용합니다. F1에는 gradient reversal domain classifier와 domain-fuzzy loss를 적용하고, F1/F2에는 cosine orthogonality loss를 적용합니다.

BOIL meta episode에서는 source domain 중 하나를 가상 meta-test domain으로 순환 선택합니다. 나머지 source domain을 support로 사용해 **F2 파라미터만 inner update**하고, predictor P와 F1은 inner loop에서 고정합니다. Query outer loss는 F2와 P를 학습합니다. 기본값은 second-order BOIL입니다.

## 실행

저장소 루트에서 가상환경을 활성화한 뒤 의존성을 설치합니다.

```bash
pip install -r calce_direct_rul_boil/requirements.txt
```

권장 4-fold 전체 실행문은 다음과 같습니다.

```bash
python -m calce_direct_rul_boil.main \
  --config calce_direct_rul_boil/config.yaml \
  --fold all \
  --seed 42 \
  --device cuda
```

한 fold만 먼저 확인하려면 다음처럼 실행합니다.

```bash
python -m calce_direct_rul_boil.main \
  --config calce_direct_rul_boil/config.yaml \
  --fold CX2_1C \
  --seed 42 \
  --device cuda
```

CPU에서는 `--device cpu`를 사용합니다. `config.yaml`의 `device: auto`를 그대로 쓸 경우 `--device` 인자를 생략할 수 있습니다.

## 중단 후 이어서 학습

`last.pt`는 기본적으로 100 outer iteration마다 저장됩니다. 이어갈 때는 checkpoint와 같은 fold/seed를 명시해야 합니다.

```bash
python -m calce_direct_rul_boil.main \
  --config calce_direct_rul_boil/config.yaml \
  --fold CX2_1C \
  --seed 42 \
  --device cuda \
  --resume "outputs/calce_direct_rul_boil/실제_RUN/fold_CX2_1C_seed42/checkpoints/last.pt"
```

`config.yaml`의 `train.iterations`를 checkpoint iteration보다 크게 바꾸면 그 지점부터 새 종료 iteration까지 이어집니다. 구조나 feature 정의를 바꾼 checkpoint는 재사용하지 마십시오.

## 출력

기본 출력 경로는 다음 형식입니다.

```text
outputs/calce_direct_rul_boil/YYYYMMDD_HHMMSS_xxxxxx_direct-rul-boil_L100_seed42/
├── resolved_config.yaml
├── experiment.log
├── fold_CS2_0.5C_seed42/
│   ├── checkpoints/best.pt
│   ├── checkpoints/last.pt
│   ├── protocol.json
│   ├── training_history.csv
│   ├── training_diagnostics.png
│   ├── target_predictions.csv
│   ├── target_metrics.json
│   └── target_prediction_scatter.png
├── ... other folds ...
├── all_target_predictions.csv
├── fold_metrics.csv
├── seed_metrics.csv
├── aggregate_metrics.json
└── aggregate_target_prediction_scatter.png
```

예측 figure 제목에는 MAE, RMSE, R²가 표시됩니다. `target_predictions.csv`에는 실제/예측 RUL, 실제/예측 EOL과 cell별 absolute error가 저장됩니다.

## Loss와 early stopping

공동 학습 loss는 다음 항의 가중합입니다.

```text
Huber direct-RUL loss
+ domain adversarial classification loss
+ domain-fuzzy uniform KL loss
+ F1/F2 orthogonality loss
```

이어지는 BOIL loss는 support/query Huber loss입니다. RUL은 source fold의 평균과 표준편차로 표준화하므로 Huber delta 1.0은 source RUL 표준편차 한 단위에 해당합니다.

`evaluation_interval: 20`, `early_stopping_patience_evaluations: 15`가 기본이므로 source-only meta-CV MAE가 15회 연속 개선되지 않으면 종료합니다. 즉, 기본 patience는 300 outer iterations입니다. Target-domain 성능은 early stopping에 절대 사용하지 않습니다.

## 표본 수에 관한 주의

13개 cell만 사용하는 4-fold 결과는 seed에 민감할 수 있습니다. 논문용 최종 보고에는 `seeds`를 여러 개로 늘리고 fold별 결과와 전체 mean/std를 함께 제시하는 것이 좋습니다. 데이터셋을 추가할 때도 domain 단위 split을 유지해야 cell 또는 운전조건 누출을 피할 수 있습니다.
