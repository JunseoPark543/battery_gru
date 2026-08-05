# Direct Target-aware Weighted MAML for Battery SOH Forecasting

논문식 unweighted full second-order MAML 재현 코드는 기존 실험과 분리된
[`paper_reproduction/`](paper_reproduction/) 패키지에 있습니다. 기본 L=500
전체 학습과 CX2_37/CX2_38 meta-test는 다음 명령으로 실행합니다.

```bash
python -m paper_reproduction.main --mode all --config paper_reproduction/config.yaml
```

CALCE 배터리의 초기 SOH 구간만으로 target-aware source weight를 계산하고, full MAML로 target별 초기화를 학습한 뒤 target support에 적응하여 미래 SOH·EOL·RUL을 예측하는 로컬 PyTorch 프로젝트입니다. Target의 미래 trajectory와 EOL label은 meta-training, 가중치 계산, fine-tuning, early stopping, checkpoint 선택에 사용되지 않습니다.

## 1. 설치

Python 3.10 이상 환경에서 프로젝트 루트에서 실행합니다.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

CPU 실행은 `configs/base.yaml`의 `device: cpu`, 자동 GPU 선택은 `device: auto`, 특정 GPU는 `device: cuda:0`으로 지정합니다.

## 2. 데이터 배치

다음처럼 13개 PKL과 label JSON을 놓습니다. 공백이 포함된 `Life labels` 경로도 코드에서 `pathlib.Path`로 처리합니다.

```text
data/
├── CALCE/
│   ├── CALCE_CX2_16.pkl
│   ├── CALCE_CX2_33.pkl ... CALCE_CX2_38.pkl
│   └── CALCE_CS2_33.pkl ... CALCE_CS2_38.pkl
└── Life labels/
    └── CALCE_labels.json
```

각 PKL의 최소 schema는 다음과 같습니다.

```python
{
    "cell_id": "CX2_33",
    "nominal_capacity_in_Ah": 1.1,
    "cycle_data": [
        {
            "cycle_number": 1,
            "discharge_capacity_in_Ah": [0.0, 0.4, 1.08],
            # 나머지 측정값은 있어도 되지만 SOH 입력에는 사용하지 않음
        }
    ],
}
```

Loader는 dictionary/pandas Series, cycle list/numpy array/pandas DataFrame, capacity list/numpy/pandas 값을 처리합니다. 필수 key 누락, 비수치 cycle, 0 이하 nominal capacity에는 파일명과 key가 포함된 오류를 냅니다. `CALCE_labels.json`은 `{"CALCE_CX2_33.pkl": 520, ...}` 형식입니다.

## 3. SOH 전처리

각 cycle의 `discharge_capacity_in_Ah`에서 마지막 finite 값을 capacity로 선택하고 다음처럼 계산합니다.

```text
SOH_t = discharge_capacity_t / nominal_capacity_in_Ah
```

SOH는 clip 또는 min-max 정규화하지 않습니다. Cycle을 정렬하고, 중복 cycle은 마지막 유효 record를 사용하며 경고를 기록합니다. 비어 있는 cycle grid와 유효하지 않은 capacity는 선형 보간하고 전후 결측 수를 CSV에 남깁니다.

```bash
python scripts/preprocess_calce.py --config configs/base.yaml
```

결과는 `outputs/preprocessed/all_cells_soh.csv`, `cell_summary.csv`, cell별 `figures/*_soh.png`에 저장됩니다.

## 4. Task와 모델

Source cell 하나가 task 하나입니다. 초기 `L`개 SOH가 support, 나머지 trajectory가 query입니다. Support 내부에서는 모든 `j=1..L-1`에 대해 `[SOH_1..SOH_j] -> [SOH_j+1..SOH_L]` pair를 생성합니다. 서로 다른 길이는 padding하며 boolean mask 밖의 loss는 제외합니다. Source query는 cell별 horizon mean MSE이므로 수명이 긴 cell이 자동으로 더 큰 task weight를 갖지 않습니다.

모델은 packed sequence를 사용하는 GRU encoder와 autoregressive GRU decoder입니다. 학습 시 teacher forcing 비율은 기본 0.5이며, 최종 예측에서는 이전 예측을 다음 입력으로 넣는 완전 recursive mode(teacher forcing 0)를 사용합니다.

`model.features: [soh, voltage_mean]` 설정에서는 각 cycle의 유한한 `voltage_in_V` 값 평균을 두 번째 encoder 입력으로 사용합니다. 전압은 cell별 초기 L개 support에서 계산한 평균과 표준편차로만 z-score 정규화하므로 미래 전압 정보는 사용하지 않습니다. Decoder와 예측 target은 계속 SOH 하나이며, 미래 전압을 입력으로 요구하지 않습니다.

## 5. Target-aware weighted full MAML

매 meta iteration의 순서는 다음과 같습니다.

1. 현재 encoder로 각 source와 target의 초기 L만 encode합니다.
2. 각 `t=1..L-1`에서 `z_t=[L2-normalize(h_t), SOH_t+1]`을 만듭니다.
3. detached empirical point들의 RBF mean kernel을 계산합니다. 기본 sigma는 모든 point 간 양의 거리 median입니다.
4. CVXPY QP로 `alpha >= 0`, `sum(alpha)=1` 제약 아래 MMD를 최소화합니다. OSQP 실패 시 SCS를 시도하고, 둘 다 실패하면 종료합니다. Uniform weight로 대체하지 않습니다.
5. 각 source를 자신의 support로 독립적으로 differentiable adaptation합니다. `higher`의 `track_higher_grads=True`, `copy_initial_weights=False`를 사용해 기본적으로 second-order full MAML gradient를 유지합니다.
6. Adapted source query mean에만 alpha를 곱해 outer Adam update를 수행합니다.

Uniform warm-up이나 target support loss의 meta objective 추가는 없습니다. Best checkpoint는 target 성능이 아닌 weighted source query loss EMA로 고릅니다.

## 6. 데이터 누수 차단

`FullCellTrajectory`는 orchestration 단계에만 있고 trainer에 전달되지 않습니다. `TargetSupportView`에는 첫 L cycle/SOH만 있으며 future와 true EOL field 또는 전체 trajectory 접근 method가 없습니다. Weight calculator와 `WeightedMAMLTrainer`는 이 view만 받습니다. Fast/full adaptation이 모두 끝난 뒤에만 `TargetEvaluationView.after_training(...)`을 생성해 evaluator가 미래와 label을 읽습니다.

Forecast horizon은 target의 실제 길이나 EOL이 아니라 `max_forecast_cycle - L`로 정합니다. Full adaptation early stopping도 target support loss만 사용합니다.

## 7. 실행 명령

단일 실험:

```bash
python scripts/run_single.py --target CALCE_CX2_37.pkl --history-length 100 --source-mode same_family --config configs/base.yaml
```

SOH와 cycle 평균 전압을 함께 입력하는 L=100, 2,300 iteration 실험:

```bash
python scripts/run_single.py --target CALCE_CX2_37.pkl --history-length 100 --source-mode same_family --config configs/l100_soh_voltage_2300.yaml
```

이 설정은 encoder 입력 차원이 2이므로 기존 SOH-only checkpoint에서 resume할 수 없습니다. 새 run으로 시작해야 합니다.

Weighted meta-learning 없이 SOH 하나만 입력하는 plain GRU encoder-decoder L=500 baseline:

```bash
python scripts/run_gru_baseline.py \
  --target CALCE_CX2_37.pkl \
  --config configs/gru_baseline_l500.yaml
```

이 baseline은 source cell, MMD, QP alpha, MAML inner/outer loop를 사용하지 않습니다. Target의 최초 500 cycle 중 앞 450개로 sliding-window 학습하고 뒤 50개 support cycle의 recursive MSE로 early stopping checkpoint를 고릅니다. 최종 예측에서는 선택된 모델의 encoder에 관측된 500개 SOH 전체를 넣습니다. 기본값은 최대 300 epoch, patience 30입니다.

중단된 baseline은 다음처럼 마지막으로 저장된 epoch에서 재개합니다.

```bash
python scripts/run_gru_baseline.py \
  --target CALCE_CX2_37.pkl \
  --config configs/gru_baseline_l500.yaml \
  --resume outputs/runs/RUN_NAME/checkpoints/last.pt
```

빠른 end-to-end 검증(meta iteration 2, full adaptation 2 step, horizon 10):

```bash
python scripts/run_single.py --target CALCE_CX2_37.pkl --history-length 50 --source-mode same_family --config configs/base.yaml --smoke-test
```

동일 family의 6개 실험:

```bash
python scripts/run_same_family.py --config configs/base.yaml
```

모든 다른 CALCE cell을 source로 쓰는 6개 실험:

```bash
python scripts/run_all_calce.py --config configs/base.yaml
```

요구된 순서대로 same-family 6개 이후 all-CALCE 6개를 순차 실행하고 집계:

```bash
python scripts/run_all_experiments.py --config configs/base.yaml
```

기존 결과 재집계:

```bash
python scripts/aggregate_results.py --outputs-dir outputs/runs
```

중단된 run의 `last.pt`에서 동일 총 iteration까지 재개:

```bash
python scripts/run_single.py --resume outputs/runs/RUN_NAME/checkpoints/last.pt --config configs/base.yaml
```

저장 iteration과 관계없이 해당 checkpoint 파라미터로 meta-training을 건너뛰고
target adaptation/평가만 다시 실행:

```bash
python scripts/run_single.py \
  --resume outputs/runs/RUN_NAME/checkpoints/last.pt \
  --config configs/base.yaml \
  --adapt-only
```

`--adapt-only`는 전달한 checkpoint 자체를 사용합니다. `last.pt`를 전달하면 마지막
저장 iteration 모델, `best_source_meta_loss.pt`를 전달하면 best EMA 모델을
adaptation합니다. 결과는 해당 run 폴더의 adaptation/metrics/predictions/figures를
갱신합니다.

Checkpoint에는 모델/outer optimizer, iteration, source-only best metric, resolved config, target/source, L, mode, seed, alpha와 Python/NumPy/Torch/CUDA RNG 상태가 포함됩니다.

Target fast adaptation은 기본적으로 하나의 연속 SGD 경로를 20 step까지 실행하고
`1, 3, 5, 10, 15, 20` step의 모델을 snapshot으로 평가합니다. 각 step을 처음부터
별도로 재실행하지 않으므로 step 1 모델은 step 3 모델의 정확한 prefix입니다.

```yaml
adaptation:
  fast_steps: [1, 3, 5, 10, 15, 20]
```

## 8. 출력

각 run은 `outputs/runs/{timestamp}_{mode}_{target}_L{L}_seed{seed}/` 아래 독립적으로 생성됩니다.

```text
config_resolved.yaml, run_manifest.json
logs/train.log
checkpoints/{best_source_meta_loss.pt,last.pt}
preprocessing/{source_summary.csv,target_support.csv}
training/{iteration_history.csv,source_loss_history.csv,gradient_history.csv}
weights/{alpha_history.csv,final_alpha.csv,kernel_matrix_final.csv,alpha_heatmap.png}
adaptation/{fast_adaptation_history.csv,full_adaptation_history.csv}
predictions/{target_fast_1_prediction.csv,...,target_fast_20_prediction.csv,target_full_prediction.csv}
metrics/{fast_1_metrics.json,...,fast_20_metrics.json,fast_metrics_by_step.csv,full_metrics.json}
figures/{target_soh_fast_1.png,...,target_soh_fast_20.png,target_soh_full.png,training_loss.png,alpha_trajectory.png,source_query_losses.png}
```

기존 집계 코드와의 호환성을 위해 `fast_metrics.json`,
`target_fast_prediction.csv`, `target_soh_fast.png`는 1-step 결과의 별칭으로 계속
저장됩니다.

12개 실험 집계는 `outputs/experiment_summary.{csv,json}`과 `outputs/comparison_figures/`에 저장됩니다. Console과 `logs/train.log`에는 source별 loss, 전체 alpha, entropy, effective source 수, MMD, sigma, solver, gradient norm, LR, 시간/ETA와 CUDA 메모리가 기록됩니다.

## 9. Metric

- SOH: MAE, RMSE, R², max absolute error, 마지막 실제 cycle absolute error, 실제 EOL 이전/이후 MAE, 평가 point 수
- EOL: 예측 SOH가 처음 0.8 이하가 되는 discrete cycle과 인접 두 점의 linear-interpolated crossing
- RUL: `EOL - L`; signed/absolute/relative absolute error

Forecast 안에 threshold crossing이 없으면 predicted EOL/RUL은 NaN이고 `crossing_found=false`입니다. R²가 정의되지 않으면 NaN 사유를 metric과 log에 기록합니다.

## 10. 테스트

```bash
python -m pytest -q
```

Schema/SOH/보간/split, prefix padding과 mask, GRU, QP alpha 제약·대칭·근접성, weighted meta gradient, second-order graph, target leakage와 synthetic end-to-end smoke output을 검사합니다.

## 11. 재현성과 알려진 한계

Python, NumPy, Torch, CUDA seed와 deterministic 설정을 적용하고 manifest에 환경·명령·package version·가능한 경우 git commit을 기록합니다. 다만 CUDA/CVX solver와 플랫폼 차이로 마지막 자릿수까지 완전히 같지 않을 수 있습니다.

기본 10,000 iteration full MAML은 source별 second-order graph와 긴 decoder horizon 때문에 CPU에서 오래 걸리고 메모리를 많이 씁니다. 메모리 부족 시 `hidden_size`, `inner_batch_size`를 낮추거나 한 번에 한 run만 실행하십시오. `full_maml: false`는 명시적으로 first-order 실험을 할 때만 사용합니다. 학습 시간을 줄이려면 `meta_iterations`를 낮출 수 있지만 연구 기본 결과와 직접 비교할 수 없습니다. CVXPY solver 오류는 숨기지 않으므로 OSQP/SCS 설치를 먼저 확인해야 합니다.
