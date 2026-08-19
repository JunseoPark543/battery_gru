# BatteryLife MATR Partial I–V Conditioned ANP

이 파이프라인은 기존 CALCE/HUST 실험과 분리된 `battery_weighted_maml.matr_anp` 패키지이며, BatteryLife의 **MATR 데이터만** 사용한다. 목표는 RUL/EOL이 아니라 현재 cycle `n*`부터 마지막 valid cycle까지의 SOH 확률 trajectory다.

## 설치와 데이터 경로

프로젝트 루트에서 설치한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

데이터 경로 우선순위는 `--data-root` → `BATTERYLIFE_DATA_ROOT` → config의 `paths.data_root`이다. 서버와 로컬의 공통 구조가 확인되어 기본값은 프로젝트 루트 기준 `data/MATR`이다.

```bash
export BATTERYLIFE_DATA_ROOT=/absolute/path/to/battery_gru/data/MATR
```

환경변수나 `--data-root`를 생략하면 자동으로 `data/MATR`을 사용한다. loader는 경로 또는 metadata로 MATR임을 확인한 pickle만 읽는다. 기대하는 canonical BatteryLife 구조는 다음과 같다.

```python
{
    "dataset": "MATR",                 # 또는 경로에 MATR 포함
    "cell_id": "...",
    "nominal_capacity_in_Ah": 1.1,
    "cycle_data": [
        {
            "cycle_number": 1,
            "voltage_in_V": [...],
            "current_in_A": [...],
            "discharge_capacity_in_Ah": [...],
            "stage": [...],             # 있으면 discharge 검증에 사용
        }
    ],
}
```

## 데이터 inspection을 가장 먼저 실행

```bash
python -m battery_weighted_maml.matr_anp.inspect_data \
  --config configs/matr_partial_iv_anp.yaml \
  --data-root "$BATTERYLIFE_DATA_ROOT"
```

`outputs/matr_partial_iv_anp/data_audit/`에 cell/cycle/신호 길이/current polarity/누락·중복·비단조 정보가 저장된다. 서버 pickle schema가 위 구조와 다르면 학습 전에 여기서 명시적인 오류가 발생한다.

`data/Life labels` 아래 MATR label 파일이 함께 있더라도 이 파이프라인에서는 읽지 않는다. 이번 목표는 RUL/EOL label 예측이 아니라 각 pickle의 cycle별 discharge capacity로 계산한 SOH trajectory 예측이기 때문이다.

## 모델과 누수 방지

- `soh_only_anp`: SOH context만 사용하는 latent ANP
- `soh_only_anp_wide`: 제안 모델과 parameter 수 차이가 ±5% 이내인 SOH-only ANP
- `partial_iv_anp`: `[ΔV, |I|, mask]` CNN embedding을 latent prior/posterior, deterministic path, decoder 모두에 조건으로 사용
- 현재 cycle의 실제 SOH는 context에서 제외하고 target에 포함한다.
- cycle 입력은 `cycle_number / max_cycle_train`이며 test cell의 전체 수명 `N`은 입력에 쓰지 않는다.
- SOH/cycle/ΔV/current scaler는 각 fold의 train cell로만 fit한다.
- q는 `Qd(t) / Q_nominal`이고 현재 cycle의 최종 capacity로 재정규화하지 않는다.
- 고정 q-grid 밖은 extrapolation하지 않고 mask 처리한다.
- reference는 같은 cell의 현재 cycle 이전 자료만 사용한다.
- 평가는 optimizer, backward, parameter update 없이 수행한다.

학습 loss는 masked Gaussian NLL과 analytic diagonal-Gaussian KL로 구성된 ANP ELBO이며 KL warm-up, gradient clipping, AMP, early stopping, best/last checkpoint와 resume을 지원한다.

## 합성 데이터 smoke test

실제 데이터를 다운로드하지 않고 SOH-only/Partial I–V 각각 1 optimizer step, checkpoint reload, 평가, plot, streaming을 CPU에서 확인한다.

```bash
python -m battery_weighted_maml.matr_anp.smoke_test \
  --config configs/matr_partial_iv_anp.yaml \
  --device cpu

pytest -q tests/test_matr_anp.py
```

## 한 fold 짧은 서버 검증

```bash
python -m battery_weighted_maml.matr_anp.train \
  --config configs/matr_partial_iv_anp.yaml \
  --model partial_iv_anp \
  --fold 0 \
  --max-steps 100 \
  --data-root "$BATTERYLIFE_DATA_ROOT" \
  --device cuda
```

resume은 해당 run의 `last.pt`를 지정한다.

```bash
python -m battery_weighted_maml.matr_anp.train \
  --config configs/matr_partial_iv_anp.yaml \
  --model partial_iv_anp \
  --fold 0 \
  --resume outputs/matr_partial_iv_anp/<RUN>/checkpoints/last.pt \
  --data-root "$BATTERYLIFE_DATA_ROOT" \
  --device cuda
```

## 전체 학습

세 모델 × 5 fold를 순차 학습하고 각 best checkpoint 평가까지 한 명령으로 실행하려면 다음을 사용한다.

```bash
python -m battery_weighted_maml.matr_anp.run_suite \
  --config configs/matr_partial_iv_anp.yaml \
  --data-root "$BATTERYLIFE_DATA_ROOT" \
  --device cuda \
  --evaluate
```

suite 진행 목록은 `outputs/matr_partial_iv_anp/suite_<timestamp>/suite_manifest.json`에 매 run마다 갱신된다.

한 모델·한 fold:

```bash
python -m battery_weighted_maml.matr_anp.train \
  --config configs/matr_partial_iv_anp.yaml \
  --model partial_iv_anp \
  --fold 0 \
  --data-root "$BATTERYLIFE_DATA_ROOT" \
  --device cuda
```

세 모델의 5개 fold를 순차 실행하려면 Linux shell에서 다음 한 명령 블록을 사용한다.

```bash
for model in soh_only_anp soh_only_anp_wide partial_iv_anp; do
  for fold in 0 1 2 3 4; do
    python -m battery_weighted_maml.matr_anp.train \
      --config configs/matr_partial_iv_anp.yaml \
      --model "$model" --fold "$fold" \
      --data-root "$BATTERYLIFE_DATA_ROOT" --device cuda || exit 1
  done
done
```

학습이 모두 끝난 뒤 각 best checkpoint도 한 번에 평가할 수 있다.

```bash
find outputs/matr_partial_iv_anp -path '*/checkpoints/best.pt' -print0 | \
while IFS= read -r -d '' checkpoint; do
  python -m battery_weighted_maml.matr_anp.evaluate \
    --config configs/matr_partial_iv_anp.yaml \
    --checkpoint "$checkpoint" \
    --data-root "$BATTERYLIFE_DATA_ROOT" --device cuda || exit 1
done
```

저장소에서 Slurm 사용 근거를 찾지 못해 scheduler 전용 script는 임의로 추가하지 않았다.

## 평가와 streaming

```bash
python -m battery_weighted_maml.matr_anp.evaluate \
  --config configs/matr_partial_iv_anp.yaml \
  --checkpoint outputs/matr_partial_iv_anp/<RUN>/checkpoints/best.pt \
  --fold 0 \
  --data-root "$BATTERYLIFE_DATA_ROOT" \
  --device cuda
```

평가는 held-out test cell 전체에 대해 `alpha={0.3,0.5,0.7}`, `beta={0,0.25,0.5,0.75,1}`를 계산한다. SOH-only 모델은 같은 prediction을 beta 축에 복제하므로 horizontal baseline이 된다. MC latent sampling으로 평균, NLL, 95% interval과 coverage를 계산한다.

```bash
python -m battery_weighted_maml.matr_anp.streaming_demo \
  --config configs/matr_partial_iv_anp.yaml \
  --checkpoint outputs/matr_partial_iv_anp/<RUN>/checkpoints/best.pt \
  --cell-id <HELD_OUT_TEST_CELL> \
  --alpha 0.5 \
  --data-root "$BATTERYLIFE_DATA_ROOT" \
  --device cuda
```

streaming은 beta만 순서대로 늘리며 parameter를 수정하지 않는다. 실행 전후 checksum과 tensor equality를 manifest에 기록하고, feature preprocessing 시간과 pure model inference의 mean/median/std latency를 분리 저장한다.

세 모델과 여러 fold의 평가가 끝난 뒤, 각 evaluation directory를 한 번에 넘기면 모델 비교 CSV와 RMSE/current-error/uncertainty 비교 plot을 만든다.

```bash
python -m battery_weighted_maml.matr_anp.compare_results \
  --evaluations \
    outputs/matr_partial_iv_anp/<SOH_FOLD0>/evaluation/best \
    outputs/matr_partial_iv_anp/<WIDE_FOLD0>/evaluation/best \
    outputs/matr_partial_iv_anp/<PARTIAL_FOLD0>/evaluation/best \
  --output-dir outputs/matr_partial_iv_anp/model_comparison
```

## 출력 경로

기본 run 경로는 `outputs/matr_partial_iv_anp/<timestamp>_fold<k>_<model>_s<seed>/`이다.

```text
checkpoints/best.pt, last.pt
training/history.csv
scalers/fold_scalers.json
audit/data_audit.csv
splits.json
resolved_config.yaml
run_manifest.json
evaluation/<checkpoint>/per_cell_metrics.csv
evaluation/<checkpoint>/aggregate_metrics.csv
evaluation/<checkpoint>/trajectory_predictions.csv
evaluation/<checkpoint>/plots/*.png
streaming/<cell_alpha>/latency.csv
streaming/<cell_alpha>/streaming_trajectory.png
```

현재 Codex 작업공간에서는 `data/MATR` 파일을 확인할 수 없어 real-data inspection/training 수치를 만들지 않았다. 데이터가 있는 서버에서는 먼저 inspection, 그 다음 fold 0의 100-step 검증을 통과시킨 후 전체 15개 run을 실행하는 순서가 안전하다.
