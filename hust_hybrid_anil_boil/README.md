# HUST General-ANIL + Specific-BOIL Direct-RUL

기존 `hust_direct_rul_boil` 코드는 변경하지 않고 추가한 독립 연구 패키지입니다. HUST의 첫 100 cycle에서 cycle 100 시점의 raw RUL(`EOL - 100`)을 직접 예측합니다.

## 핵심 구조

```text
waveform [B,100,64,8] + scalar [B,100,14]
                 |                         |
        General Encoder E_G       Specific Encoder E_S
                 | z_G                     | z_S
        +--------+--------+        +--------+--------+
        |                 |        |                 |
  GRL Domain Head    General Head  Domain Head   Residual Head
                         y_G                        delta_y_S
                           \                        /
                            y_hat = y_G + delta_y_S

concat(z_G, z_S) -> cycle-summary decoder -> [B,100,30]
```

General branch는 공통 degradation 정보를 재사용하도록 domain-adversarial하게 학습합니다. Specific branch는 protocol을 구분하도록 학습하며, 최종 예측에서 작은 signed correction을 담당합니다.

재구성 대상은 raw waveform 전체가 아닙니다. 정규화된 입력에서 cycle별 profile mean 8개, profile std 8개, scalar 14개를 합친 `[B,100,30]` intermediate summary를 복원합니다.

## Meta task와 누수 방지

직접 RUL 데이터는 cell 하나당 `(첫 100 cycles, RUL 하나)`만 있으므로 cell 내부에서 독립적인 support/query label을 만들 수 없습니다. 따라서 다음처럼 정의합니다.

- task: 하나의 charging protocol
- source support/query: 같은 protocol의 서로 다른 battery cell
- outer test: protocol 하나를 통째로 unseen target으로 제외
- target support: unseen protocol의 labeled cell 일부
- target query: support와 겹치지 않는 나머지 cell; metrics에만 사용

따라서 step 1/2/5/10은 few-shot target adaptation입니다. Step 0은 target label을 전혀 쓰지 않는 no-adaptation 기준입니다. Query 성능으로 고른 best step은 연구용 oracle diagnostic일 뿐 실제 배포 선택 규칙이 아닙니다.

## Inner-loop parameter 정책

| Method | Inner-loop update | Inner-loop freeze |
|---|---|---|
| Supervised | 없음 | 전체 |
| Full MAML | 두 encoder + 두 RUL head | domain head, decoder |
| ANIL | General/Specific RUL head | 두 encoder, domain head, decoder |
| BOIL | General/Specific encoder | RUL head, domain head, decoder |
| Proposed | General RUL head + Specific encoder | General encoder + Specific residual head + auxiliary modules |

`concat` prediction ablation에서는 하나의 concat head가 RUL head 역할을 합니다. 모든 inner-loop frozen parameter도 outer query objective에서는 학습되는 meta-parameter입니다.

## Loss

```text
L = lambda_T     * MSE((y_G + delta_y_S)/500, y/500)
  + lambda_GY    * MSE(y_G/500, y/500)
  + lambda_G     * CE(GRL(z_G), domain)
  + lambda_S     * CE(z_S, domain)
  + lambda_R     * MSE(reconstructed_summary, input_summary)
  + lambda_C     * state-aware cross-domain consistency
  + lambda_O     * normalized cross-covariance orthogonality
  + lambda_delta * mean((delta_y_S/500)^2)
```

`/500`은 dataset label 표준화가 아니라 고정된 물리 cycle scale을 이용한 loss scaling입니다. 출력과 CSV는 실제 cycle 단위입니다. Consistency는 서로 다른 source protocol이면서 `RUL/500`이 가까운 sample pair에만 적용됩니다.

기본 inner loss는 `L_T`이고, `inner_general_prediction_beta`를 0보다 크게 지정하면 `L_T + beta * L_GY`가 됩니다. Domain/GRL/reconstruction/consistency/orthogonality loss는 inner loop에 들어가지 않습니다.

## 실행 전 검사

데이터 schema와 feature만 확인하며 모델을 만들지 않습니다.

```bash
cd ~/바탕화면/battery_gru
source .venv/bin/activate

python -m hust_hybrid_anil_boil.main \
  --config hust_hybrid_anil_boil/config.yaml \
  --inspect-data-only
```

사용자가 직접 실행할 synthetic forward/backward 및 parameter-policy 검사는 다음과 같습니다.

```bash
python -m hust_hybrid_anil_boil.verification
```

이 검사는 실제 HUST 학습을 하지 않으며 forward/backward, GRL 부호, reconstruction shape, selective inner update, outer gradient, checkpoint state round-trip을 확인합니다.

## Proposed 한 fold 실행

```bash
python -m hust_hybrid_anil_boil.main \
  --config hust_hybrid_anil_boil/config.yaml \
  --method hybrid \
  --fold protocol_1 \
  --seed 42 \
  --device cuda
```

## 결과 분석 후 개선한 V2

Protocol 1의 첫 실행에서는 step-1 MAE 457.71 cycles, R² -0.538이었고,
Specific representation cosine change가 0.000155에 불과했습니다. 분석과
변경 근거는 `RESULT_ANALYSIS_20260810.md`에 기록했습니다.

개선 설정은 기존 checkpoint를 resume하지 말고 처음부터 학습합니다.

```bash
python -m hust_hybrid_anil_boil.main \
  --config hust_hybrid_anil_boil/config_v2.yaml \
  --method hybrid \
  --fold protocol_1 \
  --seed 42 \
  --device cuda
```

V2는 unseen source protocol checkpoint selection, prediction warm-up,
Specific supervised contrastive loss, five-step inner adaptation, auxiliary
loss ramp, 네 개 target support split 반복 평가를 사용합니다.

기본은 계산량을 줄인 first-order입니다. Second-order는 다음 옵션을 추가합니다.

```bash
python -m hust_hybrid_anil_boil.main \
  --config hust_hybrid_anil_boil/config.yaml \
  --method hybrid \
  --fold protocol_1 \
  --seed 42 \
  --device cuda \
  --second-order
```

## 5개 방법을 같은 split/seed로 비교

먼저 한 fold로 확인하는 것을 권장합니다.

```bash
python -m hust_hybrid_anil_boil.main \
  --config hust_hybrid_anil_boil/config.yaml \
  --method all \
  --fold protocol_1 \
  --seed 42 \
  --device cuda
```

전체 10-fold 비교는 학습이 50회이므로 시간이 많이 필요합니다.

```bash
python -m hust_hybrid_anil_boil.main \
  --config hust_hybrid_anil_boil/config.yaml \
  --method all \
  --fold all \
  --seed 42 \
  --device cuda
```

## Resume

Resume은 method/fold/seed 하나를 명확히 지정합니다.

```bash
CKPT=$(ls -t outputs/hust_hybrid_anil_boil/*/method_hybrid/fold_protocol_1_seed42/checkpoints/last.pt | head -1)

python -m hust_hybrid_anil_boil.main \
  --config hust_hybrid_anil_boil/config.yaml \
  --method hybrid \
  --fold protocol_1 \
  --seed 42 \
  --device cuda \
  --resume "$CKPT"
```

## Ablation 예시

```bash
# GRL 제거
python -m hust_hybrid_anil_boil.main --method hybrid --fold protocol_1 --device cuda --no-grl

# Specific domain classifier 제거
python -m hust_hybrid_anil_boil.main --method hybrid --fold protocol_1 --device cuda --no-specific-domain

# Reconstruction 제거
python -m hust_hybrid_anil_boil.main --method hybrid --fold protocol_1 --device cuda --no-reconstruction

# L_GY 제거
python -m hust_hybrid_anil_boil.main --method hybrid --fold protocol_1 --device cuda --no-general-prediction-loss

# residual 합 대신 concat single head
python -m hust_hybrid_anil_boil.main --method hybrid --fold protocol_1 --device cuda --prediction-mode concat
```

모든 lambda는 CLI에서 바꿀 수 있습니다.

```bash
python -m hust_hybrid_anil_boil.main \
  --method hybrid --fold protocol_1 --device cuda \
  --lambda-general-domain 0.05 \
  --lambda-specific-domain 0.05 \
  --lambda-reconstruction 0.01 \
  --lambda-consistency 0.01 \
  --lambda-orthogonal 0.01 \
  --lambda-residual 0.001
```

## 결과 구조

```text
outputs/hust_hybrid_anil_boil/<RUN>/
├── command.txt
├── dataset_summary.json
├── resolved_config.yaml
├── method_comparison.csv
├── method_comparison.png
├── aggregate_adaptation_metrics.csv
├── proposed_method_summary.json
└── method_hybrid/
    ├── representation_tsne.png
    └── fold_protocol_1_seed42/
        ├── source_split.json
        ├── inner_policy.json
        ├── protocol.json
        ├── checkpoints/best.pt
        ├── checkpoints/last.pt
        ├── training_history.csv
        ├── training_diagnostics.png
        ├── target_predictions.csv
        ├── adaptation_metrics.csv
        ├── adaptation_curve.png
        ├── primary_features.npz
        └── evaluation_manifest.json
```

`target_predictions.csv`에는 `cell_id, domain, target_y, y_G, delta_y_S, y_hat, absolute_error`가 저장됩니다. `adaptation_metrics.csv`에는 step별 MAE/RMSE/MAPE/R², General/Specific representation cosine·L2 변화, source-validation domain accuracy, residual 크기가 저장됩니다.
