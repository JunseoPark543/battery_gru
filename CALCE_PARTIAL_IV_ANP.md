# CALCE SOH-only / Partial I-V ANP

MATR 실험과 같은 ANP 구조와 hyperparameter를 CALCE에 적용한다. 두 모델의 차이는 입력 조건뿐이다.

- `soh_only_anp`: 관측된 cycle-SOH context만 사용
- `partial_iv_anp`: 같은 SOH context에 현재 cycle의 partial `[delta V, |I|, mask]`를 추가

SOH는 각 cycle의 `max(discharge_capacity_in_Ah) / nominal_capacity_in_Ah`로 계산한다. CALCE EOL JSON은 이 trajectory 예측 학습에 사용하지 않는다. split과 scaler는 cell 단위이며 train/validation/test cell이 겹치지 않는다.

## Fold 0에서 두 모델 학습 후 평가

```bash
cd ~/바탕화면/battery_gru
source .venv/bin/activate
git pull --rebase origin main

python -m battery_weighted_maml.calce_anp.run_suite \
  --device cuda \
  --models soh_only_anp partial_iv_anp \
  --folds 0 \
  --evaluate
```

결과는 `outputs/calce_partial_iv_anp/`에 저장된다. suite 폴더의 `model_comparison/`에는 두 모델의 통합 CSV와 RMSE 비교 plot이 자동 생성된다. 먼저 짧게 확인하려면 위 명령에 `--max-steps 100`을 추가한다.

## 전체 5-fold 비교

```bash
python -m battery_weighted_maml.calce_anp.run_suite \
  --device cuda \
  --models soh_only_anp partial_iv_anp \
  --folds 0 1 2 3 4 \
  --evaluate
```

중단된 단일 run은 해당 모델명과 fold를 그대로 지정해 이어간다.

```bash
python -m battery_weighted_maml.calce_anp.train \
  --model partial_iv_anp --fold 0 --device cuda \
  --resume outputs/calce_partial_iv_anp/<RUN>/checkpoints/last.pt
```
