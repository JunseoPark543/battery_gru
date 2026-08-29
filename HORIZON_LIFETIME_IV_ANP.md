# MATR Horizon Lifetime I-V ANP

This experiment is isolated from `horizon_rul_anp` and does not modify its
checkpoints or results.

## Definition

- Observation horizons: `100,120,...,300`.
- Each ANP point is one battery cell observed through the shared horizon `k`.
- Inputs per observed cycle: normalized cycle, normalized SOH, and the complete
  discharge voltage/current curve interpolated on 256 fixed `q=Qd/Qnom` points.
- `deltaSOH` is not used.
- A masked patch projection converts 256 curve points into 32 tokens. A
  within-cycle Transformer encodes the curve, and a second Transformer encodes
  the sequence of cycles `1..k` into `h`.
- Context is `{(h_i, lifetime_i)}` using train cells only.
- A held-out query supplies `h_q` but never its lifetime during inference.
- The model predicts Gaussian lifetime uncertainty. RUL is derived as
  `predicted_lifetime - k` with unchanged standard deviation.

All normalization statistics are fitted on the train split only. The
validation and test cells never enter context. Evaluation and streaming do not
update model parameters.

## Paired-horizon consistency loss

The optional experiment samples the same context and query cells at an early
horizon `k` and a later horizon `k+gap`. Both horizons keep the ordinary
lifetime ELBO. The early inference prediction is also trained toward the
detached later inference prediction:

`L = 0.5*(ELBO_k + ELBO_k+gap) + lambda*Huber(mu_k, stopgrad(mu_k+gap))`.

Only prior/inference predictions are compared, so query lifetime labels cannot
leak into this term. `lambda` is linearly warmed up. The original configuration
keeps `paired_horizon_training: false` and remains the unchanged baseline.

## Train and evaluate

```bash
python -m battery_weighted_maml.horizon_lifetime_iv_anp.run_suite \
  --config configs/matr_horizon_lifetime_iv_anp.yaml \
  --fold 0 \
  --device cuda \
  --evaluate
```

## Resume

```bash
python -m battery_weighted_maml.horizon_lifetime_iv_anp.run_suite \
  --config configs/matr_horizon_lifetime_iv_anp.yaml \
  --fold 0 \
  --device cuda \
  --resume outputs/horizon_lifetime_iv_anp/RUN/checkpoints/last.pt \
  --evaluate
```

## Streaming held-out cell

```bash
python -m battery_weighted_maml.horizon_lifetime_iv_anp.streaming \
  --config configs/matr_horizon_lifetime_iv_anp.yaml \
  --checkpoint outputs/horizon_lifetime_iv_anp/RUN/checkpoints/best.pt \
  --device cuda \
  --cell MATR_b1c14 \
  --horizons 100 120 140 160 180 200 220 240 260 280 300
```

## Sequential consistency ablation

On one GPU, run variants sequentially to avoid out-of-memory errors. The suite
contains a compute-matched paired control (`lambda=0`), three loss weights, and
one wider horizon gap. It evaluates each best checkpoint and writes
`suite_summary.csv` plus `suite_comparison.png`.

```bash
python -m battery_weighted_maml.horizon_lifetime_iv_anp.run_consistency_suite \
  --config configs/matr_horizon_lifetime_iv_anp_consistency.yaml \
  --fold 0 \
  --device cuda \
  --baseline-run outputs/horizon_lifetime_iv_anp/BASELINE_RUN
```
