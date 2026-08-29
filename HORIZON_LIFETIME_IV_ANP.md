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
