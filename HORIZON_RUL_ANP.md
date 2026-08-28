# Horizon-conditioned MATR RUL ANP

This is a separate prototype that predicts direct RUL. It does not modify the
existing SOH-trajectory ANP or HS-ANP models.

## Data and labels

The established BatteryLife loader in `matr_anp/data.py` reads each MATR
pickle. The new code attaches the server label file:

```text
data/Life labels/MATR_labels.json
{"MATR_b4c36.pkl": 599, ...}
```

For cell `i`, the JSON EOL cycle is `L_i`. At horizon `k`, the exact label is:

```text
RUL_i(k) = L_i - k
```

Only cells with `L_i > k` and an observed cycle `k` are eligible. Cycles after
the labeled EOL are excluded. Every normalization statistic is fitted using
training cells only.

## Task definition

One NP task is an observation horizon `tau_k`, not a battery trajectory.

```text
context point = one train/reference battery observed through k
query point   = another battery observed through the same k
x_i(k)        = PrefixEncoder(cycles 1:k)
y_i(k)        = L_i - k
```

During training, context and query are distinct train cells. During validation
and test, context always consists of train/reference cells, while queries are
validation or held-out test cells. Cell IDs are split before any horizons are
generated, so different prefixes of one cell cannot cross splits.

## Prefix and ANP flow

The first prototype deliberately uses no pairwise/inter-cell auxiliary branch.
Its causal per-cycle features are:

```text
[normalized cycle, normalized SOH, normalized delta-SOH]
```

Tensor flow:

```text
context/query prefix       [B_task, N_cell, K, 3]
prefix masks               [B_task, N_cell, K]
Prefix Transformer output  [B_task, N_cell, D]
context/query cell masks   [B_task, N_cell]
context RUL                [B_task, N_context, 1]
query prediction mean/std  [B_task, N_query, 1]
inter-cell attention       [B_task, N_query, N_context]
```

The deterministic path cross-attends from a query battery representation to
train/reference battery representations. The latent path constructs `q(z|C)`
and, during training only, `q(z|C union Q)`. The decoder outputs a Gaussian RUL
mean and positive standard deviation.

The optimized loss is query-only Gaussian NLL plus:

```text
beta_KL * KL(q(z | C union Q) || q(z | C))
```

RUL is standardized using train cells for optimization and converted back to
cycle units for RMSE, MAE, MAPE, plots, and CSV files.

## Commands

Train fold 0:

```bash
python -m battery_weighted_maml.horizon_rul_anp.train \
  --config configs/matr_horizon_rul_anp.yaml \
  --fold 0 \
  --device cuda
```

Train and immediately evaluate the best checkpoint:

```bash
python -m battery_weighted_maml.horizon_rul_anp.run_suite \
  --config configs/matr_horizon_rul_anp.yaml \
  --fold 0 \
  --device cuda \
  --evaluate
```

Resume:

```bash
python -m battery_weighted_maml.horizon_rul_anp.train \
  --config configs/matr_horizon_rul_anp.yaml \
  --fold 0 \
  --device cuda \
  --resume outputs/horizon_rul_anp/RUN/checkpoints/last.pt
```

Evaluate selected horizons:

```bash
python -m battery_weighted_maml.horizon_rul_anp.evaluate \
  --config configs/matr_horizon_rul_anp.yaml \
  --checkpoint outputs/horizon_rul_anp/RUN/checkpoints/best.pt \
  --device cuda \
  --horizons 20 40 60 80 100
```

Streaming inference at every cycle without gradient updates:

```bash
python -m battery_weighted_maml.horizon_rul_anp.streaming \
  --config configs/matr_horizon_rul_anp.yaml \
  --checkpoint outputs/horizon_rul_anp/RUN/checkpoints/best.pt \
  --device cuda \
  --cell MATR_b1c14 \
  --start 20 --end 100 --step 1
```

## Scope of this prototype

Implemented: horizon-conditioned tasks, prefix representation, standard ANP,
uncertainty, cell-level split, validation, checkpoint/resume, held-out metrics,
and no-update streaming inference.

Not implemented: explicit pairwise difference encoder, BatLiNet inter-cell
branch, delta-lifetime loss, learned reference selection, fusion gate,
test-time fine-tuning, or online gradient updates.
