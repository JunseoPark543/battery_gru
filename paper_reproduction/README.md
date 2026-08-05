# GRU Encoder–Decoder Full MAML Reproduction

This package is independent from the repository's weighted-MAML implementation.
It uses the fixed five-cell CALCE meta-training split, an unweighted mean query
loss, and full second-order gradients through the inner SGD update.

## Run

From the repository root:

```bash
python -m paper_reproduction.main \
  --mode all \
  --config paper_reproduction/config.yaml
```

Train only:

```bash
python -m paper_reproduction.main --mode train
```

Meta-test a saved best checkpoint:

```bash
python -m paper_reproduction.main \
  --mode test \
  --checkpoint outputs/paper_reproduction/RUN/checkpoints/best_meta_model.pt
```

Resume at the last completed meta epoch:

```bash
python -m paper_reproduction.main \
  --mode all \
  --resume outputs/paper_reproduction/RUN/checkpoints/last.pt
```

Run another history length:

```bash
python -m paper_reproduction.main --mode all --history-length 100
```

For optional Optuna TPE outer-learning-rate search, install the extra and set
`maml.optuna_trials` above zero:

```bash
python -m pip install -e ".[paper]"
python -m paper_reproduction.main --mode optuna
```

## Explicit implementation choices

- Default training loss: masked MSE (`loss.kind: mae` is also supported).
- Default outer learning rate: `1e-3`.
- Meta early stopping: mean post-adaptation query loss over the five training
  cells, because no separate meta-validation split was specified.
- Complete-adaptation early stopping: a fixed support-only recursive probe batch.
- Maximum rollout cycle: 2000.

CX2_37 and CX2_38 are loaded only after meta-training/checkpoint selection and
are never used in the outer update or early stopping.

