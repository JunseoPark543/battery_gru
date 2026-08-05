# GRU Encoder–Decoder Full MAML Reproduction

This package is independent from the repository's weighted-MAML implementation.
It uses the fixed five-cell CALCE meta-training split, an unweighted mean query
loss, and full second-order gradients through the inner SGD update.

## Environment and dependencies

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[paper]"
```

Windows PowerShell activation is:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[paper]"
```

The `paper` extra installs Optuna in addition to torch, higher, NumPy, pandas,
Matplotlib, PyYAML, and tqdm.

## Training and meta-test

From the repository root:

```bash
python -m paper_reproduction.main \
  --mode all \
  --config paper_reproduction/config.yaml \
  --device auto
```

Train only:

```bash
python -m paper_reproduction.main --mode train
```

Meta-test a saved best checkpoint:

```bash
python -m paper_reproduction.main \
  --mode test \
  --config paper_reproduction/config.yaml \
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

Force CPU execution:

```bash
python -m paper_reproduction.main --mode all --device cpu
```

Select a CUDA device explicitly (the teacher-forcing RNG uses the same index):

```bash
python -m paper_reproduction.main --mode all --device cuda:0
python -m paper_reproduction.main --mode all --device cuda:1
```

`--device auto` falls back to CPU when CUDA is unavailable. An explicitly
requested unavailable CUDA device raises a clear error instead of silently
running on another device.

## Evaluation modes

The default `paper` mode rolls out exactly `len(query_soh)` steps for each test
cell, including non-contiguous cycle labels:

```bash
python -m paper_reproduction.main --mode test --forecast-mode paper \
  --checkpoint outputs/paper_reproduction/RUN/checkpoints/best_meta_model.pt
```

Deployment-only extrapolation is separate and does not affect paper metrics:

```bash
python -m paper_reproduction.main --mode test --forecast-mode deployment \
  --max-prediction-length 1000 \
  --checkpoint outputs/paper_reproduction/RUN/checkpoints/best_meta_model.pt
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
- Paper evaluation horizon: exact held-out query length.
- Legacy `max_forecast_cycle: 2000`: retained only for backward-compatible
  deployment mode when `max_prediction_length` is not supplied.
- Checkpoint selection: deterministic post-outer-update meta query loss.

SOH JSON/CSV summaries retain raw fraction metrics (`mae`, `rmse`) and also
report paper-comparison units (`mae_percent`, `rmse_percent`). Figure titles and
console meta-test logs show percent values.

CX2_37 and CX2_38 are loaded only after meta-training/checkpoint selection and
are never used in the outer update or early stopping.

## Tests

```bash
python -m compileall -q paper_reproduction
python -m pytest tests/test_paper_reproduction.py -q
python -m pytest -q
```
