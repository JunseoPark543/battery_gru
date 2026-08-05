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

Adapt one cell with the deployment-safe defaults (the default cell for
`--mode adapt` is `CALCE_CX2_37.pkl`):

```bash
python -m paper_reproduction.main \
  --mode adapt \
  --checkpoint outputs/paper_reproduction/RUN/checkpoints/best_meta_model.pt \
  --complete-learning-rate 0.005 \
  --scheduler constant \
  --test-cell CALCE_CX2_37.pkl
```

Useful adaptation overrides are `--loss-reduction`, `--sampling-mode`,
`--fast-sampling-mode`, `--gradient-clip-norm`, `--complete-max-steps`, and
`--complete-patience`. Pass `--gradient-clip-norm null` to disable clipping.

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
- Fast adaptation: one continuous random-batch SGD path at learning rate 0.05;
  steps 0, 1, 3, and 5 are snapshots of that path, not independent reruns.
- Complete adaptation: learning rate 0.005, sample-balanced recursive loss,
  length-stratified batches, and global gradient clipping at 1.0 by default.
- Complete checkpoint selection and early stopping: recursive MAE on the last
  chronological 20% of the observed support (bounded to 20–100 points).
- Oracle checkpoint selection by query MAE is saved only as a clearly labeled
  diagnostic and is never used as deployment performance.
- Paper evaluation horizon: exact held-out query length.
- Legacy `max_forecast_cycle: 2000`: retained only for backward-compatible
  deployment mode when `max_prediction_length` is not supplied.
- Checkpoint selection: deterministic post-outer-update meta query loss.

SOH JSON/CSV summaries retain raw fraction metrics (`mae`, `rmse`) and also
report paper-comparison units (`mae_percent`, `rmse_percent`). Figure titles and
console meta-test logs show percent values.

CX2_37 and CX2_38 are loaded only after meta-training/checkpoint selection and
are never used in the outer update or meta early stopping. During meta-test,
query labels are used under `torch.no_grad()` for metrics/oracle diagnostics;
only chronological support validation can select the deployment checkpoint.

Every complete trajectory writes per-step support/query metrics, pre/post-clip
gradient norms, update norms, sampled split indices, recursive forecast
endpoints, EOL, and RUL to `adaptation/adaptation_diagnostics.csv`. The selected
deployment state and the final state are both retained.

## CX2_37 L=500 comparison matrix

The baseline measured before the adaptation rewrite is preserved in
`paper_reproduction/baseline_cx2_37_l500.csv`. Run experiments A–F with:

```bash
python -m paper_reproduction.run_adaptation_experiments \
  --checkpoint outputs/paper_reproduction/RUN/checkpoints/best_meta_model.pt \
  --device cuda:0
```

The script writes `experiment_comparison.csv`, one diagnostics CSV per
experiment, and best/final complete states. Its default is the configured
500-step maximum with chronological-validation early stopping. The optional
`--complete-max-steps` flag is for smoke/debug runs; results produced with it
must not be reported as the full 500-step comparison.

For C–E, point-balanced loss, random sampling, and no clipping are held fixed so
only complete learning rate changes. F is the stabilized combination:
sample-balanced loss, length-stratified sampling, and clip norm 1.0.

## Output layout

A single-cell `--mode adapt` run writes:

```text
<run>/
├── config.yaml
├── preprocessing_summary.csv
├── checkpoints/
│   ├── meta_model.pt
│   ├── complete_best_model.pt
│   └── complete_final_model.pt
├── adaptation/
│   ├── adaptation_diagnostics.csv
│   ├── fast_0_metrics.json
│   ├── fast_1_metrics.json
│   ├── fast_3_metrics.json
│   ├── fast_5_metrics.json
│   ├── complete_deployment_safe_metrics.json
│   └── complete_oracle_diagnostic_metrics.json
└── plots/
    ├── adaptation_step_vs_support_loss.png
    ├── adaptation_step_vs_query_mae.png
    ├── adaptation_step_vs_gradient_norm.png
    ├── adaptation_step_vs_update_norm.png
    └── recursive_forecast_by_step.png
```

The current CALCE pickle gives CX2_37 1,072 cleaned cycles and therefore 572
query points after L=500. `preprocessing_summary.csv` records the corresponding
counts and extraction choices for all seven cells, making a different paper
source/preprocessing version visible rather than silently forcing equal lengths.

## Run-directory naming

New run names contain the settings that most often change, plus an eight-digit
fingerprint of the complete resolved configuration. For example:

```text
20260806_101530_train_stabilized_L500_mi1_ilr0p05_olr0p001_ml-sb_s42_c12ab34cd

20260806_111500_adapt_stabilized_L500_CX2_37_flr0p05_clr0p005_al-sb_sp-ls_cp1_sc-const_s42_c56ef78ab
```

Abbreviations are `mi` (MAML inner steps), `ilr`/`olr` (inner/outer LR),
`flr`/`clr` (fast/complete LR), `ml`/`al` (meta/adaptation reduction), `sp`
(complete sampling), `cp` (clip), `sc` (scheduler), `s` (seed), and `c`
(config fingerprint). `sb`/`pb` mean sample/point-balanced and `ls`/`rnd`
mean length-stratified/random. Decimal points use `p`, so `0p005` means
`0.005`.

The fingerprint changes when any resolved setting changes, including a setting
not shown in the readable portion. `config.yaml` and `run_manifest.json` remain
the authoritative full record. Existing run directories are intentionally not
renamed.

## Tests

```bash
python -m compileall -q paper_reproduction
python -m pytest tests/test_paper_reproduction.py -q
python -m pytest -q
```
