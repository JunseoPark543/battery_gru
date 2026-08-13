# Non-meta baselines

This directory contains experiments with meta-learning and source weighting
disabled.

Run names encode only the main conditions. For example:

`cx2_37_gru_l100_soh_rec_10kstep_noes_s42`

- `cx2_37`: target cell
- `gru`: model family
- `l100`: observed history length
- `soh`: input feature
- `rec`: recursive training and inference
- `10kstep`: optimizer-update budget
- `noes`: no early stopping (`es` means enabled)
- `s42`: random seed

Complete settings and status are stored inside each run in
`config_resolved.yaml` and `run_manifest.json`. An existing run directory is
never overwritten. Continue it with `--resume`, or use `--run-name` for a
distinct variant.

The source-pretrained, target-fine-tuned L100 baseline is run with:

```bash
python scripts/run_source_pretrained_gru_baseline.py \
  --target CALCE_CX2_37.pkl \
  --config configs/baseline/source_pretrained_gru_l100_soh.yaml \
  --device cuda
```

Its automatic directory is
`cx2_37_gru_l100_soh_transfer_samefam_s42`. It uses equal source weights and
ordinary supervised gradients; MAML, higher-order gradients, and MMD-QP are
absent.

The variable-cutpoint generalization experiment is run with:

```bash
python scripts/run_source_pretrained_gru_baseline.py \
  --target CALCE_CX2_37.pkl \
  --config configs/baseline/source_varcut_gru_l100_soh.yaml \
  --device cuda
```

Its automatic directory is `cx2_37_gru_l100_soh_varcut_samefam_s42`. For every
source cell, all points over its complete recorded life are eligible cut points
for `cycles 1..c -> c+1..end`. Each epoch samples 16 lifetime-stratified cuts
per source and always includes `c=100`; teacher forcing remains 0.5. The run
evaluates target zero-shot, fast 1/3/5/10/15/20-step fine-tuning, and full
fine-tuning through the target's complete available future.
