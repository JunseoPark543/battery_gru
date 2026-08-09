# V3 usage

V2 showed that step 0 was better than every adapted checkpoint and that
`delta_y_S` had collapsed to about 5.7 cycles. The evidence and design response
are documented in `RESULT_ANALYSIS_V2_20260810.md`.

V3 adds:

- non-zero Specific-head initialization so BOIL gradients reach `E_S`;
- a detached residual-fit loss;
- a source-only within-protocol pairwise RUL-difference loss;
- source-query adaptation-path and positive-regret losses;
- repeated source validation support splits;
- support-only leave-one-out adaptation-step selection.

Target query labels are never used for step selection. Train V3 from scratch:

```bash
cd ~/바탕화면/battery_gru
source .venv/bin/activate

python -m hust_hybrid_anil_boil.main \
  --config hust_hybrid_anil_boil/config_v3.yaml \
  --method hybrid \
  --fold protocol_1 \
  --seed 42 \
  --device cuda
```

Do not resume a V1/V2 checkpoint with `config_v3.yaml`; the checkpoint loader
rejects that objective mismatch. V3 produces:

```text
aggregate_adaptation_metrics.csv  # fixed-step diagnostic curve
deployment_metrics.csv            # support-only selected deployment result
deployment_predictions.csv        # matching per-cell predictions
```

For the first protocol-1 rerun, use these predeclared checks instead of judging
only the training loss:

- `deployment_metrics.csv` MAE should beat the V2 no-adaptation reference of
  377.33 cycles;
- fixed step-1/2/5 MAE should no longer rise monotonically from step 0;
- `mean_absolute_specific_residual` should be materially larger than V2's
  5.71 cycles, while staying below the configured residual-ratio warning;
- General source-validation domain accuracy should move toward the eight-class
  chance level of 12.5%, while Specific accuracy remains clearly above chance;
- report the selected-step distribution—frequent step 0 is a legitimate safe
  fallback, not a failed evaluation.
