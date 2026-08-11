# CALCE SOH-only GRU baseline suite

Four matched experiments are provided for `CALCE_CX2_37.pkl`:

1. `nometa_l500`: plain GRU encoder-decoder, target SOH only, L=500
2. `nometa_l100`: plain GRU encoder-decoder, target SOH only, L=100
3. `meta_l500`: target-aware weighted second-order MAML, SOH only, L=500
4. `meta_l100`: target-aware weighted second-order MAML, SOH only, L=100

Both model implementations have the same scalar GRU encoder, scalar GRU decoder,
hidden size 64, one recurrent layer, linear SOH head, and teacher-forcing ratio
0.5. Every experiment recursively forecasts from `L+1` through the target's
final observed cycle.

The two non-meta configs use the same supervised settings (`encoder_window=50`,
`forecast_horizon=10`, `validation_cycles=20`). This differs from the historical
`configs/gru_baseline_l500.yaml` so that L is the only configuration difference
inside the new non-meta pair. The L100 prefix cannot support the historical
`100 + 25 + 50` train/validation construction.

The meta pair differs only in L. Both use same-family CX2 sources, MMD-QP target
weights, full second-order MAML, 2300 iterations, and fast/full target adaptation.

Run one experiment:

```bash
python scripts/run_calce_gru_baseline_suite.py \
  --experiment nometa_l500 \
  --target CALCE_CX2_37.pkl \
  --device cuda
```

Replace the experiment with `nometa_l100`, `meta_l500`, or `meta_l100`.

Run all four sequentially:

```bash
python scripts/run_calce_gru_baseline_suite.py \
  --experiment all \
  --target CALCE_CX2_37.pkl \
  --source-mode same_family \
  --device cuda
```

Individual results remain under `outputs/runs/`. The suite runner also writes a
small index containing all four paths under
`outputs/calce_gru_baseline_suites/<timestamp>/suite_manifest.json`.
The same directory contains `baseline_comparison.csv`,
`baseline_comparison.json`, and `baseline_comparison.png` with matched
trajectory and RUL metrics. These files are updated after each completed run,
so already-finished experiments remain easy to inspect if a later run stops.
