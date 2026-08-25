# Streaming V/I-conditioned SOH trajectory model

This package predicts the current and future SOH trajectory while samples from
the current discharge cycle arrive. Existing ANP and partial V-Q packages are
not modified.

## Causal inputs

- completed cycles: measured SOH and full discharge V/I-Q curves
- current cycle: only the V/I-Q prefix observed so far
- q is accumulated discharged Ah divided by nominal Ah; final cycle capacity,
  final duration, EOL, and future measurements are never input features

For each new prefix of the same cycle, the model recomputes a candidate state
from the immutable state of the previous completed cycle. It does not feed the
same cycle through the GRU repeatedly.

## Model

1. CNN + masked point attention encodes each V/I-Q curve.
2. A two-layer GRU tracks degradation across completed cycles.
3. The current partial curve creates a temporary current-cycle state.
4. A coordinate decoder predicts current/future SOH mean and uncertainty.
5. V-Q completion and q-end heads provide auxiliary training supervision.

## Commands

Train and test fold 0:

```bash
python -m battery_weighted_maml.streaming_soh.run_suite \
  --config configs/matr_streaming_soh.yaml \
  --fold 0 \
  --device cuda
```

Resume training:

```bash
python -m battery_weighted_maml.streaming_soh.run_suite \
  --config configs/matr_streaming_soh.yaml \
  --fold 0 \
  --device cuda \
  --resume outputs/streaming_soh/matr_stream_soh_f0_s42/checkpoints/last.pt
```

Replay cycle 130 as progressively arriving data:

```bash
python -m battery_weighted_maml.streaming_soh.streaming_demo \
  --checkpoint outputs/streaming_soh/matr_stream_soh_f0_s42/checkpoints/best.pt \
  --cycle 130 \
  --betas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --device cuda
```

The streaming replay changes the inference state and forecast, not the trained
weights. Weight adaptation should only be performed after the completed cycle
reveals its measured SOH label.

For a live system, use `OnlineSOHSession.observe(q, voltage, current)`. If the
cycler does not directly provide accumulated discharge capacity, construct q
causally with `integrate_discharge_q(time_s, current_a, nominal_capacity_ah)`.
