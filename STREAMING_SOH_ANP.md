# Streaming SOH Latent ANP

This is a separate uncertainty-aware extension of the non-NP `streaming_soh`
model. Existing models and checkpoints are unchanged.

## Causal model

1. CNN + masked point attention encodes completed and partial V/I-Q curves.
2. A two-layer GRU produces ordered degradation context tokens.
3. Cross-attention supplies a deterministic query-specific context path.
4. A latent context prior `p(z|C)` represents uncertainty over the entire cell
   degradation function.
5. The decoder samples coherent full SOH trajectories and separates epistemic
   and aleatoric uncertainty.

During training only, future targets form `q(z|C,T)`. The ELBO matches that
posterior to `p(z|C)`. Validation, test, and online prediction use only
`p(z|C)`; future SOH never enters inference.

## Train and evaluate

```bash
python -m battery_weighted_maml.streaming_soh_anp.run_suite \
  --config configs/matr_streaming_soh_anp.yaml \
  --fold 0 \
  --device cuda
```

Resume:

```bash
python -m battery_weighted_maml.streaming_soh_anp.run_suite \
  --config configs/matr_streaming_soh_anp.yaml \
  --fold 0 \
  --device cuda \
  --resume outputs/streaming_soh_anp/matr_stream_anp_f0_s42/checkpoints/last.pt
```

## Streaming replay

```bash
python -m battery_weighted_maml.streaming_soh_anp.streaming_demo \
  --checkpoint outputs/streaming_soh_anp/matr_stream_anp_f0_s42/checkpoints/best.pt \
  --cycle 130 \
  --betas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --latent-samples 100 \
  --device cuda
```

Reported uncertainty metrics include Gaussian NLL, CRPS, 50/80/90/95%
coverage, interval width, and epistemic/aleatoric standard deviations.
The best checkpoint and early stopping use validation CRPS by default so model
selection accounts for both trajectory accuracy and predictive uncertainty.
