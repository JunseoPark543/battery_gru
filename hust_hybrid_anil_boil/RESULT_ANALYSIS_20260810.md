# Result analysis: protocol_1, seed 42

Analyzed run:

`20260810_024249_132081_hust_hybrid_L100_fo_inner1_seed42`

## Observed result

| Step | MAE | RMSE | R² | Specific cosine change |
|---:|---:|---:|---:|---:|
| 0 | 459.96 | 647.09 | -0.540 | 0.000000 |
| 1 | 457.71 | 646.76 | -0.538 | 0.000155 |
| 2 | 455.77 | 646.59 | -0.538 | 0.000539 |
| 5 | 451.39 | 646.71 | -0.538 | 0.002274 |
| 10 | 446.89 | 647.97 | -0.544 | 0.005549 |

The General representation stayed unchanged as intended. The Specific
representation changed far too little at one step, so the BOIL path was weak.
The final correction was also small: mean `|delta_y_S|=31.65` cycles versus mean
`|y_G|=1908.59` cycles at step 1.

General domain accuracy was 0.111, exactly the chance level for nine source
classes, which is consistent with domain invariance. Specific domain accuracy
was only 0.333 on source-validation cells even though training `L_S` approached
zero. This indicates protocol-classification overfitting rather than reusable
Specific domain structure.

The aggregate RMSE was dominated by `HUST_1-6.pkl`: target 1079 cycles, step-1
prediction 2453.64 cycles, error 1374.64 cycles. A single fixed two-cell support
split therefore gives a fragile estimate for this eight-cell protocol.

## V2 changes

`config_v2.yaml` addresses the observed failure modes without changing the
original `config.yaml` behavior.

1. Checkpoint selection now holds out an entire source protocol. It matches the
   final unseen-protocol evaluation better than validation cells drawn from
   protocols already seen during training.
2. The first 100 iterations train the RUL objectives only. Auxiliary losses ramp
   in over the next 200 iterations instead of dominating regression from the
   first update.
3. General-domain and Specific-domain coefficients are reduced. GRL maximum
   strength is reduced from 1.0 to 0.5.
4. A supervised contrastive loss is added to Specific features so different
   cells from the same protocol cluster without relying only on a classifier.
5. Domain CE uses 0.05 label smoothing.
6. The hybrid inner loop uses five steps with learning rates 0.02, aligned with
   the primary five-step evaluation. The original one-step update was too weak.
7. The inner loss includes `0.25 * L_GY` to keep General head adaptation stable.
8. All eight meta-training protocols participate per iteration, giving each
   protocol positive pairs for contrastive learning and more stable gradients.
9. Four deterministic, disjointly shifted target support splits are evaluated.
   This reduces dependence on one lucky/unlucky pair and ensures that extreme
   cells such as `HUST_1-6.pkl` appear as both support and query across repeats.
10. Orthogonality uses the requested squared Frobenius sum in V2. The original
    config retains its prior mean reduction for reproducibility.

These changes require training from scratch. A V1 checkpoint must not be resumed
with `config_v2.yaml` because the source-domain split and classifier size differ.

