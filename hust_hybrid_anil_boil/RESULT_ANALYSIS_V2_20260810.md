# V2 result analysis and V3 response

Analyzed run:

`20260810_025857_758973_hust_hybrid_L100_fo_inner5_valdomain_sc0p02_rep4_seed42`

## What improved

Across four deterministic target-support splits, no-adaptation MAE improved from
459.96 to 377.33 cycles and RMSE improved from 647.09 to 503.04 cycles.  The
single-split bias in V1 was removed: every protocol-1 cell is used as support in
one repeat and as query in the other repeats.

## Remaining failures

1. Adaptation is harmful on the unseen protocol.  MAE is 377.33 at step 0,
   383.50 at step 1, 391.86 at step 2, 412.26 at step 5, and 440.84 at step 10.
   R-squared also falls from -0.151 at step 0 to -0.441 at step 5 and -0.643
   at step 10.
2. The Specific branch has collapsed as a predictor.  At step 5, mean
   `|delta_y_S|` is only 5.71 cycles, while `y_G` alone has MAE 408.73 cycles.
   Adding the Specific residual worsens final MAE to 412.26 cycles.
3. V2 initializes the last Specific residual layer to exactly zero.  Because
   BOIL freezes this head and adapts only `E_S`, the initial prediction gradient
   into `E_S` is also exactly zero.  That conflicts with the intended BOIL path.
4. In the last 400 iterations, General-domain training accuracy averages 52.3%
   for eight training protocols (chance is 12.5%).  General features therefore
   retain substantial source-protocol identity.
5. Protocol 1 is intrinsically difficult: `HUST_1-6` has RUL 1079 but the
   no-adaptation prediction is about 2125 cycles, whereas `HUST_1-2` and
   `HUST_1-8` are under-predicted by roughly 640 and 630 cycles.  Two support
   cells can have a mean RUL from 1506 to 2048.5 cycles, so a fixed five-step
   update is unsafe.
6. The prediction scatter shows regression to the mean and incorrect within-
   protocol ordering: the shortest-life target cell is strongly over-predicted,
   while the longest-life cells are under-predicted.

## V3 changes

`config_v3.yaml` and the supporting code make the following changes without
using target-query labels:

- small non-zero Specific-head initialization so BOIL gradients reach `E_S`;
- an explicit detached residual-fit objective for `delta_y_S`;
- a source-only within-protocol pairwise RUL-difference loss to preserve cell
  ordering and spacing instead of regressing every battery toward the mean;
- source-query adaptation-path and positive-regret losses at steps 0/1/2/5;
- four support rotations for source-domain checkpoint validation;
- stronger General GRL pressure because V2 was not domain invariant;
- a smaller General-head inner learning rate, retaining the Specific-encoder LR;
- support-only leave-one-out step selection with a conservative 25-cycle gain
  requirement; step 0 is a valid safe fallback;
- separate `deployment_metrics.csv` and `deployment_predictions.csv`, so the
  deployment-safe result is not confused with oracle diagnostic curves.

V3 must be trained from scratch.  V2 checkpoints must not be resumed because
the training objective and residual-head initialization changed.
