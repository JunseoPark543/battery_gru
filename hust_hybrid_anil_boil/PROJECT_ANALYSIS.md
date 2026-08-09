# Existing project analysis and integration decisions

## Reused pipeline

The active direct-RUL implementation is `hust_direct_rul_boil`. Its entry point is
`python -m hust_direct_rul_boil.main`. The new study imports its HUST pickle
loader, cycle-10 feature extraction, source-only input normalizer, protocol
parser, raw-cycle metrics, seed handling, and device handling. Existing files are
not modified.

Each cell provides:

- waveform input `[100, 64, 8]`;
- scalar input `[100, 14]`;
- one target, `EOL - 100` raw cycles;
- one domain, parsed from `HUST_<protocol>-<replicate>.pkl`.

The existing hierarchical backbone is reused independently for the General and
Specific encoders: charge-profile CNN, scalar MLP, multi-scale cycle CNN,
Transformer-CBAM stages, and attentive statistics pooling.

## Split and task decision

The outer evaluation is leave-one-protocol-out. The target protocol is absent
from training, input-normalization fitting, early stopping, and checkpoint
selection. Within every source protocol, validation cells are also excluded from
training and normalization.

A direct-RUL cell has only one supervised pair: `(first 100 cycles, RUL at cycle
100)`. It therefore cannot be split into independent labeled support/query
examples without duplicating the same target. The scientifically auditable task
definition is consequently one charging protocol per task, with disjoint cells
from that protocol used as support and query.

At final evaluation, labeled support cells from the unseen protocol are used for
adaptation and the remaining, disjoint query cells are used only for metrics.
This is few-shot domain adaptation, not zero-label domain generalization. Step 0
is included as its no-adaptation reference. The best target step is reported as
an oracle diagnostic and must not be described as a deployable selection rule.

## Reconstruction target

The decoder does not reconstruct the full `[100, 64, 8]` profile. At each cycle
it reconstructs the normalized intermediate summary
`[profile mean (8), profile std (8), scalar features (14)]`, giving
`[batch, 100, 30]`. This retains cycle-wise degradation information while keeping
the auxiliary decoder small.

## Auxiliary-loss placement

All domain, reconstruction, consistency, orthogonality, and residual losses are
computed on post-adaptation query cells from source protocols. Only prediction
loss, optionally plus General prediction loss, enters the inner loop. This keeps
support and query roles separate and prevents target-query leakage.

