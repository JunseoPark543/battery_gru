"""MMD weighting, MAML, and target adaptation."""

from .kernel_weights import KernelWeightResult, compute_target_aware_weights
from .maml import TaskMetaLoss, adapt_source_task, weighted_meta_loss
from .target_adaptation import AdaptationResult, adapt_target

__all__ = [
    "KernelWeightResult",
    "compute_target_aware_weights",
    "TaskMetaLoss",
    "adapt_source_task",
    "weighted_meta_loss",
    "AdaptationResult",
    "adapt_target",
]

