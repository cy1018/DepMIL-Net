"""Public DepMIL-Net model interface."""

from .depmil_net import (
    FEATURE_DIM,
    Motion32Stage1Model,
    SeverityAwareProtoDual,
    cumulative_ordinal_targets,
)

__all__ = [
    "FEATURE_DIM",
    "Motion32Stage1Model",
    "SeverityAwareProtoDual",
    "cumulative_ordinal_targets",
]
