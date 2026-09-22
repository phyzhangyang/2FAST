"""Public interface for 2FAST."""

from twofast.estimator import (
    ActionEstimate,
    FailedCondition,
    QuarticParameters,
    SSMParameters,
    estimate_quartic_action,
    estimate_ssm_action,
)

__all__ = [
    "ActionEstimate",
    "FailedCondition",
    "QuarticParameters",
    "SSMParameters",
    "estimate_quartic_action",
    "estimate_ssm_action",
]

__version__ = "0.1.0"

