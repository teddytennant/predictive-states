"""Predictive-state generative sequence modeling, arXiv:2609.38524."""

import jax

jax.config.update("jax_enable_x64", True)

from predictive_states.conditional import GaussianConditional
from predictive_states.dgp import (
    DEFAULT_A_S,
    DEFAULT_A_X,
    TRUNCATION_BOUND,
    simulate_dgp1,
    truncated_normal_mean,
)
from predictive_states.model import PredictiveStateModel, algorithm1, max_abs_correlation
from predictive_states.rff import phi, sample_fourier
from predictive_states.ridge import apply_ridge, fit_ridge

__all__ = [
    "DEFAULT_A_S",
    "DEFAULT_A_X",
    "TRUNCATION_BOUND",
    "GaussianConditional",
    "PredictiveStateModel",
    "algorithm1",
    "apply_ridge",
    "fit_ridge",
    "max_abs_correlation",
    "phi",
    "sample_fourier",
    "simulate_dgp1",
    "truncated_normal_mean",
]
