"""Multi-output ridge regression with an unpenalized intercept."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def fit_ridge(features, targets, penalty):
    """Minimize ||Y - X W - b||_F^2 + penalty * ||W||_F^2.

    features: (n, p). targets: (n, q) or (n,). Returns coef of shape (p + 1, q),
    last row the intercept. The penalty is strictly positive so the solve is
    not an unregularized normal equation.
    """
    if penalty <= 0:
        raise ValueError("ridge penalty must be positive")
    features = jnp.asarray(features)
    targets = jnp.asarray(targets)
    if targets.ndim == 1:
        targets = targets[:, None]
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features must be (n, p) and targets (n, q)")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must share the sample axis")
    n, p = features.shape
    design = jnp.concatenate([features, jnp.ones((n, 1), dtype=features.dtype)], axis=1)
    gram = design.T @ design
    pen = penalty * jnp.eye(p + 1, dtype=features.dtype)
    pen = pen.at[-1, -1].set(0.0)
    return jnp.linalg.solve(gram + pen, design.T @ targets)


def apply_ridge(features, coef):
    """Apply a coefficient matrix from fit_ridge."""
    features = jnp.asarray(features)
    coef = jnp.asarray(coef)
    squeeze = features.ndim == 1
    if squeeze:
        features = features[None, :]
    if features.ndim != 2:
        raise ValueError("features must have shape (n, p) or (p,)")
    design = jnp.concatenate(
        [features, jnp.ones((features.shape[0], 1), dtype=features.dtype)],
        axis=1,
    )
    out = design @ coef
    if squeeze:
        out = out[0]
    return out
