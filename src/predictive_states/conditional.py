"""Gaussian stand-in for the mixture density network in section 3.2.

Theorem 3 identifies Q_z = L(X_{t+1} | Z_t = z). This module fits that law as
a Gaussian whose mean is linear in z, with diagonal variance from residuals.
It is not the paper's mixture density network.
"""

import jax
import jax.numpy as jnp

from predictive_states.ridge import apply_ridge, fit_ridge

jax.config.update("jax_enable_x64", True)


class GaussianConditional:
    """N(W z + b, diag(v)), fit by ridge regression of x on z."""

    def __init__(self, coef, variance):
        self.coef = jnp.asarray(coef)
        self.variance = jnp.asarray(variance)

    @classmethod
    def fit(cls, z, x, penalty):
        z = jnp.asarray(z)
        x = jnp.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        coef = fit_ridge(z, x, penalty)
        fitted = apply_ridge(z, coef)
        variance = jnp.mean((x - fitted) ** 2, axis=0)
        variance = jnp.maximum(variance, 1e-6)
        return cls(coef, variance)

    def mean(self, z):
        z = jnp.asarray(z)
        single = z.ndim == 1
        if single:
            z = z[None, :]
        mu = apply_ridge(z, self.coef)
        return mu[0] if single else mu

    def log_prob(self, x, z):
        """Gaussian log density of x given z. x may be a single point or a batch."""
        mu = self.mean(z)
        x = jnp.asarray(x)
        if x.shape == mu.shape[:-1]:
            x = x[..., None]
        x = jnp.broadcast_to(x, mu.shape)
        var = self.variance
        quad = jnp.sum((x - mu) ** 2 / var, axis=-1)
        log_det = jnp.sum(jnp.log(2.0 * jnp.pi * var))
        return -0.5 * (quad + log_det)

    def sample(self, key, z):
        """Draw X ~ N(mean(z), diag(variance))."""
        mu = self.mean(z)
        noise = jax.random.normal(key, shape=mu.shape, dtype=mu.dtype)
        return mu + noise * jnp.sqrt(self.variance)
