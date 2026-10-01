"""Random Fourier features from equation (10)."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def sample_fourier(key, num_features, window, dim):
    """Draw frequencies and phases for equation (10).

    Omega has shape (num_features, window, dim). Each Omega_{i,j} is N(0, I_d),
    the Gaussian spectral measure of the RBF kernel exp(-||u - v||^2 / 2).
    alpha_i is Uniform(-pi, pi), as written in equation (10), not Uniform(0, 2 pi).
    `window` is m + 1.
    """
    if num_features < 1 or window < 1 or dim < 1:
        raise ValueError("num_features, window, and dim must be positive")
    key_w, key_a = jax.random.split(key)
    omega = jax.random.normal(key_w, (num_features, window, dim))
    alpha = jax.random.uniform(key_a, (num_features,), minval=-jnp.pi, maxval=jnp.pi)
    return omega, alpha


def phi(x, omega, alpha):
    """Apply equation (10).

    phi_i(x) = cos(sum_{j=1}^{m+1} Omega_{i,j} · x_j + alpha_i).

    x is a window in R^{(m+1) x d}, a batch of those windows, or the same
    array flattened along the last axis in that order.
    """
    omega = jnp.asarray(omega)
    alpha = jnp.asarray(alpha)
    if omega.ndim != 3:
        raise ValueError("omega must have shape (num_features, window, dim)")
    num_features, window, dim = omega.shape
    if alpha.shape != (num_features,):
        raise ValueError("alpha must have shape (num_features,)")
    windows = _as_windows(x, window, dim)
    dots = jnp.einsum("...jd,kjd->...k", windows, omega)
    return jnp.cos(dots + alpha)


def _as_windows(x, window, dim):
    x = jnp.asarray(x)
    if x.ndim >= 2 and x.shape[-2:] == (window, dim):
        return x
    flat = window * dim
    if x.shape[-1] == flat:
        return x.reshape(x.shape[:-1] + (window, dim))
    raise ValueError(
        f"expected shape (..., {window}, {dim}) or flattened (..., {flat}), got {tuple(x.shape)}"
    )
