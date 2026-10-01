"""DGP 1 from section 4, scaled to a small stationary system."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", True)

DEFAULT_A_S = 0.7
DEFAULT_A_X = 0.9
TRUNCATION_BOUND = 2.5


def simulate_dgp1(
    key,
    n,
    a_s=DEFAULT_A_S,
    a_x=DEFAULT_A_X,
    burn=400,
    truncated=True,
    low=-TRUNCATION_BOUND,
    high=TRUNCATION_BOUND,
):
    """Simulate DGP 1.

    S_t = tanh(A_s S_{t-1} + A_x X_t), applied coordinate-wise by tanh.
    If truncated is true (the default, matching section 4), X_{t+1} is drawn
    from Normal(S_t, I) truncated to [low, high]^d. If truncated is false,
    X_{t+1} is untruncated Normal(S_t, I).

    A_s must have spectral radius below 1. With the default scalar 0.7 that
    radius is 0.7, so the linear part is a contraction and the process has a
    unique stationary distribution. `burn` steps are discarded.

    Returns x of shape (n, d) and s of shape (n, r). They are aligned so s[t]
    is the state after observing x[t], and the next observation is drawn from
    the conditional law of X_{t+1} given s[t].
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if burn < 0:
        raise ValueError("burn must be nonnegative")
    a_s, a_x = _system_matrices(a_s, a_x)
    radius = _spectral_radius(a_s)
    if radius >= 1.0:
        raise ValueError(f"spectral radius of A_s must be below 1, got {radius}")
    r = a_s.shape[0]
    d = a_x.shape[1]
    key_init, key_steps = jax.random.split(key)
    x_cur = jax.random.normal(key_init, (d,))
    s_prev = jnp.zeros((r,))
    step_keys = jax.random.split(key_steps, burn + n)

    def step(carry, step_key):
        s_prev, x_cur = carry
        s = jnp.tanh(a_s @ s_prev + a_x @ x_cur)
        x_next = _sample_observation(step_key, s, truncated, low, high)
        return (s, x_next), (x_cur, s)

    _, (xs, ss) = jax.lax.scan(step, (s_prev, x_cur), step_keys)
    return xs[burn:], ss[burn:]


def truncated_normal_mean(loc, low=-TRUNCATION_BOUND, high=TRUNCATION_BOUND, scale=1.0):
    """Mean of a normal truncated to [low, high], coordinate-wise."""
    loc = jnp.asarray(loc)
    alpha = (low - loc) / scale
    beta = (high - loc) / scale
    denom = norm.cdf(beta) - norm.cdf(alpha)
    adjust = (norm.pdf(alpha) - norm.pdf(beta)) / denom
    return loc + scale * adjust


def _system_matrices(a_s, a_x):
    a_s = jnp.atleast_2d(jnp.asarray(a_s, dtype=jnp.float64))
    a_x = jnp.atleast_2d(jnp.asarray(a_x, dtype=jnp.float64))
    if a_s.shape[0] != a_s.shape[1]:
        raise ValueError("A_s must be square")
    if a_x.shape[0] != a_s.shape[0]:
        raise ValueError("A_x must have the same number of rows as A_s")
    return a_s, a_x


def _spectral_radius(a_s):
    eigvals = jnp.linalg.eigvals(a_s)
    return float(jnp.max(jnp.abs(eigvals)))


def _sample_observation(key, loc, truncated, low, high):
    if not truncated:
        return loc + jax.random.normal(key, shape=loc.shape)
    u = jax.random.uniform(key, shape=loc.shape, minval=1e-6, maxval=1.0 - 1e-6)
    lo = norm.cdf(low, loc=loc, scale=1.0)
    hi = norm.cdf(high, loc=loc, scale=1.0)
    quantile = jnp.clip(lo + u * (hi - lo), 1e-8, 1.0 - 1e-8)
    return norm.ppf(quantile, loc=loc, scale=1.0)
