"""Algorithm 1 with a linear ridge instantiation of both stages."""

import jax
import jax.numpy as jnp

from predictive_states.conditional import GaussianConditional
from predictive_states.rff import phi, sample_fourier
from predictive_states.ridge import apply_ridge, fit_ridge

jax.config.update("jax_enable_x64", True)


def valid_times(n, lag, m):
    """Times t with a lag window ending at t and a future window after t."""
    start = lag - 1
    stop = n - m - 1
    if stop <= start:
        raise ValueError(
            f"series length {n} is too short for lag={lag} and m={m}"
        )
    return jnp.arange(start, stop)


def lag_windows(x, times, lag):
    """Windows X_{t-lag+1:t} for each t. Shape (len(times), lag, d)."""
    offsets = jnp.arange(lag)
    index = times[:, None] - (lag - 1) + offsets[None, :]
    return x[index]


def future_windows(x, times, m):
    """Windows X_{t+1:t+m+1}. Shape (len(times), m + 1, d)."""
    offsets = jnp.arange(m + 1)
    index = times[:, None] + 1 + offsets[None, :]
    return x[index]


class PredictiveStateModel:
    """Two-stage estimator from Algorithm 1.

    Stage 1 regresses phi(X_{t+1:t+m+1}) on the flattened lag window of the
    past. The fitted values are Z_hat_t, the linear instantiation of equation
    (13). Stage 2 fits a Gaussian conditional law of X_{t+1} given Z_hat_t.
    """

    def __init__(
        self,
        lag=8,
        m=1,
        num_features=16,
        penalty=1.0,
        mean_penalty=1e-2,
        cross_fit=False,
        n_folds=5,
    ):
        if lag < 1:
            raise ValueError("lag must be at least 1")
        if m < 0:
            raise ValueError("m must be nonnegative")
        if num_features < 1:
            raise ValueError("num_features must be positive")
        self.lag = int(lag)
        self.m = int(m)
        self.num_features = int(num_features)
        self.penalty = float(penalty)
        self.mean_penalty = float(mean_penalty)
        self.cross_fit = bool(cross_fit)
        self.n_folds = int(n_folds)

    def fit(self, x, key):
        x = jnp.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError("x must have shape (n,) or (n, d)")
        n, d = x.shape
        times = valid_times(n, self.lag, self.m)
        omega, alpha = sample_fourier(key, self.num_features, self.m + 1, d)
        past = lag_windows(x, times, self.lag)
        future = future_windows(x, times, self.m)
        features = past.reshape(past.shape[0], -1)
        targets = phi(future, omega, alpha)
        state_coef = fit_ridge(features, targets, self.penalty)
        if self.cross_fit:
            z_hat = _cross_fit(features, targets, self.penalty, self.n_folds)
        else:
            z_hat = apply_ridge(features, state_coef)
        conditional = GaussianConditional.fit(z_hat, x[times + 1], self.mean_penalty)
        self.dim = d
        self.omega = omega
        self.alpha = alpha
        self.state_coef = state_coef
        self.z_hat = z_hat
        self.times = times
        self.conditional = conditional
        self._train_mean = jnp.mean(x, axis=0)
        return self

    def predictive_state(self, past):
        """Map a lag window, or a longer history, to Z_hat."""
        windows = _past_batch(past, self.lag, self.dim)
        features = windows.reshape(windows.shape[0], -1)
        z = apply_ridge(features, self.state_coef)
        return z

    def predict_next(self, past):
        """One-step predictive mean from the fitted maps. past -> Z_hat -> mean."""
        single = _is_single_past(past, self.dim)
        z = self.predictive_state(past)
        mu = self.conditional.mean(z)
        if single:
            return mu[0]
        return mu

    def log_prob(self, x, z):
        return self.conditional.log_prob(x, z)

    def sample(self, key, z):
        return self.conditional.sample(key, z)

    def one_step_mse(self, x):
        """Mean squared error of predict_next on a series, held out or not."""
        x = jnp.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        times = valid_times(x.shape[0], self.lag, self.m)
        past = lag_windows(x, times, self.lag)
        pred = self.predict_next(past)
        target = x[times + 1]
        return jnp.mean((pred - target) ** 2)


def algorithm1(x, key, **kwargs):
    """Algorithm 1 on one sequence, ridge instantiation.

    Build phi features of future windows, regress them on lag features of the
    past to get Z_hat, and fit the one-step conditional law of X_{t+1} given
    Z_hat. predict_next uses those fitted maps.
    """
    return PredictiveStateModel(**kwargs).fit(x, key)


def max_abs_correlation(z, s):
    """Largest absolute Pearson correlation between coordinates of z and s."""
    z = jnp.asarray(z)
    s = jnp.asarray(s).reshape(z.shape[0], -1)
    if z.shape[0] != s.shape[0]:
        raise ValueError("z and s must have the same number of rows")
    z_centered = z - jnp.mean(z, axis=0)
    s_centered = s - jnp.mean(s, axis=0)
    denom_z = jnp.sqrt(jnp.sum(z_centered ** 2, axis=0))
    denom_s = jnp.sqrt(jnp.sum(s_centered ** 2, axis=0))
    corr = (z_centered.T @ s_centered) / (denom_z[:, None] * denom_s[None, :])
    corr = jnp.where(jnp.isfinite(corr), corr, 0.0)
    return jnp.max(jnp.abs(corr))


def _cross_fit(features, targets, penalty, n_folds):
    n = features.shape[0]
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > n:
        raise ValueError("n_folds exceeds the number of regression rows")
    fold_id = [i * n_folds // n for i in range(n)]
    z_hat = jnp.zeros_like(targets)
    for fold in range(n_folds):
        test = [i for i, group in enumerate(fold_id) if group == fold]
        train = [i for i, group in enumerate(fold_id) if group != fold]
        test_idx = jnp.asarray(test)
        train_idx = jnp.asarray(train)
        coef = fit_ridge(features[train_idx], targets[train_idx], penalty)
        pred = apply_ridge(features[test_idx], coef)
        z_hat = z_hat.at[test_idx].set(pred)
    return z_hat


def _is_single_past(past, dim):
    past = jnp.asarray(past)
    if past.ndim == 1:
        return True
    if past.ndim == 2 and past.shape[-1] == dim:
        return True
    if past.ndim == 3:
        return False
    raise ValueError("past must have shape (lag,), (lag, d), or (batch, lag, d)")


def _past_batch(past, lag, dim):
    past = jnp.asarray(past)
    if past.ndim == 1:
        past = past[:, None]
    if past.ndim == 2:
        past = past[None, :, :]
    if past.ndim != 3 or past.shape[-1] != dim:
        raise ValueError(f"past must end with feature dimension {dim}")
    if past.shape[1] < lag:
        raise ValueError(f"past length {past.shape[1]} is shorter than lag {lag}")
    return past[:, -lag:, :]
