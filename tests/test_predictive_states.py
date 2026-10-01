"""Checks for equation (10), DGP 1, and the ridge Algorithm 1 instantiation."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from predictive_states import (
    PredictiveStateModel,
    max_abs_correlation,
    phi,
    simulate_dgp1,
    truncated_normal_mean,
)
from predictive_states.model import valid_times

jax.config.update("jax_enable_x64", True)


def _hand_cosine(window, omega, alpha):
    """Equation (10) with an explicit sum, not an einsum."""
    values = []
    for i, phase in enumerate(alpha):
        total = phase
        for j, x_j in enumerate(window):
            for c, x_jc in enumerate(x_j):
                total += omega[i][j][c] * x_jc
        values.append(math.cos(total))
    return values


def test_equation_10_matches_hand_cosine():
    omega = [
        [[0.2, -0.4], [0.5, 0.1], [-0.3, 0.7]],
        [[1.0, 0.0], [-0.2, 0.3], [0.4, -0.5]],
    ]
    alpha = [0.3, -1.1]
    window = [[0.5, -1.0], [1.2, 0.3], [-0.4, 0.8]]
    expected = _hand_cosine(window, omega, alpha)

    got = phi(jnp.asarray(window), jnp.asarray(omega), jnp.asarray(alpha))
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0.0)

    flat = jnp.asarray(window).reshape(-1)
    got_flat = phi(flat, jnp.asarray(omega), jnp.asarray(alpha))
    np.testing.assert_allclose(got_flat, expected, atol=1e-12, rtol=0.0)

    batch = jnp.stack([jnp.asarray(window), jnp.asarray(window) + 0.25])
    got_batch = phi(batch, jnp.asarray(omega), jnp.asarray(alpha))
    hand_second = _hand_cosine((batch[1]).tolist(), omega, alpha)
    np.testing.assert_allclose(got_batch[0], expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(got_batch[1], hand_second, atol=1e-12, rtol=0.0)


def test_z_hat_correlates_with_latent_state():
    corrs = []
    for seed in range(4):
        key = jax.random.PRNGKey(seed)
        key_data, key_fit = jax.random.split(key)
        x, s = simulate_dgp1(key_data, n=2000)
        model = PredictiveStateModel().fit(x, key_fit)
        corr = float(max_abs_correlation(model.z_hat, s[model.times]))
        corrs.append(corr)
        print(f"corr seed {seed} {corr:.4f}")
    mean_corr = float(np.mean(corrs))
    print(f"mean_abs_corr {mean_corr:.4f}")
    assert min(corrs) > 0.5
    assert mean_corr > 0.5


def test_one_step_mse_beats_memoryless_baselines():
    rows = []
    for seed in range(3):
        key = jax.random.PRNGKey(100 + seed)
        key_train, key_test, key_fit = jax.random.split(key, 3)
        x_train, _ = simulate_dgp1(key_train, n=800)
        x_test, _ = simulate_dgp1(key_test, n=1500)
        model = PredictiveStateModel().fit(x_train, key_fit)
        mse_ps = float(model.one_step_mse(x_test))
        mse_u, mse_ar = _baselines(x_train, x_test, model.lag, model.m)
        rows.append((mse_ps, mse_u, mse_ar))
        print(f"mse seed {seed} ps {mse_ps:.4f} uncond {mse_u:.4f} ar1 {mse_ar:.4f}")
    mean_ps = float(np.mean([row[0] for row in rows]))
    mean_u = float(np.mean([row[1] for row in rows]))
    mean_ar = float(np.mean([row[2] for row in rows]))
    print(f"mean_mse ps {mean_ps:.4f} uncond {mean_u:.4f} ar1 {mean_ar:.4f}")
    assert all(row[0] < row[1] for row in rows)
    assert mean_ps < mean_u
    assert mean_ps <= mean_ar


def test_more_data_does_not_increase_one_step_mse():
    diffs = []
    for seed in range(3):
        key = jax.random.PRNGKey(200 + seed)
        key_data, key_test, key_fit = jax.random.split(key, 3)
        x_long, _ = simulate_dgp1(key_data, n=800)
        x_test, _ = simulate_dgp1(key_test, n=1500)
        mse_200 = float(PredictiveStateModel().fit(x_long[:200], key_fit).one_step_mse(x_test))
        mse_800 = float(PredictiveStateModel().fit(x_long, key_fit).one_step_mse(x_test))
        diffs.append(mse_800 - mse_200)
        print(f"mse_n seed {seed} n200 {mse_200:.4f} n800 {mse_800:.4f}")
    print(f"mean_diff_800_minus_200 {float(np.mean(diffs)):.4f}")
    assert all(diff <= 0.05 for diff in diffs)
    assert float(np.mean(diffs)) <= 0.05


def test_log_prob_true_mean_beats_far_point():
    key = jax.random.PRNGKey(7)
    key_data, key_fit, key_sample = jax.random.split(key, 3)
    x, s = simulate_dgp1(key_data, n=600)
    model = PredictiveStateModel().fit(x, key_fit)
    row = model.times.shape[0] // 2
    t = int(model.times[row])
    z = model.z_hat[row]
    true_mean = truncated_normal_mean(s[t])
    far = true_mean + 6.0
    lp_true = float(model.log_prob(true_mean, z))
    lp_far = float(model.log_prob(far, z))
    print(f"log_prob true_mean {lp_true:.4f} far {lp_far:.4f}")
    assert lp_true > lp_far
    draw = model.sample(key_sample, z)
    assert draw.shape == (x.shape[1],)
    assert jnp.all(jnp.isfinite(draw))


def _baselines(train_x, test_x, lag, m):
    times = valid_times(test_x.shape[0], lag, m)
    target = test_x[times + 1]
    mu = jnp.mean(train_x, axis=0)
    mse_u = float(jnp.mean((target - mu) ** 2))
    design = jnp.concatenate([train_x[:-1], jnp.ones((train_x.shape[0] - 1, 1))], axis=1)
    gram = design.T @ design + 1e-8 * jnp.eye(design.shape[1])
    coef = jnp.linalg.solve(gram, design.T @ train_x[1:])
    test_design = jnp.concatenate([test_x[times], jnp.ones((times.shape[0], 1))], axis=1)
    pred = test_design @ coef
    mse_ar = float(jnp.mean((pred - target) ** 2))
    return mse_u, mse_ar
