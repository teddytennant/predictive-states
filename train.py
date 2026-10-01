"""Fit Algorithm 1 on DGP 1 and print correlation and one-step MSE."""

import jax
import jax.numpy as jnp

from predictive_states import PredictiveStateModel, max_abs_correlation, simulate_dgp1
from predictive_states.model import valid_times


def main():
    key = jax.random.PRNGKey(0)
    key_train, key_test, key_fit = jax.random.split(key, 3)
    x_train, s_train = simulate_dgp1(key_train, n=1200)
    x_test, _ = simulate_dgp1(key_test, n=1500)
    model = PredictiveStateModel().fit(x_train, key_fit)
    corr = max_abs_correlation(model.z_hat, s_train[model.times])
    mse_ps = model.one_step_mse(x_test)
    mse_u = _unconditional_mse(x_train, x_test, model.lag, model.m)
    print(f"corr_Z_hat_S {float(corr):.6f}")
    print(f"mse_predictive_state {float(mse_ps):.6f}")
    print(f"mse_unconditional {float(mse_u):.6f}")


def _unconditional_mse(train_x, test_x, lag, m):
    times = valid_times(test_x.shape[0], lag, m)
    mu = jnp.mean(train_x, axis=0)
    target = test_x[times + 1]
    return jnp.mean((target - mu) ** 2)


if __name__ == "__main__":
    main()
