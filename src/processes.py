import numpy as np


def simulate_gbm(S0,
                 mu,
                 sigma,
                 T,
                 M,
                 N,
                 seed=None
):

    rng = np.random.default_rng(seed)
    dt = T / N

    Z = rng.normal(loc=0, scale=1, size=(M, N))

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion

    cum_log_returns = np.cumsum(log_returns, axis=1)
    zeros = np.zeros((M, 1))

    cum_log_returns = np.hstack([zeros, cum_log_returns])
    S = S0 * np.exp(cum_log_returns)

    return S


def simulate_merton_jump_diffusion(
    S0,
    mu,
    sigma,
    lam,
    mu_j,
    delta,
    T,
    M,
    N,
    seed=None,
):
    rng = np.random.default_rng(seed)
    dt = T / N

    Z_diff = rng.normal(loc=0.0, scale=1.0, size=(M, N))
    Z_jump = rng.normal(loc=0.0, scale=1.0, size=(M, N))
    n_jumps = rng.poisson(lam * dt, size=(M, N))

    kappa = np.exp(mu_j + 0.5 * delta**2) - 1.0

    drift = (mu - lam * kappa - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z_diff
    jumps = n_jumps * mu_j + np.sqrt(n_jumps) * delta * Z_jump

    log_returns = drift + diffusion + jumps

    cum_log_returns = np.cumsum(log_returns, axis=1)
    zeros = np.zeros((M, 1))
    cum_log_returns = np.hstack([zeros, cum_log_returns])

    S = S0 * np.exp(cum_log_returns)

    return S


