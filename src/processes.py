import numpy as np
import pandas as pd

def simulate_gbm(S0, mu, sigma, T, M, N, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.normal(loc=0, scale=1, size=(M, N+1))

    dt = T / N

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion

    cum_log_returns = np.cumsum(log_returns, axis=1)
    zeros = np.zeros((M, 1))

    cum_log_returns = np.hstack([zeros, cum_log_returns])
    S = S0 * np.exp(cum_log_returns)

    return S
