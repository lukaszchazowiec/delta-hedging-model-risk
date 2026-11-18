import numpy as np
import pandas as pd

from black_scholes import bsm_price, bsm_delta
from processes import simulate_gbm


def run_delta_hedging(prices_matrix, N, S, K, T, sigma, r):

    account = 0
    dt = T / N
    tau_vector = np.linspace(T, 1e-14, N+1)

    current_price = bsm_price(S[:, 0], K, T, sigma, r)
    current_delta = bsm_delta(S[:, 0], K, T, sigma, r)

    account = current_price - current_delta * S[:, 0]

    for i in range(2, N):
        current_tau = tau_vector[i]

        new_price = bsm_price(S[:,i], K, current_tau, sigma, r)
        new_delta = bsm_delta(S[:,i], K, current_tau, sigma, r)

        account += current_price - current_delta * S[:,i]

    pass