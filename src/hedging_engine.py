import numpy as np
import pandas as pd
import math

from black_scholes import bsm_price, bsm_delta
from processes import simulate_gbm


def run_delta_hedging(N, S, K, T, sigma, r):

    account = 0
    dt = T / N
    tau_vector = np.linspace(T, 1e-14, N+1)

    current_price = bsm_price(S[:, 0], K, T, sigma, r)
    current_delta = bsm_delta(S[:, 0], K, T, sigma, r)

    account = current_price - current_delta * S[:, 0]

    for i in range(1, N):
        # interest:
        account *= np.exp(r*dt)

        current_tau = tau_vector[i]

        new_delta = bsm_delta(S[:,i], K, current_tau, sigma, r)
        delta_change = new_delta - current_delta

        account -= delta_change * S[:,i]
        current_delta = new_delta

    account *= np.exp(r*dt)
    account += current_delta * S[:, -1]
    account -= np.maximum(S[:, -1] - K, 0.0)

    return account