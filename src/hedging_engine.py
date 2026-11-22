import numpy as np
import pandas as pd
import math

from black_scholes import bsm_price, bsm_delta
from processes import simulate_gbm


def run_delta_hedging(S, K, T, sigma, r, rebalance=21, cost=0.001, option_type="call"):

    N = S.shape[1] - 1
    account = 0
    transaction_cost = cost
    total_transaction_cost = 0
    rebalance_every = rebalance

    dt = T / N
    tau_vector = np.linspace(T, 1e-14, N+1)

    current_price = bsm_price(S[:, 0], K, T, sigma, r, option_type=option_type)
    current_delta = bsm_delta(S[:, 0], K, T, sigma, r, option_type=option_type)

    initial_trade = current_delta * S[:, 0]
    initial_fee = np.abs(initial_trade) * transaction_cost
    account = current_price - initial_trade - initial_fee

    for i in range(1, N):
        # interest:
        account *= np.exp(r*dt)

        current_tau = tau_vector[i]

        if i % rebalance_every == 0:

            new_delta = bsm_delta(S[:,i], K, current_tau, sigma, r)
            delta_change = new_delta - current_delta

            trade_value = delta_change * S[:, i]
            fee = np.abs(trade_value) * transaction_cost

            account -= trade_value + fee
            current_delta = new_delta

    account *= np.exp(r*dt)

    final_trade = current_delta * S[:, -1]
    final_fee = np.abs(final_trade) * transaction_cost
    account += final_trade - final_fee

    if option_type == "call":
        payoff = np.maximum(S[:, -1] - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S[:, -1], 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    account -= payoff

    return account