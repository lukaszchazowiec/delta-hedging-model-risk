import numpy as np
from black_scholes import bsm_price, bsm_delta


def run_delta_hedging(
    S,
    K,
    T,
    sigma,
    r,
    rebalance_every=21,
    cost=0.0,
    option_type="call",
):
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    N = S.shape[1] - 1
    dt = T / N
    tau_vector = np.linspace(T, 1e-14, N + 1)

    current_price = bsm_price(S[:, 0], K, T, sigma, r, option_type=option_type)
    current_delta = bsm_delta(S[:, 0], K, T, sigma, r, option_type=option_type)

    initial_trade = current_delta * S[:, 0]
    initial_fee = np.abs(initial_trade) * cost
    account = current_price - initial_trade - initial_fee

    for i in range(1, N):
        account *= np.exp(r * dt)

        if i % rebalance_every == 0:
            current_tau = tau_vector[i]
            new_delta = bsm_delta(
                S[:, i], K, current_tau, sigma, r, option_type=option_type
            )
            delta_change = new_delta - current_delta

            trade_value = delta_change * S[:, i]
            fee = np.abs(trade_value) * cost

            account -= trade_value + fee
            current_delta = new_delta

    account *= np.exp(r * dt)

    final_trade = current_delta * S[:, -1]
    final_fee = np.abs(final_trade) * cost
    account += final_trade - final_fee

    if option_type == "call":
        payoff = np.maximum(S[:, -1] - K, 0.0)
    else:
        payoff = np.maximum(K - S[:, -1], 0.0)

    account -= payoff
    return account