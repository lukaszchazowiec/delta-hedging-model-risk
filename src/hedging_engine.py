import numpy as np

from black_scholes import bsm_price, bsm_delta


def run_delta_hedging(S, K, T, sigma, r, *, sigma_price=None,
                      rebalance_every=21, cost=0.0, option_type="call"):

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if S.ndim != 2 or S.shape[1] < 2:
        raise ValueError("S must have shape (M, N+1) with N >= 1")
    if not (isinstance(rebalance_every, (int, np.integer)) and rebalance_every >= 1):
        raise ValueError("rebalance_every must be a positive integer")
    if cost < 0:
        raise ValueError("cost must be non-negative")

    if sigma_price is None:
        sigma_price = sigma

    N = S.shape[1] - 1
    dt = T / N
    growth = np.exp(r * dt)

    tau = np.maximum(T - np.arange(N + 1) * dt, 1e-12)

    premium = bsm_price(S[:, 0], K, T, sigma_price, r, option_type=option_type)
    delta = bsm_delta(S[:, 0], K, T, sigma, r, option_type=option_type)

    trade = delta * S[:, 0]
    account = premium - trade - cost * np.abs(trade)

    for i in range(1, N):
        account *= growth

        if i % rebalance_every == 0:
            new_delta = bsm_delta(S[:, i], K, tau[i], sigma, r, option_type=option_type)
            trade = (new_delta - delta) * S[:, i]
            account -= trade + cost * np.abs(trade)
            delta = new_delta

    account *= growth
    final_trade = delta * S[:, -1]
    account += final_trade - cost * np.abs(final_trade)

    if option_type == "call":
        payoff = np.maximum(S[:, -1] - K, 0.0)
    else:
        payoff = np.maximum(K - S[:, -1], 0.0)

    return account - payoff