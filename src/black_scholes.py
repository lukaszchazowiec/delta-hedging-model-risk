import numpy as np
import math

from scipy.stats import norm

"""
Option price depends on:
S - current stock (or other underlying) price
K - strike price
T - time to expiry
sigma - volatility
r - interest rates
"""


def _d1_d2(S, K , T, sigma, r):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2


def bsm_price(S, K, T, sigma, r, option_type="call"):
    d1, d2 = _d1_d2(S, K, T, sigma, r)

    if option_type == "call":
        C = S * norm.cdf(d1, loc=0, scale=1) - K * math.e**(-r*T) * norm.cdf(d2, loc=0, scale=1)
    elif option_type == "put":
        C = K * math.e**(-r*T) * norm.cdf(-d2, loc=0, scale=1) - S * norm.cdf(-d1, loc=0, scale=1)

    return C


def bsm_delta(S, K, T, sigma, r, option_type="call"):
    d1, d2 = _d1_d2(S, K, T, sigma, r)

    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1

    return delta

