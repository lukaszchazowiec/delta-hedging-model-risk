import numpy as np
import pandas as pd

from scipy import stats

from hedging_engine import run_delta_hedging

def summarize_pnl(pnl):
    pnl = np.asarray(pnl)
    var95 = np.percentile(pnl, 5)
    tail_losses = pnl[pnl <= var95]
    cvar95 = float(np.mean(tail_losses))

    return {
        "mean p&l":  float(np.mean(pnl)),
        "std of p&l": float(np.std(pnl)),
        "median of p&l": float(np.median(pnl)),
        "skewness": float(stats.skew(pnl)),
        "excess kurtosis": float(stats.kurtosis(pnl)),
        "VaR_95": float(var95),
        "CVaR_95": float(cvar95)
    }