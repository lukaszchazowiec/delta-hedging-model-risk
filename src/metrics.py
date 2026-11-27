import numpy as np

from scipy import stats


def summarize_pnl(pnl):
    pnl = np.asarray(pnl)

    var95 = np.percentile(pnl, 5)
    var99 = np.percentile(pnl, 1)
    cvar95 = np.mean(pnl[pnl <= var95])
    cvar99 = np.mean(pnl[pnl <= var99])

    return {
        "mean p&l": float(np.mean(pnl)),
        "std of p&l": float(np.std(pnl)),
        "median of p&l": float(np.median(pnl)),
        "skewness": float(stats.skew(pnl)),
        "excess kurtosis": float(stats.kurtosis(pnl)),
        "VaR_95": float(var95),
        "CVaR_95": float(cvar95),
        "VaR_99": float(var99),
        "CVaR_99": float(cvar99),
    }