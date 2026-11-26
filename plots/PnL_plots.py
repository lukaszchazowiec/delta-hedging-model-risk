import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_pnl(
    data: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "P&L Distribution",
    bins: int = 50,
) -> plt.Axes:
    data = np.asarray(data, dtype=float)
    var95 = float(np.percentile(data, 5))
    tail = data[data <= var95]
    es95 = float(np.mean(tail)) if tail.size > 0 else var95

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
        created_fig = True

    counts, edges, patches = ax.hist(
        data,
        bins=bins,
        density=True,
        color="#4C72B0",
        alpha=0.35,
        edgecolor="white",
        linewidth=0.5,
        label="P&L histogram",
    )

    # Podświetlenie słupków w lewym ogonie (poniżej VaR 95%)
    for left_edge, patch in zip(edges[:-1], patches):
        if left_edge < var95:
            patch.set_facecolor("#DD8452")
            patch.set_alpha(0.75)

    # Gładka krzywa gęstości (KDE) dla lepszej czytelności kształtu rozkładu
    if np.std(data) > 0:
        x_grid = np.linspace(data.min(), data.max(), 300)
        kde = gaussian_kde(data)
        ax.plot(x_grid, kde(x_grid), color="#1F3A5F", lw=1.8, label="KDE")

    ax.axvline(var95, color="#C44E52", linestyle="--", lw=1.8, label=f"VaR 95%: {var95:.2f}")
    ax.axvline(es95, color="#E08214", linestyle="-.", lw=1.8, label=f"CVaR 95%: {es95:.2f}")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("P&L")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, frameon=True)

    if created_fig:
        ax.set_ylabel("Density")
        plt.tight_layout()
        plt.show()

    return ax


def plot_pnl_comparison(scenarios: dict[str, np.ndarray], bins: int = 50) -> plt.Figure:
    n_plots = len(scenarios)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.8), sharey=True)

    if n_plots == 1:
        axes = [axes]

    for ax, (name, pnl) in zip(axes, scenarios.items()):
        plot_pnl(pnl, ax=ax, title=name, bins=bins)

    axes[0].set_ylabel("Density")
    fig.suptitle("Delta-Hedging P&L: GBM vs. Merton Jump-Diffusion", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()
    return fig