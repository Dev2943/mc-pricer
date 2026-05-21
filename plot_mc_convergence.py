"""
Convergence comparison: vanilla MC vs variance-reduction techniques.

Plots the standard error of each estimator as a function of the number of paths N,
on log-log axes. All four methods should follow a 1/sqrt(N) decay (parallel lines
with slope -1/2), but shifted vertically by the variance reduction factor.

The plot demonstrates two things at once:
    1. Every Monte Carlo estimator converges at the theoretical 1/sqrt(N) rate
    2. Variance reduction shifts the convergence curve down, equivalent to
       getting more samples "for free"

Run with: python3 plot_mc_convergence.py
Produces: mc_convergence.png
"""

import numpy as np
import matplotlib.pyplot as plt

from black_scholes import BlackScholesInputs, OptionType
from monte_carlo import price_european_mc


# Range of path counts to test
N_VALUES = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]

# How many independent trials per N to estimate the SE reliably
N_TRIALS = 30

# Standard test case
inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
option_type = OptionType.CALL


def measure_se(antithetic: bool, control_variate: bool):
    """For each N, run N_TRIALS independent simulations and record the empirical SE."""
    empirical_ses = []

    for N in N_VALUES:
        trial_prices = []
        for trial in range(N_TRIALS):
            # Different seed per trial to get independent samples
            result = price_european_mc(
                inputs, option_type,
                n_paths=N,
                seed=trial,
                antithetic=antithetic,
                control_variate=control_variate,
            )
            trial_prices.append(result.price)

        # Empirical SE = std of the prices we got across trials
        # This is a more honest measure than the reported SE because it sidesteps
        # any subtle dependence-structure issues in the within-trial SE formula.
        empirical_ses.append(np.std(trial_prices, ddof=1))

    return empirical_ses


if __name__ == "__main__":
    print("Computing convergence curves — this takes a minute...")
    print(f"  N values: {N_VALUES}")
    print(f"  Trials per N: {N_TRIALS}")
    print()

    methods = [
        ("Vanilla",                 False, False, "#4C72B0", "o"),
        ("Antithetic",              True,  False, "#55A868", "s"),
        ("Control variate",         False, True,  "#C44E52", "^"),
        ("Antithetic + Control",    True,  True,  "#8172B2", "D"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for label, antithetic, control, color, marker in methods:
        print(f"  Running {label}...")
        ses = measure_se(antithetic, control)
        ax.loglog(N_VALUES, ses, marker=marker, markersize=7, linewidth=2,
                  color=color, label=label)

    # Reference line: theoretical 1/sqrt(N) decay (slope -1/2 on log-log)
    # Anchor it to vanilla's SE at the smallest N for visual comparison
    N_arr = np.array(N_VALUES)
    ax.loglog(N_arr, 1.5 / np.sqrt(N_arr), '--', color='gray', alpha=0.5,
              label=r'$1/\sqrt{N}$ reference slope')

    ax.set_xlabel("Number of paths (N)", fontsize=12)
    ax.set_ylabel("Standard error of the price estimator", fontsize=12)
    ax.set_title(
        "Monte Carlo Convergence — Vanilla vs Variance Reduction\n"
        "ATM 1y call (S=K=100, r=5%, $\\sigma$=20%)",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    # Annotation
    ax.text(
        0.97, 0.97,
        "All four methods converge at the theoretical $1/\\sqrt{N}$ rate\n"
        "(parallel lines on log-log axes), but variance reduction shifts\n"
        "the curve down — equivalent to free additional samples.",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig("mc_convergence.png", dpi=150, bbox_inches="tight")
    print("\nSaved mc_convergence.png")

    try:
        plt.show()
    except Exception:
        pass
