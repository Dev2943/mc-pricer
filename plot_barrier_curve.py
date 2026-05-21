"""
Barrier Option Price Curve.

Shows how a down-and-out barrier call's price varies with the barrier level B.
As B rises toward the spot price, the barrier becomes more likely to be hit,
the option more likely to knock out, and the price drops sharply.

The curve has a characteristic shape: nearly flat (at vanilla) for B far below
spot, then collapsing rapidly as B approaches spot. The "in-play zone" where
the barrier matters is narrow — typically the range from ~85% to ~100% of spot
for a 1-year ATM option with moderate volatility.

Run with: python3 plot_barrier_curve.py
Produces: barrier_price_curve.png
"""

import numpy as np
import matplotlib.pyplot as plt

from black_scholes import BlackScholesInputs, OptionType, price as bs_price
from exotics import price_barrier_down_out_call


# Standard ATM 1-year call setup
inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
N_PATHS = 100_000
N_STEPS = 252  # daily monitoring
SEED = 42

# Range of barrier levels to evaluate (must all be < spot)
barriers = np.arange(50, 100, 2.5)  # 50, 52.5, 55, ..., 97.5

print(f"Computing barrier prices across {len(barriers)} levels...")
print(f"  Spot = ${inputs.S}, Strike = ${inputs.K}, T = {inputs.T}y, sigma = {inputs.sigma:.0%}")
print(f"  N_paths = {N_PATHS:,}, n_steps = {N_STEPS}")
print()

prices = []
ses = []

for B in barriers:
    result = price_barrier_down_out_call(
        inputs, barrier=float(B), n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED,
    )
    prices.append(result.price)
    ses.append(result.standard_error)
    pct_vanilla = result.price / bs_price(inputs, OptionType.CALL) * 100
    print(f"  B = {B:5.1f}:  price = {result.price:.4f}  (SE {result.standard_error:.4f})  — {pct_vanilla:5.1f}% of vanilla")

vanilla_price = bs_price(inputs, OptionType.CALL)

# ----- Plot -----
fig, ax = plt.subplots(figsize=(10, 6.5))

# Error band: price ± 1.96 * SE
prices = np.array(prices)
ses = np.array(ses)
ax.fill_between(barriers, prices - 1.96*ses, prices + 1.96*ses,
                alpha=0.2, color="#C44E52", label="95% confidence band")
ax.plot(barriers, prices, marker="o", markersize=7, linewidth=2,
        color="#C44E52", label="Barrier call price (MC)")

# Reference: vanilla price
ax.axhline(vanilla_price, color="#4C72B0", linestyle="--", linewidth=2,
           label=f"Vanilla European = ${vanilla_price:.2f}")

# Reference: spot price (where barrier would equal spot — option immediately worthless)
ax.axvline(inputs.S, color="gray", linestyle=":", alpha=0.6,
           label=f"Spot price = ${inputs.S}")

ax.set_xlabel("Barrier level B  ($)", fontsize=12)
ax.set_ylabel("Down-and-out call price  ($)", fontsize=12)
ax.set_title(
    "Down-and-Out Barrier Call Price vs. Barrier Level\n"
    "ATM 1y call, $S$=$K$=$100, $r$=5%, $\\sigma$=20%, daily monitoring",
    fontsize=13,
)
ax.legend(fontsize=10, loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Annotation explaining the shape
ax.text(
    0.97, 0.97,
    "The 'in-play zone' is narrow:\n"
    "below $\\sim$$80, the barrier rarely matters;\n"
    "between $\\sim$$80–$100, the price collapses;\n"
    "the option carries crash protection priced in.",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

plt.tight_layout()
plt.savefig("barrier_price_curve.png", dpi=150, bbox_inches="tight")
print()
print("Saved barrier_price_curve.png")

try:
    plt.show()
except Exception:
    pass
