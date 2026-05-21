"""
Exotic Options Pricing via Monte Carlo — Day 3.

Adds path-dependent option pricers to monte_carlo.py:
    - Arithmetic-average Asian call (no closed form under BS)
    - Down-and-out barrier call (closed form is fragile in practice)

Both require simulating the full price path, not just the terminal price.
Under GBM, the exact step-by-step solution is:

    S_{t+dt} = S_t * exp[(r - sigma^2/2)*dt + sigma*sqrt(dt)*Z],   Z ~ N(0,1)

We vectorize this across paths.

"""

from dataclasses import dataclass

import numpy as np

from black_scholes import BlackScholesInputs, OptionType, price as bs_price
from monte_carlo import MCResult, price_european_mc


# ----------------------------- PATH SIMULATION (provided) -----------------------------
def simulate_paths(
    inputs: BlackScholesInputs,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate `n_paths` full price paths over `n_steps` time steps.

    Each path is a sequence S_0, S_{t_1}, S_{t_2}, ..., S_T where t_i = i*T/n_steps.

    Returns
    -------
    np.ndarray
        Array of shape (n_steps + 1, n_paths). Row 0 is all S_0; row -1 is S_T;
        rows in between are the intermediate prices on each path.

    Implementation
    --------------
    Vectorized exact GBM stepping. log-returns are drawn jointly, then cumulatively
    summed along the time axis (because log-returns add over time), exponentiated,
    and multiplied by S_0.
    """
    rng = np.random.default_rng(seed)
    dt = inputs.T / n_steps

    # Step-by-step log returns: shape (n_steps, n_paths)
    Z = rng.standard_normal((n_steps, n_paths))
    log_returns = (
        (inputs.r - 0.5 * inputs.sigma**2) * dt
        + inputs.sigma * np.sqrt(dt) * Z
    )

    # Cumulative sum along time axis → cumulative log returns
    cumulative_log_returns = np.cumsum(log_returns, axis=0)

    # Prepend zeros for t=0 (log return of 0 → price ratio of 1)
    zeros_row = np.zeros((1, n_paths))
    cumulative_log_returns = np.vstack([zeros_row, cumulative_log_returns])

    # Convert back to prices: S(t) = S_0 * exp(cumulative log returns)
    paths = inputs.S * np.exp(cumulative_log_returns)

    return paths  # shape (n_steps + 1, n_paths)


# ----------------------------- ASIAN CALL -----------------------------
def price_asian_call(
    inputs: BlackScholesInputs,
    n_paths: int = 100_000,
    n_steps: int = 252,  # ~daily monitoring for a 1y option
    seed: int | None = None,
) -> MCResult:
    """Price an arithmetic-average Asian call by Monte Carlo.

    Payoff:
        max( mean(S_{t_1}, ..., S_{t_n}) - K, 0 )

    The monitoring dates are t_1, ..., t_n (excluding t=0 = today's known price).
    """
    # Simulate the full paths
    paths = simulate_paths(inputs, n_paths, n_steps, seed)

    # Average across monitoring dates, EXCLUDING t=0 (we exclude row 0).
    # paths[1:] has shape (n_steps, n_paths) — the prices at t_1, ..., t_n.
   
    avg_S = paths[1:].mean(axis=0)

    payoffs = np.maximum(avg_S - inputs.K, 0.0)

    # Discount and stats
    discount = np.exp(-inputs.r * inputs.T)
    discounted = discount * payoffs

    price = np.mean(discounted)
    standard_error = np.std(discounted, ddof=1) / np.sqrt(n_paths)
    ci_lower = price - 1.96 * standard_error
    ci_upper = price + 1.96 * standard_error

    return MCResult(
        price=price,
        standard_error=standard_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_paths=n_paths,
        method=f"asian (n_steps={n_steps})",
    )


# ----------------------------- DOWN-AND-OUT BARRIER CALL -----------------------------
def price_barrier_down_out_call(
    inputs: BlackScholesInputs,
    barrier: float,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> MCResult:
    """Price a down-and-out barrier call by Monte Carlo.

    Payoff:
        max(S_T - K, 0)  if min_{t in [0,T]} S_t > B
        0                otherwise

    Where B is the barrier (typically set below S_0). If the stock ever
    drops to or below B, the option knocks out and pays nothing.
    """
    if barrier >= inputs.S:
        raise ValueError(
            f"Barrier {barrier} must be below spot {inputs.S} for a down-and-out call."
        )

    # Simulate paths
    paths = simulate_paths(inputs, n_paths, n_steps, seed)

    
    # paths has shape (n_steps+1, n_paths). We want one min per path.
    
    min_prices = paths.min(axis=0)

    #  boolean mask indicating which paths did NOT breach the barrier
    #  min_prices > barrier
    not_breached = min_prices > barrier

    # Terminal price for each path (last row of paths)
    S_T = paths[-1]

    # compute the payoff per path.
    # If not_breached: vanilla call payoff max(S_T - K, 0)
    # If breached: 0
    
    payoffs = np.where(
    not_breached,
    np.maximum(S_T - inputs.K, 0.0),
    0.0,
)

    # Discount and stats
    discount = np.exp(-inputs.r * inputs.T)
    discounted = discount * payoffs

    price = np.mean(discounted)
    standard_error = np.std(discounted, ddof=1) / np.sqrt(n_paths)
    ci_lower = price - 1.96 * standard_error
    ci_upper = price + 1.96 * standard_error

    return MCResult(
        price=price,
        standard_error=standard_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_paths=n_paths,
        method=f"barrier (B={barrier})",
    )


# ----------------------------- SANITY CHECKS -----------------------------
if __name__ == "__main__":
    # Common test inputs: ATM 1y call
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    N = 100_000
    SEED = 42

    print("EXOTIC OPTIONS PRICING via MONTE CARLO")
    print("=" * 70)

    # Reference: vanilla European call price
    bs = bs_price(inputs, OptionType.CALL)
    vanilla_mc = price_european_mc(inputs, OptionType.CALL, n_paths=N, seed=SEED)
    print(f"\nReference vanilla European call:")
    print(f"  Black-Scholes:  {bs:.4f}")
    print(f"  Vanilla MC:     {vanilla_mc.price:.4f}  (SE: {vanilla_mc.standard_error:.4f})")

    # --- Sanity check 1: Asian with n_steps=1 should match vanilla ---
    print("\n" + "-" * 70)
    print("Sanity check 1: Asian call with 1 monitoring date == vanilla European")
    print("-" * 70)
    asian_n1 = price_asian_call(inputs, n_paths=N, n_steps=1, seed=SEED)
    print(f"  Asian (n_steps=1):  {asian_n1.price:.4f}  (SE: {asian_n1.standard_error:.4f})")
    print(f"  Vanilla MC:         {vanilla_mc.price:.4f}  (SE: {vanilla_mc.standard_error:.4f})")
    diff = asian_n1.price - vanilla_mc.price
    print(f"  Difference:         {diff:+.4f}  (should be tiny)")

    # --- Sanity check 2: Daily-averaged Asian — should be lower than vanilla ---
    print("\n" + "-" * 70)
    print("Asian with daily averaging (n_steps=252)")
    print("-" * 70)
    asian_daily = price_asian_call(inputs, n_paths=N, n_steps=252, seed=SEED)
    print(f"  Asian (daily avg):  {asian_daily.price:.4f}  (SE: {asian_daily.standard_error:.4f})")
    print(f"  Vanilla European:   {bs:.4f}")
    print(f"  Difference:         {asian_daily.price - bs:+.4f}  (Asian < vanilla, as expected)")
    print("  Why: averaging dampens the upside — same downside (zero) but lower expected payoff.")

    # --- Sanity check 3: Barrier with very low B should match vanilla ---
    print("\n" + "-" * 70)
    print("Sanity check 2: Down-and-out call with B=1 (essentially never hits) == vanilla")
    print("-" * 70)
    barrier_far = price_barrier_down_out_call(inputs, barrier=1.0, n_paths=N, n_steps=252, seed=SEED)
    print(f"  Barrier (B=1):      {barrier_far.price:.4f}  (SE: {barrier_far.standard_error:.4f})")
    print(f"  Vanilla MC:         {vanilla_mc.price:.4f}  (SE: {vanilla_mc.standard_error:.4f})")
    print(f"  Difference:         {barrier_far.price - vanilla_mc.price:+.4f}  (should be tiny)")

    # --- Show the barrier discount: price drops as B rises ---
    print("\n" + "-" * 70)
    print("Barrier price decreases as B rises toward spot (B=100):")
    print("-" * 70)
    for B in [50, 70, 80, 85, 90, 95]:
        result = price_barrier_down_out_call(
            inputs, barrier=B, n_paths=N, n_steps=252, seed=SEED,
        )
        discount_pct = (1 - result.price / bs) * 100
        print(f"  B = {B:3d}:  price = {result.price:.4f}  "
              f"(SE {result.standard_error:.4f})  — {discount_pct:5.1f}% cheaper than vanilla")

    print()
    print("Why barrier calls get cheaper as B rises:")
    print("  Higher B → barrier more likely to be hit → more paths knock out → lower expected payoff.")
    print("  At B=95 (5% below spot), substantial chance of knock-out within 1 year.")
