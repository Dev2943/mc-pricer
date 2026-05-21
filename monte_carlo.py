"""
Monte Carlo Options Pricer — Day 1: Vanilla European options.

Prices European calls and puts by simulating terminal stock prices under
the risk-neutral measure and averaging the discounted payoffs.

Risk-neutral pricing:
    V_0 = exp(-rT) * E^Q[payoff at T]

Under Q, the stock follows GBM:
    dS = rS dt + sigma S dW
    
The exact solution at time T (no discretization needed for European options):
    S_T = S_0 * exp[(r - sigma^2/2) * T + sigma * sqrt(T) * Z],   Z ~ N(0, 1)

Run `python3 monte_carlo.py` to verify your MC price agrees with the
analytical Black-Scholes price.
"""

from dataclasses import dataclass

import numpy as np

from black_scholes import BlackScholesInputs, OptionType, price as bs_price


@dataclass(frozen=True)
class MCResult:
    """Container for Monte Carlo pricing output.

    Reports the point estimate, standard error, and 95% confidence interval.
    """
    price: float          # the MC estimate of the option's value
    standard_error: float # std(discounted payoffs) / sqrt(N)
    ci_lower: float       # 95% confidence interval lower bound
    ci_upper: float       # 95% confidence interval upper bound
    n_paths: int          # number of simulated paths

    def __repr__(self) -> str:
        return (
            f"MCResult(price={self.price:.4f}, "
            f"SE={self.standard_error:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"N={self.n_paths})"
        )


# ----------------------------- PATH SIMULATION -----------------------------
def simulate_terminal_prices(
    inputs: BlackScholesInputs,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate `n_paths` samples of S_T under risk-neutral GBM.

    Parameters
    ----------
    inputs : BlackScholesInputs
        S, K (unused here but passed for consistency), T, r, sigma.
    n_paths : int
        Number of independent samples of S_T to generate.
    seed : int or None
        If provided, the RNG is seeded for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths,) with simulated terminal prices.

    Formula
    -------
    S_T = S_0 * exp[(r - sigma^2/2) * T + sigma * sqrt(T) * Z]
    """
    rng = np.random.default_rng(seed)

    # Draw n_paths independent standard normal random variables.
    # Each Z[i] will be plugged into the GBM formula to give one S_T sample.
    Z = rng.standard_normal(n_paths)

    exponent = (
    (inputs.r - 0.5 * inputs.sigma**2) * inputs.T
    + inputs.sigma * np.sqrt(inputs.T) * Z
    )


    # Apply S_0 * exp(...) to get terminal prices
    S_T = inputs.S * np.exp(exponent)

    return S_T


# ----------------------------- EUROPEAN OPTION PRICING -----------------------------
def price_european_mc(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Price a European option by Monte Carlo.

    Algorithm:
        1. Simulate n_paths samples of S_T
        2. Compute payoff at T on each path
        3. Discount each payoff to today
        4. Average the discounted payoffs → MC price estimate
        5. Compute standard error and 95% CI
    """
    # Step 1: simulate terminal prices
    S_T = simulate_terminal_prices(inputs, n_paths, seed)

    # Step 2: compute payoffs at maturity
    # Calls pay max(S_T - K, 0); puts pay max(K - S_T, 0).
    # We can vectorize using np.maximum (elementwise max, NOT np.max).
    if option_type == OptionType.CALL:
    
        payoffs = np.maximum(S_T - inputs.K, 0.0)
    elif option_type == OptionType.PUT:
        
        payoffs = np.maximum(inputs.K - S_T, 0.0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    # Step 3: discount payoffs to today.
    # Each payoff is in dollars at time T; we want dollars today.
    # Multiply by exp(-r * T).
    
    discounted_payoffs = (
    np.exp(-inputs.r * inputs.T) * payoffs
    )

    # Step 4: the MC price estimate is the average of the discounted payoffs.
   
    price = np.mean(discounted_payoffs)

    # Step 5: standard error.
    # SE = std(discounted_payoffs) / sqrt(N).
    # We use sample std (ddof=1) to get an unbiased estimator.
    
    standard_error = (
    np.std(discounted_payoffs, ddof=1)
    / np.sqrt(n_paths)
    )

    # 95% confidence interval: price ± 1.96 * SE
    # (1.96 is the 97.5th percentile of the standard normal)
    ci_lower = price - 1.96 * standard_error
    ci_upper = price + 1.96 * standard_error

    return MCResult(
        price=price,
        standard_error=standard_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_paths=n_paths,
    )


# ----------------------------- VALIDATION AGAINST BS -----------------------------
if __name__ == "__main__":
    # The truth-checker: MC price should agree with closed-form BS price
    # within a few standard errors. This is the moment Project 1 saves Project 2.

    print("Monte Carlo vs Black-Scholes validation")
    print("=" * 60)

    test_cases = [
        # (S, K, T, r, sigma, option_type, label)
        (100, 100, 1.0, 0.05, 0.20, OptionType.CALL, "ATM call"),
        (100, 100, 1.0, 0.05, 0.20, OptionType.PUT,  "ATM put"),
        (100,  80, 0.5, 0.03, 0.30, OptionType.CALL, "ITM call, short-dated"),
        (100, 120, 2.0, 0.05, 0.25, OptionType.PUT,  "OTM put, long-dated"),
    ]

    N = 100_000
    SEED = 42

    for S, K, T, r, sigma, opt_type, label in test_cases:
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)
        bs = bs_price(inputs, opt_type)
        mc = price_european_mc(inputs, opt_type, n_paths=N, seed=SEED)

        diff = mc.price - bs
        std_errors_away = diff / mc.standard_error if mc.standard_error > 0 else 0

        print(f"\n{label}: S={S}, K={K}, T={T}, sigma={sigma}, {opt_type.value}")
        print(f"  Black-Scholes:    {bs:.4f}")
        print(f"  Monte Carlo:      {mc.price:.4f}  (SE: {mc.standard_error:.4f})")
        print(f"  95% CI:           [{mc.ci_lower:.4f}, {mc.ci_upper:.4f}]")
        print(f"  Difference:       {diff:+.4f}  ({std_errors_away:+.2f} std errors)")

        if abs(std_errors_away) < 3:
            print(f"  ✓ Within 3 std errors of BS — MC code looks correct.")
        else:
            print(f"  ✗ MORE than 3 std errors away — there's a bug.")
