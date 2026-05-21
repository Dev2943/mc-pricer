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
    """Container for Monte Carlo pricing output."""
    price: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    method: str = "vanilla"  # "vanilla", "antithetic", "control_variate", "both"
 
    def __repr__(self) -> str:
        return (
            f"MCResult({self.method}, price={self.price:.4f}, "
            f"SE={self.standard_error:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"N={self.n_paths})"
        )
 
 
# ----------------------------- PATH SIMULATION -----------------------------
def simulate_terminal_prices(
    inputs: BlackScholesInputs,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Generate `n_paths` samples of S_T under risk-neutral GBM.
 
    If antithetic=True, draw n_paths//2 independent normals and append
    their negatives — yielding n_paths antithetic-paired samples.
 
    Formula:
        S_T = S_0 * exp[(r - sigma^2/2) * T + sigma * sqrt(T) * Z]
    """
    rng = np.random.default_rng(seed)
 
    if antithetic:
        
            Z_half = rng.standard_normal(n_paths // 2)
            Z = np.concatenate([Z_half, -Z_half])
    else:
        Z = rng.standard_normal(n_paths)
 
    exponent = (
        (inputs.r - 0.5 * inputs.sigma**2) * inputs.T
        + inputs.sigma * np.sqrt(inputs.T) * Z
    )
    S_T = inputs.S * np.exp(exponent)
 
    return S_T
 
 
# ----------------------------- EUROPEAN OPTION PRICING -----------------------------
def price_european_mc(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MCResult:
    """Price a European option by Monte Carlo, optionally with variance reduction.
 
    Algorithm with variance reduction:
        1. Simulate terminal prices (optionally antithetic)
        2. Compute payoffs and discount them
        3. If control_variate: adjust the estimator using S_T as the control
        4. Compute mean, std error, CI
    """
    # Step 1: simulate
    S_T = simulate_terminal_prices(inputs, n_paths, seed, antithetic)
    discount = np.exp(-inputs.r * inputs.T)
 
    # Step 2: payoffs at maturity
    if option_type == OptionType.CALL:
        payoffs = np.maximum(S_T - inputs.K, 0.0)
    elif option_type == OptionType.PUT:
        payoffs = np.maximum(inputs.K - S_T, 0.0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")
 
    # Discount payoffs to today (these are our "Y" samples)
    Y = discount * payoffs
 
    # Step 3: optional control variate adjustment.
    # Control X = discounted terminal stock price = e^(-rT) * S_T.
    # Its true mean under Q is S_0 (martingale property of discounted prices).
    if control_variate:
        # The "X" samples — same shape as Y
        X = discount * S_T
 
        # The known mean of X under Q
        mu_X = inputs.S
 
        #beta* = Cov(X, Y) / Var(X)
        beta = np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1)
 
        # form the controlled estimator
        # Y_cv = Y - beta * (X - mu_X)
        Y_cv = Y - beta * (X - mu_X)
 
        # Replace Y with the controlled samples for downstream stats
        Y = Y_cv
 
    # Step 4: point estimate and standard error
    # With antithetic sampling, samples come in N/2 negatively-correlated pairs.
    # The correct SE is computed from pair-averages, NOT from std(Y)/sqrt(N)
    # which incorrectly assumes independence between all N samples.
    if antithetic:
        n_pairs = n_paths // 2
        Y_pairs = 0.5 * (Y[:n_pairs] + Y[n_pairs:])
        price = np.mean(Y_pairs)
        standard_error = np.std(Y_pairs, ddof=1) / np.sqrt(n_pairs)
    else:
        price = np.mean(Y)
        standard_error = np.std(Y, ddof=1) / np.sqrt(n_paths)
    ci_lower = price - 1.96 * standard_error
    ci_upper = price + 1.96 * standard_error
 
    # Label for reporting
    if antithetic and control_variate:
        method = "antithetic + control"
    elif antithetic:
        method = "antithetic"
    elif control_variate:
        method = "control variate"
    else:
        method = "vanilla"
 
    return MCResult(
        price=price,
        standard_error=standard_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_paths=n_paths,
        method=method,
    )
 
 
# ----------------------------- DEMO: COMPARE METHODS -----------------------------
if __name__ == "__main__":
    print("Variance Reduction Comparison — ATM 1y call (S=K=100, r=5%, sigma=20%)")
    print("=" * 78)
 
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    bs = bs_price(inputs, OptionType.CALL)
    N = 100_000
    SEED = 42
 
    print(f"\nBlack-Scholes (truth):        {bs:.4f}\n")
 
    # Run all four methods with the same seed
    methods = [
        (False, False),
        (True,  False),
        (False, True),
        (True,  True),
    ]
 
    results = []
    baseline_se = None
 
    for antithetic, control in methods:
        result = price_european_mc(
            inputs, OptionType.CALL,
            n_paths=N, seed=SEED,
            antithetic=antithetic, control_variate=control,
        )
        results.append(result)
 
        if baseline_se is None:
            baseline_se = result.standard_error
            variance_ratio = 1.0
        else:
            variance_ratio = (result.standard_error / baseline_se) ** 2
 
        improvement = (1 - variance_ratio) * 100
 
        print(f"  {result.method:25s}  price={result.price:.4f}  SE={result.standard_error:.5f}  "
              f"variance vs vanilla: {variance_ratio*100:5.1f}%  ({improvement:+.1f}% reduction)")
 
    print()
    print("Interpretation:")
    print("  - Every method's price agrees with BS within a couple standard errors")
    print("  - Antithetic alone reduces variance by ~30-50%")
    print("  - Control variate alone reduces variance by ~95% (correlation between")
    print("    payoff and S_T is very high for an ATM call)")
    print("  - Combining the two stacks further, but with diminishing returns")
 