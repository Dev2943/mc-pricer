# Project 2 — Day 1: Monte Carlo Pricing under GBM

The foundation. By the end of today, you'll have a Monte Carlo pricer that prices European options by simulating random stock paths and averaging the payoff — and you'll verify it gives the same answer as your Project 1 Black-Scholes pricer.

---

## Part 1 — Why simulate when we have a closed-form?

Black-Scholes works only for vanilla European options on lognormal underlyings with no dividends and constant volatility. The moment any of those assumptions breaks, there's no closed-form formula:

- **Asian options** pay off based on the average stock price over the option's life — depends on the *path*, not just the endpoint. No closed form.
- **Barrier options** activate or deactivate when the stock hits a level. Path-dependent. No closed form.
- **Lookback options** pay off based on the maximum or minimum stock price reached. Path-dependent. No closed form.
- **Multi-asset options** (e.g., on a basket of 5 stocks) require multi-dimensional integration. Closed form usually impossible.
- **Heston model** (stochastic vol): closed form exists but is complicated and slow.
- **American options:** no closed form even in BS world.

**Monte Carlo handles all of these the same way:** simulate many random paths under the risk-neutral measure, compute the payoff on each path, average, discount. The technique scales to anything.

The price for this generality: instead of one formula evaluation, you do millions of random simulations. CPU-cheap by 2026 standards, but it's why modern quant codebases run on GPUs or massive CPU clusters.

---

## Part 2 — Risk-neutral pricing in one paragraph

Here's the entire conceptual core of modern derivatives pricing, in one paragraph. Read this twice.

> **Under the risk-neutral measure Q, the price today of any derivative is the discounted expected payoff at maturity:**
> 
> $$V_0 = e^{-rT} \cdot \mathbb{E}^{\mathbb{Q}}[\text{payoff at time } T]$$

That's it. That's the whole game.

The risk-neutral measure is a mathematical re-weighting of probabilities under which all assets earn the risk-free rate on average. It's not the "real-world" probability of stock moves — it's a fictitious probability that makes the math of pricing work without arbitrage. You don't need the full proof for now; you need to know **three facts:**

1. The formula `V = e^(-rT) · E[payoff]` is exact for any derivative, as long as the expectation is taken under Q.
2. Under Q, a non-dividend stock follows the SDE `dS = rS dt + σS dW` — same form as the real-world SDE but with `r` replacing the real expected return `μ`.
3. We can simulate from Q by drawing from the appropriate distribution, computing payoffs, averaging, and discounting.

**This is the bridge from theory to code.** The Monte Carlo pricer is literally a direct implementation of the above equation: simulate many `S_T` values from the right distribution, compute payoff on each, average, discount.

---

## Part 3 — Simulating Geometric Brownian Motion

Under the risk-neutral measure, the stock price follows:

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

This is a stochastic differential equation. Solving it (using Ito's lemma):

$$S_T = S_0 \exp\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma \sqrt{T} \cdot Z\right]$$

where `Z ~ N(0, 1)` is a standard normal random variable.

**This formula is the engine of your simulator.** To generate one random terminal stock price, you draw one Z from a standard normal, plug it in, get S_T. To generate N paths' terminal prices, draw N independent Z's.

For European options (payoff depends only on `S_T`), you don't need to simulate the whole path — just `S_T` directly via this formula. **You're effectively sampling from the lognormal distribution of S_T.**

For path-dependent options (Asian, barrier, lookback — Day 3), you'll need to simulate the full path step by step. We'll get there.

### Why the `(r - σ²/2)` correction term?

A common confusion: why isn't it just `r·T` in the exponent? Why subtract `σ²/2`?

Because the log of a stock price is normally distributed (`ln S_T ~ Normal`), but the stock price itself is lognormal (`S_T ~ Lognormal`). The mean of a lognormal isn't the exponential of the mean of its log — Jensen's inequality bites.

The math: if `ln(S_T/S_0) ~ N(μ_log·T, σ²·T)`, then `E[S_T] = S_0 · exp(μ_log·T + σ²·T/2)`. For risk-neutral pricing we want `E[S_T] = S_0 · exp(r·T)`. Solving, `μ_log = r - σ²/2`. That's the correction.

Interviewers love this question. The answer in one sentence: "It's the Itô correction — the difference between the drift of the log price and the drift of the price itself."

---

## Part 4 — The Monte Carlo estimator

The plan:
1. Generate `N` independent samples of `S_T` using the GBM formula.
2. For each sample, compute the option payoff: `max(S_T - K, 0)` for a call, `max(K - S_T, 0)` for a put.
3. Average the N payoffs to get an estimate of `E^Q[payoff]`.
4. Discount: multiply by `exp(-rT)`.

That gives you `V̂_N`, your Monte Carlo estimate of the option price. As `N → ∞`, `V̂_N → V` (the true price) by the Law of Large Numbers.

### Standard error — how accurate is this?

The Monte Carlo estimator has a known **standard error**:

$$\text{SE}(\hat{V}_N) = \frac{\sigma_{\text{payoff}}}{\sqrt{N}}$$

where `σ_payoff` is the sample standard deviation of the payoffs (estimated from your N samples).

**Two crucial implications:**

1. **MC converges at rate 1/√N.** To halve the standard error, you need 4× the samples. To get 10× more precision, you need 100× more samples. This is the curse of Monte Carlo — it converges slowly compared to deterministic methods.
2. **Variance reduction (Day 2) is about reducing `σ_payoff` without changing the estimator's mean.** A 50% variance reduction is equivalent to 4× more samples — free precision.

Real quant code always reports a 95% confidence interval: `V̂_N ± 1.96 · SE`. We'll do this too.

---

## Part 5 — Validation against Black-Scholes

Here's where Project 1 saves you: any vanilla European option you can price with Monte Carlo, you can also price exactly with your BS pricer. If your MC code is correct, the two prices must agree (within the MC standard error).

**This is the single most important test you'll write today.** Without it, you don't know if your code is right.

Concretely: run MC with N=100,000 paths on an ATM 1-year call. Compute the BS price. Check that `|V_MC - V_BS| < 3 · SE`. If yes, your code is correct. If no, there's a bug.

For the standard ATM case (S=100, K=100, T=1, r=5%, σ=20%):
- BS price: 10.4506
- MC price with N=100k should be 10.4506 ± ~0.04

When you see those two numbers agree, you've validated stochastic simulation against closed-form analytics. **That's a deep moment.** Two completely different mathematical machineries — calculus on PDEs vs. probability on simulated trajectories — agreeing because they're solutions to the same underlying problem.

---

## Part 6 — Implementation plan for today

You'll create one file: `monte_carlo.py`. It will have:

1. A function `simulate_terminal_prices(S0, r, sigma, T, n_paths, seed)` that returns an array of `n_paths` samples of `S_T` under the risk-neutral GBM dynamics.
2. A function `price_european_mc(S, K, T, r, sigma, option_type, n_paths, seed)` that uses #1 to compute the MC price + standard error + confidence interval.
3. A `__main__` block that:
   - Prices the standard ATM call with N=100k paths
   - Prints the MC price, its 95% CI, and the BS price for comparison
   - Confirms they agree

Plus a test file `test_monte_carlo.py` that automates the BS-vs-MC agreement test across multiple parameter sets.

Code complexity: small. The simulation is one line (`S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)` where Z is `np.random.standard_normal(n_paths)`). The hard work is conceptual — understanding *why* this line works.

---

## Part 7 — Interview questions to be ready for

After today you should be able to answer:

1. **What does "risk-neutral pricing" mean?** Pricing under a measure where all assets earn r, so we can discount expected payoffs at the risk-free rate.
2. **Why use Monte Carlo when Black-Scholes has a closed form?** For options where no closed form exists — exotics, path-dependent, multi-asset, American (with LSM).
3. **At what rate does MC converge?** 1/√N. To halve the error, you need 4× more samples.
4. **Where does the `(r - σ²/2)` correction in the GBM exponent come from?** Itô correction — the difference between the drift of log(S) and the drift of S, by Itô's lemma applied to the log function.
5. **What's the standard error of an MC estimator and how do you compute it?** `σ_payoff / √N`, where `σ_payoff` is the sample std dev of payoffs. Often reported as a 95% CI: `V̂ ± 1.96·SE`.
6. **Why do you discount the average payoff by `exp(-rT)`?** Because risk-neutral pricing gives present value as the discounted expectation of future cash flows: `V_0 = e^(-rT) · E^Q[payoff]`.
7. **How would you validate a new MC pricer?** Run it on a problem where the answer is known analytically (e.g., a European option you can also price with BS) and confirm the MC estimate is within ~3 standard errors of the analytical price.

That last one is the meta-skill — and you're about to live it.

---

## What to do now

1. Read this whole lesson. Take 30 minutes. The math is denser than Project 1.
2. Stop and answer the seven interview questions in your head (or write them out). Anything you can't answer is a gap to close before we code.
3. Reply "read" and I send the scaffolded `monte_carlo.py`.

This is the foundational day. The math we cover here unlocks everything else in Project 2 (variance reduction, exotics, convergence analysis). Don't skim.
