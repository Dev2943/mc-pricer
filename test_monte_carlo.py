"""
Test suite for the Monte Carlo pricer.

Categories:
    1. CONVERGENCE TO BLACK-SCHOLES — vanilla MC must agree with BS to within
       a few standard errors across moneyness and maturities.

    2. VARIANCE REDUCTION — antithetic and control variates must reduce SE.

    3. EXOTIC LIMIT CASES — Asian at n=1 monitoring matches vanilla; barrier
       with B far below spot matches vanilla.

    4. STRUCTURAL PROPERTIES — barrier price decreases monotonically in B.

Run with: pytest test_monte_carlo.py -v
"""

import numpy as np
import pytest

from black_scholes import BlackScholesInputs, OptionType, price as bs_price
from monte_carlo import price_european_mc
from exotics import price_asian_call, price_barrier_down_out_call


# ========== 1. CONVERGENCE TO BLACK-SCHOLES ==========

BS_TEST_CASES = [
    (100, 100, 1.0,  0.05, 0.20, OptionType.CALL),
    (100, 100, 1.0,  0.05, 0.20, OptionType.PUT),
    (100,  80, 0.5,  0.03, 0.30, OptionType.CALL),
    (100, 120, 2.0,  0.05, 0.25, OptionType.PUT),
    (100, 100, 0.25, 0.05, 0.40, OptionType.CALL),
]


@pytest.mark.parametrize("S, K, T, r, sigma, opt_type", BS_TEST_CASES)
def test_mc_agrees_with_black_scholes(S, K, T, r, sigma, opt_type):
    """Vanilla MC must agree with closed-form BS within 4 standard errors."""
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)
    bs = bs_price(inputs, opt_type)
    mc = price_european_mc(inputs, opt_type, n_paths=100_000, seed=42)

    diff = mc.price - bs
    std_errors = abs(diff) / mc.standard_error
    assert std_errors < 4.0, (
        f"MC price {mc.price:.4f} disagrees with BS {bs:.4f} "
        f"by {std_errors:.2f} standard errors"
    )


# ========== 2. VARIANCE REDUCTION ==========

def test_antithetic_reduces_variance():
    """Antithetic should reduce variance for a vanilla call."""
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    vanilla = price_european_mc(inputs, OptionType.CALL, n_paths=100_000,
                                 seed=42, antithetic=False, control_variate=False)
    antithetic = price_european_mc(inputs, OptionType.CALL, n_paths=100_000,
                                    seed=42, antithetic=True, control_variate=False)

    # Variance reduction of at least 20% expected (typically ~50%)
    variance_ratio = (antithetic.standard_error / vanilla.standard_error) ** 2
    assert variance_ratio < 0.80, (
        f"Antithetic should reduce variance, but ratio is {variance_ratio:.2%}"
    )


def test_control_variate_reduces_variance_dramatically():
    """Control variate should reduce variance by >50% for an ATM call (typically ~85%)."""
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    vanilla = price_european_mc(inputs, OptionType.CALL, n_paths=100_000,
                                 seed=42, antithetic=False, control_variate=False)
    cv = price_european_mc(inputs, OptionType.CALL, n_paths=100_000,
                            seed=42, antithetic=False, control_variate=True)

    variance_ratio = (cv.standard_error / vanilla.standard_error) ** 2
    assert variance_ratio < 0.50, (
        f"Control variate should reduce variance >50%, but ratio is {variance_ratio:.2%}"
    )


def test_variance_reduction_does_not_change_estimator_mean():
    """Variance reduction techniques must produce unbiased estimators."""
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    bs = bs_price(inputs, OptionType.CALL)

    for antithetic in [False, True]:
        for control in [False, True]:
            result = price_european_mc(
                inputs, OptionType.CALL, n_paths=100_000, seed=42,
                antithetic=antithetic, control_variate=control,
            )
            assert abs(result.price - bs) < 0.1, (
                f"Method (antithetic={antithetic}, control={control}) "
                f"gave price {result.price:.4f}, expected ~{bs:.4f}"
            )


# ========== 3. EXOTIC LIMIT CASES ==========

def test_asian_with_one_monitoring_date_equals_vanilla():
    """Asian call with n_steps=1 reduces to a vanilla European."""
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    asian = price_asian_call(inputs, n_paths=100_000, n_steps=1, seed=42)
    vanilla = price_european_mc(inputs, OptionType.CALL, n_paths=100_000, seed=42)

    # Same seed + identical math → same answer
    assert abs(asian.price - vanilla.price) < 1e-6, (
        f"Asian at n_steps=1 ({asian.price}) should match vanilla ({vanilla.price})"
    )


def test_asian_daily_is_cheaper_than_vanilla():
    """Daily-monitored Asian is cheaper than vanilla (averaging dampens upside)."""
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    asian = price_asian_call(inputs, n_paths=100_000, n_steps=252, seed=42)
    bs = bs_price(inputs, OptionType.CALL)
    assert asian.price < bs, (
        f"Asian {asian.price:.4f} should be cheaper than vanilla {bs:.4f}"
    )


def test_barrier_far_below_spot_matches_vanilla():
    """Barrier with B near zero should never knock out, matching vanilla."""
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    barrier = price_barrier_down_out_call(
        inputs, barrier=1.0, n_paths=100_000, n_steps=252, seed=42,
    )
    bs = bs_price(inputs, OptionType.CALL)

    # Different RNG draw than vanilla MC, so allow a few standard errors of slack
    diff = abs(barrier.price - bs)
    assert diff < 4 * barrier.standard_error, (
        f"Barrier (B=1) {barrier.price:.4f} should match vanilla {bs:.4f}, "
        f"differs by {diff:.4f}"
    )


def test_barrier_rejects_invalid_inputs():
    """Barrier above spot should raise ValueError."""
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    with pytest.raises(ValueError):
        price_barrier_down_out_call(inputs, barrier=100.0, n_paths=1000, n_steps=10)
    with pytest.raises(ValueError):
        price_barrier_down_out_call(inputs, barrier=200.0, n_paths=1000, n_steps=10)


# ========== 4. STRUCTURAL PROPERTIES ==========

def test_barrier_price_decreases_as_B_rises():
    """As the barrier B rises toward spot, the option becomes cheaper."""
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    barriers = [50, 70, 80, 90, 95]
    prices = []
    for B in barriers:
        result = price_barrier_down_out_call(
            inputs, barrier=B, n_paths=100_000, n_steps=252, seed=42,
        )
        prices.append(result.price)

    # Prices must be monotonically non-increasing as B rises
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1] - 0.05, (
            f"Barrier price should decrease as B rises, but "
            f"price(B={barriers[i]})={prices[i]:.4f} < price(B={barriers[i+1]})={prices[i+1]:.4f}"
        )
