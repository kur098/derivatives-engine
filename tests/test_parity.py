"""AT-02 and spec 7: put–call parity and no-arbitrage bounds."""

import pytest

from conftest import CASE_IDS, CASES
from derivatives_engine import (NumericalConfig, OptionParams, monte_carlo, price_bounds,
                                validate_price_bounds, validate_put_call_parity)
from derivatives_engine.parity import parity_residual


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("method", ["black_scholes", "binomial"])
def test_deterministic_parity(p, method):
    res = validate_put_call_parity(p, method, NumericalConfig(tree_steps=400))
    assert res.passed, res.diagnostics
    assert abs(res.residual) < 1e-9


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_monte_carlo_parity_is_statistical(p):
    res = validate_put_call_parity(p, "monte_carlo", NumericalConfig(n_paths=50_000, seed=3))
    assert res.passed, res.diagnostics
    assert res.std_error > 0 and res.z_score is not None


def test_monte_carlo_parity_uses_shared_paths(ref_call):
    cfg = NumericalConfig(n_paths=20_000, seed=11)
    est = monte_carlo.call_put_forward(ref_call, cfg)
    assert est["call"].price - est["put"].price == pytest.approx(est["forward"].price, abs=1e-12)


def test_parity_with_control_variate_is_exact(ref_call):
    res = validate_put_call_parity(
        ref_call, "monte_carlo", NumericalConfig(n_paths=20_000, seed=5, control_variate=True))
    assert abs(res.residual) < 1e-9 and res.passed


def test_parity_failure_is_reported(ref_call):
    res = validate_put_call_parity(ref_call, "monte_carlo",
                                   NumericalConfig(n_paths=10_000, seed=1), z=0.0, tolerance=1e-12)
    assert not res.passed
    assert res.diagnostics.startswith("PARITY FAILURE")


def test_residual_formula(ref_call):
    assert parity_residual(ref_call, 10.450583572185565, 5.573526022256971) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_bounds_formulae(p):
    lo, hi = price_bounds(p)
    s, k = p.discounted_spot, p.discounted_strike
    if p.is_call:
        assert (lo, hi) == (max(s - k, 0.0), s)
    else:
        assert (lo, hi) == (max(k - s, 0.0), k)


def test_bound_violations_detected():
    p = OptionParams(100, 90, 0.05, 0.2, 1.0)
    lo, hi = price_bounds(p)
    assert validate_price_bounds(p, (lo + hi) / 2).passed
    below = validate_price_bounds(p, lo - 0.01)
    assert not below.passed and "below lower bound" in below.violations[0]
    above = validate_price_bounds(p, hi + 0.01)
    assert not above.passed and "above upper bound" in above.violations[0]


def test_bounds_allow_monte_carlo_noise():
    p = OptionParams(100, 300, 0.05, 0.2, 1.0, 0.0, "put")
    lo, _ = price_bounds(p)
    # A slightly-low MC estimate is not a failure when within z standard errors.
    assert validate_price_bounds(p, lo - 0.02, std_error=0.01).passed
    assert not validate_price_bounds(p, lo - 0.02, std_error=0.001).passed
