"""AT-03, AT-11: CRR binomial tree."""

import math

import numpy as np
import pytest

from conftest import CASE_IDS, CASES
from derivatives_engine import OptionParams, ValidationError, binomial, black_scholes as bs


def _naive_crr(p: OptionParams, n: int) -> float:
    """Straightforward double-loop reference implementation."""
    dt = p.maturity / n
    u = math.exp(p.volatility * math.sqrt(dt))
    d = 1 / u
    prob = (math.exp((p.rate - p.dividend_yield) * dt) - d) / (u - d)
    disc = math.exp(-p.rate * dt)
    vals = []
    for j in range(n + 1):
        st = p.spot * u ** j * d ** (n - j)
        vals.append(max(st - p.strike, 0) if p.is_call else max(p.strike - st, 0))
    for i in range(n, 0, -1):
        vals = [disc * (prob * vals[j + 1] + (1 - prob) * vals[j]) for j in range(i)]
    return vals[0]


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("n", [1, 2, 7, 50])
def test_vectorised_matches_naive(p, n):
    if n < binomial.min_valid_steps(p):
        pytest.skip("tree invalid at this N")
    assert binomial.price(p, n) == pytest.approx(_naive_crr(p, n), rel=1e-12, abs=1e-12)


def test_one_step_tree_by_hand():
    p = OptionParams(100, 100, 0.05, 0.2, 1.0)
    u, d = math.exp(0.2), math.exp(-0.2)
    q = (math.exp(0.05) - d) / (u - d)
    assert binomial.price(p, 1) == pytest.approx(math.exp(-0.05) * q * (100 * u - 100))


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_converges_to_black_scholes(p):
    """AT-03: error shrinks roughly like 1/N over the defined scenario set."""
    ref = bs.price(p)
    errors = {n: abs(binomial.price(p, n) - ref) for n in (25, 100, 1000, 4000)}
    assert errors[1000] < errors[25]
    assert errors[4000] < 0.002 * max(1.0, ref) + 1e-3
    # Smoothed (average of N, N+1) error is close to monotone in N.
    smooth = [abs(0.5 * (binomial.price(p, n) + binomial.price(p, n + 1)) - ref)
              for n in (50, 200, 800)]
    assert smooth[2] < smooth[0]


def test_reference_case_accuracy(ref_call, ref_put):
    assert binomial.price(ref_call, 1000) == pytest.approx(10.4506, abs=5e-3)
    assert binomial.price(ref_put, 1000) == pytest.approx(5.5735, abs=5e-3)


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_put_call_parity_is_exact(p):
    c = binomial.price(p.replace(option_type="call"), 301)
    q = binomial.price(p.replace(option_type="put"), 301)
    assert c - q == pytest.approx(p.discounted_spot - p.discounted_strike, abs=1e-10)


def test_invalid_probability_rejected_with_min_steps():
    p = OptionParams(100, 100, 0.5, 0.01, 1.0)
    assert binomial.min_valid_steps(p) == 2500
    with pytest.raises(ValidationError, match="outside \\[0, 1\\].*at least 2500 steps"):
        binomial.price(p, 10)
    assert math.isfinite(binomial.price(p, 2500))


def test_crr_parameters_are_risk_neutral():
    p = OptionParams(100, 100, 0.03, 0.25, 2.0, 0.01)
    crr = binomial.crr_parameters(p, 200)
    assert 0 <= crr.p <= 1
    assert crr.u * crr.d == pytest.approx(1.0)
    growth = crr.p * crr.u + (1 - crr.p) * crr.d
    assert growth == pytest.approx(math.exp(0.02 * crr.dt))


def test_zero_volatility_handled_separately():
    p = OptionParams(100, 90, 0.05, 0.0, 1.0)
    assert binomial.price(p, 10) == pytest.approx(bs.price(p), abs=1e-14)


def test_invalid_steps():
    p = OptionParams(100, 100, 0.05, 0.2, 1.0)
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValidationError):
            binomial.price(p, bad)


def test_terminal_prices_symmetric():
    p = OptionParams(100, 100, 0.05, 0.2, 1.0)
    st = binomial.terminal_prices(p, 10)
    assert np.allclose(st * st[::-1], 100 ** 2)
    assert np.all(np.diff(st) > 0)


def test_negative_rate_and_short_maturity():
    p = OptionParams(100, 100, -0.01, 0.15, 2.0, 0.0, "put")
    assert binomial.price(p, 2000) == pytest.approx(bs.price(p), abs=5e-3)
    s = OptionParams(100, 102, 0.05, 0.2, 1 / 365)
    assert binomial.price(s, 2000) == pytest.approx(bs.price(s), abs=1e-3)
