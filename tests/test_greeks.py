"""AT-05, AT-06: analytical and finite-difference Greeks."""

import math

import pytest

from conftest import CASE_IDS, CASES
from derivatives_engine import (GreekMethod, NumericalConfig, OptionParams, black_scholes as bs,
                                calculate_greeks)

GREEKS = ("delta", "gamma", "vega", "theta")


def _close(est, ref, rel, abs_):
    return abs(est - ref) <= max(rel * abs(ref), abs_)


def test_default_greek_method_selection(ref_call):
    assert calculate_greeks(ref_call).method is GreekMethod.ANALYTICAL
    assert calculate_greeks(ref_call, "binomial").method is GreekMethod.FINITE_DIFFERENCE


def test_spec_example_usage(ref_call):
    g = calculate_greeks(ref_call, method="black_scholes", greek_method="analytical")
    assert (round(g.delta, 4), round(g.gamma, 4), round(g.vega, 3), round(g.theta, 3)) == \
        (0.6368, 0.0188, 37.524, -6.414)


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_fd_on_black_scholes_matches_analytical(p):
    a = bs.greeks(p)
    fd = calculate_greeks(p, "black_scholes", "finite_difference")
    for name in GREEKS:
        assert _close(getattr(fd, name), getattr(a, name), 1e-4, 1e-6), name
    assert fd.bumps["spot"] == pytest.approx(1e-3 * p.spot)
    assert set(fd.schemes.values()) == {"central"}


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_fd_on_binomial_matches_analytical(p):
    a = bs.greeks(p)
    g = calculate_greeks(p, "binomial", "fd", NumericalConfig(tree_steps=1000))
    for name in GREEKS:
        assert _close(getattr(g, name), getattr(a, name), 1.5e-2, 1e-4), name
    assert g.schemes["delta"] == "lattice_aligned_central"
    assert g.bumps["lattice_spacings"] >= 1


def test_binomial_spot_bumps_are_lattice_aligned(ref_call):
    """Naive small spot bumps give garbage Gamma on a tree; aligned bumps do not."""
    g = calculate_greeks(ref_call, "binomial", "fd", NumericalConfig(tree_steps=500),
                         bumps={"spot_rel": 0.001})
    assert g.gamma == pytest.approx(bs.greeks(ref_call).gamma, rel=5e-3)


def test_binomial_far_strike_uses_phase_aligned_vega():
    p = OptionParams(100, 120, 0.05, 0.2, 0.25)
    g = calculate_greeks(p, "binomial", "fd", NumericalConfig(tree_steps=1000))
    assert g.schemes["vega"] == g.schemes["theta"] == "strike_phase_aligned"
    a = bs.greeks(p)
    assert g.vega == pytest.approx(a.vega, rel=1e-2)
    assert g.theta == pytest.approx(a.theta, rel=1e-2)


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_fd_on_monte_carlo_with_common_random_numbers(p):
    a = bs.greeks(p)
    g = calculate_greeks(p, "monte_carlo", "fd",
                         NumericalConfig(n_paths=400_000, seed=2024, antithetic=True,
                                         control_variate=True))
    assert _close(g.delta, a.delta, 1.5e-2, 2e-3)
    assert _close(g.gamma, a.gamma, 8e-2, 1e-3)
    assert _close(g.vega, a.vega, 3e-2, 0.1)
    assert _close(g.theta, a.theta, 3e-2, 0.05)
    assert any("common random numbers" in n for n in g.notes)


def test_monte_carlo_greeks_without_seed_generate_one(ref_call):
    g = calculate_greeks(ref_call, "monte_carlo", "fd", NumericalConfig(n_paths=20_000))
    assert any("generated seed" in n for n in g.notes)


def test_crn_reduces_noise_versus_independent_draws(ref_call):
    """Delta from independent seeds is far noisier than with shared draws."""
    from derivatives_engine import monte_carlo as mc
    h = 2.0
    up, dn = ref_call.replace(spot=102.0), ref_call.replace(spot=98.0)
    crn = [(mc.price(up, NumericalConfig(n_paths=20_000, seed=s))
            - mc.price(dn, NumericalConfig(n_paths=20_000, seed=s))) / (2 * h) for s in range(10)]
    ind = [(mc.price(up, NumericalConfig(n_paths=20_000, seed=s))
            - mc.price(dn, NumericalConfig(n_paths=20_000, seed=s + 1000))) / (2 * h)
           for s in range(10)]
    spread = lambda xs: max(xs) - min(xs)  # noqa: E731
    assert spread(crn) * 5 < spread(ind)


def test_one_sided_vega_at_zero_volatility():
    p = OptionParams(100, 120, 0.05, 0.0, 1.0)
    g = calculate_greeks(p, "black_scholes", "fd")
    assert g.schemes["vega"] == "forward"
    assert any("one-sided" in n for n in g.notes)
    assert g.vega == pytest.approx(0.0, abs=1e-6)


def test_reduced_vega_bump_when_vol_below_bump():
    p = OptionParams(100, 100, 0.05, 5e-5, 1.0)
    g = calculate_greeks(p, "black_scholes", "fd")
    assert g.schemes["vega"] == "central" and g.bumps["vol"] == pytest.approx(5e-6)
    assert any("bump reduced" in n for n in g.notes)


def test_reduced_theta_bump_for_tiny_maturity():
    p = OptionParams(100, 100, 0.05, 0.2, 5e-5)
    g = calculate_greeks(p, "black_scholes", "fd")
    assert g.schemes["theta"] == "central" and g.bumps["time"] == pytest.approx(5e-6)
    assert g.theta == pytest.approx(bs.greeks(p).theta, rel=0.02)


def test_one_sided_spot_with_absolute_bump():
    p = OptionParams(1.0, 1.0, 0.0, 0.2, 1.0)
    g = calculate_greeks(p, "black_scholes", "fd", bumps={"spot_abs": 2.0})
    assert g.schemes["delta"] == "forward"


def test_bump_sizes_reported_and_configurable(ref_call):
    g = calculate_greeks(ref_call, "black_scholes", "fd", bumps={"spot_rel": 0.01, "vol": 0.002})
    assert g.bumps["spot"] == pytest.approx(1.0)
    assert g.bumps["vol"] == 0.002
    d = g.to_dict()
    assert d["units"]["theta"].startswith("per year")
    assert d["bumps"]["spot_rel"] == pytest.approx(0.01)


def test_zero_volatility_fd_greeks():
    p = OptionParams(100, 90, 0.05, 0.0, 1.0)
    g = calculate_greeks(p, "binomial", "fd", NumericalConfig(tree_steps=200))
    a = bs.greeks(p)
    assert g.delta == pytest.approx(a.delta, abs=1e-6)
    assert g.theta == pytest.approx(a.theta, rel=1e-3)
    # A vol bump from 0 makes an invalid CRR tree at N=200: documented NaN, not a crash.
    assert math.isnan(g.vega) and g.schemes["vega"] == "unavailable"
    assert any("Vega: not available" in n for n in g.notes)
