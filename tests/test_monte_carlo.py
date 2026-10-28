"""AT-04, AT-08, AT-11: Monte Carlo engine."""

import math

import numpy as np
import pytest

from conftest import CASE_IDS, CASES
from derivatives_engine import NumericalConfig, OptionParams, black_scholes as bs, monte_carlo as mc


def cfg(**kw):
    kw.setdefault("seed", 12345)
    return NumericalConfig(**kw)


def test_reproducible_with_same_seed(ref_call):
    a = mc.estimate(ref_call, cfg(n_paths=50_000))
    b = mc.estimate(ref_call, cfg(n_paths=50_000))
    assert a == b


def test_different_seeds_differ(ref_call):
    assert mc.price(ref_call, cfg(seed=1)) != mc.price(ref_call, cfg(seed=2))


def test_batching_does_not_change_the_stream(ref_call):
    full = mc.estimate(ref_call, cfg(n_paths=100_000))
    batched = mc.estimate(ref_call, cfg(n_paths=100_000, batch_size=7_777))
    assert batched.price == pytest.approx(full.price, rel=1e-13)
    assert batched.std_error == pytest.approx(full.std_error, rel=1e-10)


def test_matches_explicit_formula(ref_call):
    """Estimator equals a hand-written vectorised computation for the same draws."""
    z = np.random.default_rng(99).standard_normal(10_000)
    p = ref_call
    st = p.spot * np.exp((p.rate - 0.5 * p.volatility ** 2) * p.maturity
                         + p.volatility * math.sqrt(p.maturity) * z)
    x = math.exp(-p.rate * p.maturity) * np.maximum(st - p.strike, 0)
    est = mc.estimate(p, NumericalConfig(n_paths=10_000, seed=99))
    assert est.price == pytest.approx(x.mean(), rel=1e-12)
    assert est.std_error == pytest.approx(x.std(ddof=1) / math.sqrt(x.size), rel=1e-10)


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("variance_reduction", [False, True])
def test_within_sampling_error_of_black_scholes(p, variance_reduction):
    c = cfg(n_paths=200_000, antithetic=variance_reduction, control_variate=variance_reduction)
    est = mc.estimate(p, c)
    assert abs(est.price - bs.price(p)) < 4.5 * est.std_error


def test_confidence_interval(ref_call):
    est = mc.estimate(ref_call, cfg(n_paths=100_000))
    lo, hi = est.confidence_interval
    assert lo < est.price < hi
    assert (hi - lo) / 2 == pytest.approx(1.959963984540054 * est.std_error)


def test_ci_coverage_is_close_to_nominal(ref_call):
    """Over many seeds ~95% of intervals cover the BS price (statistical, loose bound)."""
    ref = bs.price(ref_call)
    estimates = (mc.estimate(ref_call, NumericalConfig(n_paths=5_000, seed=s)) for s in range(200))
    hits = sum(e.ci_low <= ref <= e.ci_high for e in estimates)
    assert 0.88 <= hits / 200 <= 0.995


def test_standard_error_scales_as_inverse_sqrt(ref_call):
    """AT-04: SE ~ 1/sqrt(M) and error generally decreases with paths."""
    ses = [mc.estimate(ref_call, cfg(n_paths=m)).std_error for m in (10_000, 40_000, 160_000)]
    assert ses[0] / ses[1] == pytest.approx(2.0, rel=0.1)
    assert ses[1] / ses[2] == pytest.approx(2.0, rel=0.1)
    ref = bs.price(ref_call)
    mean_err = lambda m: np.mean([abs(mc.price(ref_call, NumericalConfig(n_paths=m, seed=s)) - ref)  # noqa: E731
                                  for s in range(20)])
    assert mean_err(100_000) < mean_err(1_000)


def test_variance_reduction_lowers_standard_error(ref_call):
    plain = mc.estimate(ref_call, cfg(n_paths=100_000)).std_error
    anti = mc.estimate(ref_call, cfg(n_paths=100_000, antithetic=True)).std_error
    both = mc.estimate(ref_call, cfg(n_paths=100_000, antithetic=True, control_variate=True))
    assert anti < plain
    assert both.std_error < anti
    assert both.control_beta is not None and both.independent_samples == 50_000


def test_control_variate_prices_forward_exactly(ref_call):
    est = mc.call_put_forward(ref_call, cfg(n_paths=10_000, control_variate=True))
    fwd = ref_call.discounted_spot - ref_call.discounted_strike
    assert est["forward"].price == pytest.approx(fwd, abs=1e-10)


def test_zero_volatility_is_deterministic():
    p = OptionParams(100, 90, 0.05, 0.0, 1.0)
    est = mc.estimate(p, cfg(n_paths=1_000))
    assert est.price == pytest.approx(bs.price(p), abs=1e-12)
    assert est.std_error == pytest.approx(0.0, abs=1e-12)


def test_single_path_reports_nan_standard_error(ref_call):
    est = mc.estimate(ref_call, cfg(n_paths=1))
    assert math.isfinite(est.price) and math.isnan(est.std_error)


def test_deep_otm_and_negative_rate():
    deep = OptionParams(100, 300, 0.02, 0.2, 0.5)
    assert mc.price(deep, cfg(n_paths=10_000)) == 0.0
    neg = OptionParams(100, 100, -0.01, 0.15, 2.0, 0.0, "put")
    est = mc.estimate(neg, cfg(n_paths=200_000))
    assert abs(est.price - bs.price(neg)) < 4.5 * est.std_error
