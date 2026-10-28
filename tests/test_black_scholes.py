"""AT-01, AT-05, AT-11: Black–Scholes prices, analytical Greeks and edge cases."""

import math

import pytest
from scipy import integrate

from conftest import CASE_IDS, CASES
from derivatives_engine import OptionParams, black_scholes as bs, price_bounds

# High-precision reference values for S=K=100, r=5%, sigma=20%, T=1, q=0.
REF_CALL = 10.450583572185565
REF_PUT = 5.573526022256971
REF_CALL_DELTA = 0.6368306511756191
REF_PUT_DELTA = -0.3631693488243809
REF_GAMMA = 0.018762017345846895
REF_VEGA = 37.52403469169379
REF_CALL_THETA = -6.414027546438197
REF_PUT_THETA = -1.657880423934626


def test_reference_prices(ref_call, ref_put):
    assert bs.price(ref_call) == pytest.approx(REF_CALL, abs=1e-12)
    assert bs.price(ref_put) == pytest.approx(REF_PUT, abs=1e-12)
    assert round(bs.price(ref_call), 4) == 10.4506
    assert round(bs.price(ref_put), 4) == 5.5735


def test_reference_greeks(ref_call, ref_put):
    gc, gp = bs.greeks(ref_call), bs.greeks(ref_put)
    assert gc.delta == pytest.approx(REF_CALL_DELTA, abs=1e-12)
    assert gp.delta == pytest.approx(REF_PUT_DELTA, abs=1e-12)
    for g in (gc, gp):
        assert g.gamma == pytest.approx(REF_GAMMA, abs=1e-12)
        assert g.vega == pytest.approx(REF_VEGA, abs=1e-10)
    assert gc.theta == pytest.approx(REF_CALL_THETA, abs=1e-10)
    assert gp.theta == pytest.approx(REF_PUT_THETA, abs=1e-10)
    assert round(gc.delta, 4) == 0.6368 and round(gc.gamma, 4) == 0.0188


def test_display_conventions(ref_call):
    g = bs.greeks(ref_call)
    assert g.vega_per_pct == pytest.approx(g.vega / 100)
    assert g.theta_per_day == pytest.approx(g.theta / 365)


def _quadrature_price(p: OptionParams) -> float:
    """Independent check: integrate the discounted payoff against the lognormal density."""
    mu = math.log(p.spot) + (p.rate - p.dividend_yield - 0.5 * p.volatility ** 2) * p.maturity
    s = p.volatility * math.sqrt(p.maturity)

    def integrand(z):
        st = math.exp(mu + s * z)
        payoff = max(st - p.strike, 0.0) if p.is_call else max(p.strike - st, 0.0)
        return payoff * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    z_k = (math.log(p.strike) - mu) / s
    lo, hi = (z_k, 12.0) if p.is_call else (-12.0, z_k)
    val, _ = integrate.quad(integrand, lo, hi, epsabs=1e-12, epsrel=1e-12, limit=200)
    return p.discount_factor * val


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_price_matches_quadrature(p):
    assert bs.price(p) == pytest.approx(_quadrature_price(p), abs=1e-8)


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_analytical_greeks_match_high_precision_differences(p):
    """AT-05: compare against Richardson-extrapolated central differences of the price."""
    g = bs.greeks(p)

    def deriv(field, h, order=1):
        x = getattr(p, field)
        f = lambda v: bs.price(p.replace(**{field: v}))  # noqa: E731

        def central(hh):
            if order == 1:
                return (f(x + hh) - f(x - hh)) / (2 * hh)
            return (f(x + hh) - 2 * f(x) + f(x - hh)) / hh ** 2

        return (4 * central(h / 2) - central(h)) / 3  # Richardson

    assert g.delta == pytest.approx(deriv("spot", 0.5), abs=1e-7)
    assert g.gamma == pytest.approx(deriv("spot", 1.0, 2), abs=1e-6)
    assert g.vega == pytest.approx(deriv("volatility", 1e-3), abs=1e-6)
    assert g.theta == pytest.approx(-deriv("maturity", 1e-3), abs=1e-6)


@pytest.mark.parametrize("p", CASES, ids=CASE_IDS)
def test_put_call_delta_relation(p):
    c = bs.greeks(p.replace(option_type="call"))
    q = bs.greeks(p.replace(option_type="put"))
    assert c.delta - q.delta == pytest.approx(p.dividend_discount, abs=1e-14)
    assert c.gamma == pytest.approx(q.gamma) and c.vega == pytest.approx(q.vega)


# ---------------------------------------------------------------- AT-11 edge cases
def test_zero_volatility_call_itm():
    p = OptionParams(100, 90, 0.05, 0.0, 1.0, 0.01, "call")
    expected = 100 * math.exp(-0.01) - 90 * math.exp(-0.05)
    assert bs.price(p) == pytest.approx(expected, abs=1e-14)
    g = bs.greeks(p)
    assert g.delta == pytest.approx(math.exp(-0.01))
    assert g.gamma == 0.0 and g.vega == 0.0
    assert g.theta == pytest.approx(0.01 * 100 * math.exp(-0.01) - 0.05 * 90 * math.exp(-0.05))
    assert g.notes


def test_zero_volatility_out_of_the_money_is_worthless():
    p = OptionParams(100, 120, 0.05, 0.0, 1.0, 0.0, "call")
    assert bs.price(p) == 0.0
    assert bs.greeks(p).delta == 0.0
    put = p.replace(option_type="put")
    assert bs.price(put) == pytest.approx(120 * math.exp(-0.05) - 100)


def test_zero_vol_is_limit_of_small_vol():
    p = OptionParams(100, 95, 0.03, 0.0, 1.0)
    assert bs.price(p) == pytest.approx(bs.price(p.replace(volatility=1e-6)), abs=1e-9)


@pytest.mark.parametrize("strike", [1e-3, 1.0, 1e3, 1e5])
def test_deep_moneyness_respects_bounds(strike):
    for kind in ("call", "put"):
        p = OptionParams(100, strike, 0.05, 0.2, 1.0, 0.0, kind)
        v = bs.price(p)
        lo, hi = price_bounds(p)
        assert lo - 1e-9 <= v <= hi + 1e-9
        assert v >= 0.0


def test_deep_otm_tail_accuracy():
    # Deep OTM put: tiny but strictly positive and finite.
    p = OptionParams(100, 20, 0.01, 0.2, 0.5, 0.0, "put")
    v = bs.price(p)
    assert 0.0 <= v < 1e-12
    assert math.isfinite(bs.greeks(p).delta)


def test_short_maturity_converges_to_intrinsic():
    p = OptionParams(100, 95, 0.05, 0.2, 1e-6)
    assert bs.price(p) == pytest.approx(5.0, abs=1e-3)
    assert bs.greeks(p).delta == pytest.approx(1.0, abs=1e-6)


def test_negative_rate_and_dividend():
    p = OptionParams(100, 100, -0.02, 0.2, 1.0, -0.01, "call")
    c, q = bs.call_put_prices(p)
    assert c - q == pytest.approx(p.discounted_spot - p.discounted_strike, abs=1e-12)
