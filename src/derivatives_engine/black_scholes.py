"""Black–Scholes(–Merton) analytical pricing and Greeks with continuous dividend yield.

This module is the reference implementation that the numerical methods are
validated against (spec 5.1, 6.2).

Edge cases:

* Zero (or numerically negligible) total volatility ``sigma*sqrt(T)`` is
  priced as the discounted deterministic payoff
  ``max(S0 e^{-qT} - K e^{-rT}, 0)`` for calls (mirror for puts), so no
  division by zero ever occurs.
* Prices are clamped at zero to remove round-off negatives in the far tails.
"""

from __future__ import annotations

import math

from scipy.special import ndtr

from .types import GreekMethod, Greeks, OptionParams, PricingMethod
from .validation import NumericalError

#: Total volatility sigma*sqrt(T) below which the model is treated as deterministic.
DETERMINISTIC_VOL_THRESHOLD = 1e-12

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF N(x) (SciPy ``ndtr``; accurate in the tails)."""
    return float(ndtr(x))


def norm_pdf(x: float) -> float:
    """Standard normal density n(x)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def is_deterministic(params: OptionParams) -> bool:
    return params.volatility * math.sqrt(params.maturity) < DETERMINISTIC_VOL_THRESHOLD


def d1_d2(params: OptionParams) -> tuple[float, float]:
    """Return (d1, d2). Raises NumericalError in the zero-volatility limit."""
    vol_sqrt_t = params.volatility * math.sqrt(params.maturity)
    if vol_sqrt_t < DETERMINISTIC_VOL_THRESHOLD:
        raise NumericalError("d1/d2 are undefined for zero volatility; use price() instead.")
    d1 = (math.log(params.spot / params.strike)
          + (params.rate - params.dividend_yield + 0.5 * params.volatility ** 2) * params.maturity
          ) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _indicator(x: float) -> float:
    """Limit of N(d) as sigma -> 0: 1 above, 0 below, 1/2 exactly at-the-forward."""
    return 1.0 if x > 0 else (0.0 if x < 0 else 0.5)


def call_put_prices(params: OptionParams) -> tuple[float, float]:
    """Return (call, put) Black–Scholes prices for the contract in ``params``."""
    s_disc, k_disc = params.discounted_spot, params.discounted_strike
    if is_deterministic(params):
        return max(s_disc - k_disc, 0.0), max(k_disc - s_disc, 0.0)
    d1, d2 = d1_d2(params)
    call = s_disc * norm_cdf(d1) - k_disc * norm_cdf(d2)
    put = k_disc * norm_cdf(-d2) - s_disc * norm_cdf(-d1)
    return max(call, 0.0), max(put, 0.0)


def price(params: OptionParams) -> float:
    """Black–Scholes price of the call or put described by ``params``."""
    call, put = call_put_prices(params)
    return call if params.is_call else put


def greeks(params: OptionParams) -> Greeks:
    """Analytical Delta, Gamma, Vega and Theta (Theta per year = -dV/dT)."""
    S, K, r, q, sigma, T = (params.spot, params.strike, params.rate,
                            params.dividend_yield, params.volatility, params.maturity)
    dq, df = params.dividend_discount, params.discount_factor
    notes: list[str] = []

    if is_deterministic(params):
        # sigma -> 0 limit: N(d1), N(d2) become indicators and n(d1) -> 0.
        moneyness = params.discounted_spot - params.discounted_strike
        n1 = n2 = _indicator(moneyness)
        m1 = m2 = 1.0 - n1
        pdf1 = 0.0
        gamma = 0.0
        vega = 0.0
        notes.append("zero-volatility limit: Gamma and Vega reported as 0"
                     + ("; at-the-forward Delta uses N=1/2" if moneyness == 0 else ""))
    else:
        d1, d2 = d1_d2(params)
        sqrt_t = math.sqrt(T)
        n1, n2, pdf1 = norm_cdf(d1), norm_cdf(d2), norm_pdf(d1)
        m1, m2 = norm_cdf(-d1), norm_cdf(-d2)  # N(-d) computed directly for tail accuracy
        gamma = dq * pdf1 / (S * sigma * sqrt_t)
        vega = S * dq * pdf1 * sqrt_t

    decay = -(S * dq * pdf1 * sigma) / (2.0 * math.sqrt(T)) if pdf1 else 0.0
    if params.is_call:
        delta = dq * n1
        theta = decay + q * S * dq * n1 - r * K * df * n2
    else:
        delta = -dq * m1
        theta = decay - q * S * dq * m1 + r * K * df * m2

    return Greeks(
        delta=delta, gamma=gamma, vega=vega, theta=theta,
        method=GreekMethod.ANALYTICAL, pricing_method=PricingMethod.BLACK_SCHOLES,
        schemes={g: "analytical" for g in ("delta", "gamma", "vega", "theta")},
        notes=tuple(notes),
    )
