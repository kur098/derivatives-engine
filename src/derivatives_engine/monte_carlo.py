"""Monte Carlo pricing under the risk-neutral measure (spec 5.3).

    S_T = S0 exp[(r - q - sigma^2/2) T + sigma sqrt(T) Z],   Z ~ N(0, 1)

Only the terminal price is needed for European payoffs, so one normal draw
per path is exact (no time-discretisation bias).

Features
--------
* ``numpy.random.default_rng(seed)``: identical seed + configuration gives
  identical results (AT-08). Draws are consumed sequentially, so the stream
  does not depend on ``batch_size``.
* Batching bounds memory; running moments are merged with Chan et al.'s
  pairwise update, which is numerically stable.
* Optional variance reduction (FR-12): antithetic variates and a control
  variate on the discounted terminal price, whose mean S0 e^{-qT} is known.
* Standard error and a normal-approximation confidence interval are reported
  separately from any error against Black–Scholes. The CI describes sampling
  uncertainty; it does not guarantee that a given run covers the true price.
* Because the draws do not depend on the option parameters, re-pricing with
  the same seed reuses the same random numbers, which gives common random
  numbers for finite-difference Greeks automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.special import ndtri

from .types import NumericalConfig, OptionParams

PayoffFn = Callable[[np.ndarray, float], np.ndarray]


def call_payoff(s_t: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(s_t - strike, 0.0)


def put_payoff(s_t: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(strike - s_t, 0.0)


def forward_payoff(s_t: np.ndarray, strike: float) -> np.ndarray:
    """Linear payoff S_T - K (call minus put), used for parity checks."""
    return s_t - strike


@dataclass(frozen=True)
class MCEstimate:
    """A Monte Carlo price estimate with its sampling uncertainty."""

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_paths: int
    independent_samples: int
    seed: int | None
    antithetic: bool
    control_variate: bool
    control_beta: float | None = None

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)


class _RunningMoments:
    """Streaming mean/variance/covariance of (X, C) using Chan's merge."""

    __slots__ = ("n", "mean_x", "mean_c", "m2x", "m2c", "cxc")

    def __init__(self) -> None:
        self.n = 0
        self.mean_x = self.mean_c = 0.0
        self.m2x = self.m2c = self.cxc = 0.0

    def update(self, x: np.ndarray, c: np.ndarray | None) -> None:
        nb = x.size
        mx = float(x.mean())
        dx_b = x - mx
        m2x_b = float(dx_b @ dx_b)
        if c is not None:
            mc = float(c.mean())
            dc_b = c - mc
            m2c_b = float(dc_b @ dc_b)
            cxc_b = float(dx_b @ dc_b)
        else:
            mc = m2c_b = cxc_b = 0.0
        if self.n == 0:
            self.n, self.mean_x, self.mean_c = nb, mx, mc
            self.m2x, self.m2c, self.cxc = m2x_b, m2c_b, cxc_b
            return
        n_new = self.n + nb
        w = self.n * nb / n_new
        delta_x, delta_c = mx - self.mean_x, mc - self.mean_c
        self.mean_x += delta_x * nb / n_new
        self.mean_c += delta_c * nb / n_new
        self.m2x += m2x_b + delta_x * delta_x * w
        self.m2c += m2c_b + delta_c * delta_c * w
        self.cxc += cxc_b + delta_x * delta_c * w
        self.n = n_new


def terminal_prices(params: OptionParams, z: np.ndarray) -> np.ndarray:
    """Exact GBM terminal prices for standard normal draws ``z``."""
    drift = (params.rate - params.dividend_yield - 0.5 * params.volatility ** 2) * params.maturity
    vol = params.volatility * math.sqrt(params.maturity)
    return params.spot * np.exp(drift + vol * z)


def simulate(
    params: OptionParams,
    config: NumericalConfig,
    payoffs: Mapping[str, PayoffFn],
) -> dict[str, MCEstimate]:
    """Estimate several payoffs from one shared set of simulated paths.

    Sharing the paths makes, e.g., the call-minus-put estimate far more stable
    than pricing the call and the put independently (spec 7.1).
    """
    rng = np.random.default_rng(config.seed)
    disc = params.discount_factor
    control_mean = params.discounted_spot
    use_cv = config.control_variate
    n_units = config.independent_samples
    units_per_batch = max(1, config.batch_size // 2) if config.antithetic else config.batch_size

    moments = {name: _RunningMoments() for name in payoffs}
    remaining = n_units
    while remaining > 0:
        b = min(units_per_batch, remaining)
        z = rng.standard_normal(b)
        s_t = terminal_prices(params, z)
        s_t_anti = terminal_prices(params, -z) if config.antithetic else None

        control = None
        if use_cv:
            control = disc * s_t if s_t_anti is None else 0.5 * disc * (s_t + s_t_anti)
        for name, fn in payoffs.items():
            x = disc * fn(s_t, params.strike)
            if s_t_anti is not None:
                x = 0.5 * (x + disc * fn(s_t_anti, params.strike))
            moments[name].update(x, control)
        remaining -= b

    z_crit = float(ndtri(0.5 + 0.5 * config.confidence_level))
    results: dict[str, MCEstimate] = {}
    for name, m in moments.items():
        beta = None
        if use_cv:
            beta = m.cxc / m.m2c if m.m2c > 0.0 else 0.0
            mean = m.mean_x - beta * (m.mean_c - control_mean)
            resid_ss = m.m2x - beta * m.cxc  # = m2x - cxc^2 / m2c
            var = max(resid_ss, 0.0) / (m.n - 2)  # one extra dof for the fitted beta
        else:
            mean = m.mean_x
            var = m.m2x / (m.n - 1) if m.n > 1 else math.nan
        se = math.sqrt(var / m.n) if math.isfinite(var) else math.nan
        results[name] = MCEstimate(
            price=mean, std_error=se,
            ci_low=mean - z_crit * se, ci_high=mean + z_crit * se,
            confidence_level=config.confidence_level,
            n_paths=config.n_paths, independent_samples=n_units, seed=config.seed,
            antithetic=config.antithetic, control_variate=use_cv, control_beta=beta,
        )
    return results


def estimate(params: OptionParams, config: NumericalConfig | None = None) -> MCEstimate:
    """Monte Carlo estimate (price, SE, CI) for the option in ``params``."""
    config = config or NumericalConfig()
    payoff = call_payoff if params.is_call else put_payoff
    return simulate(params, config, {"option": payoff})["option"]


def price(params: OptionParams, config: NumericalConfig | None = None) -> float:
    """Monte Carlo price only (common pricer interface)."""
    return estimate(params, config).price


def call_put_forward(params: OptionParams, config: NumericalConfig | None = None) -> dict[str, MCEstimate]:
    """Call, put and forward (S_T - K) estimates from the same simulated paths."""
    config = config or NumericalConfig()
    return simulate(params, config, {"call": call_payoff, "put": put_payoff,
                                     "forward": forward_payoff})
