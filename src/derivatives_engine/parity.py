"""Put–call parity and no-arbitrage bound validation (spec 7.1, 7.2).

Deterministic methods (Black–Scholes, CRR) are checked with an absolute
floating-point tolerance. Monte Carlo estimates are checked statistically:
a residual is only flagged if it exceeds ``z`` standard errors (plus a tiny
absolute floor), so ordinary sampling noise is not reported as model failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import monte_carlo
from .pricers import call_put_prices
from .types import NumericalConfig, OptionParams, PricingMethod, parse_method
from .validation import require_finite

#: Default number of standard errors allowed for Monte Carlo checks.
DEFAULT_MC_Z = 4.0


def default_abs_tolerance(params: OptionParams) -> float:
    """Floating-point tolerance scaled to the size of the contract."""
    return 1e-9 * max(1.0, params.spot, params.strike)


@dataclass(frozen=True)
class ParityResult:
    method: PricingMethod
    call: float
    put: float
    residual: float
    tolerance: float
    passed: bool
    std_error: float | None = None
    z_score: float | None = None
    diagnostics: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method.value, "call": self.call, "put": self.put,
                "residual": self.residual, "tolerance": self.tolerance, "passed": self.passed,
                "std_error": self.std_error, "z_score": self.z_score,
                "diagnostics": self.diagnostics}


@dataclass(frozen=True)
class BoundsResult:
    price: float
    lower: float
    upper: float
    slack: float
    passed: bool
    violations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"price": self.price, "lower": self.lower, "upper": self.upper,
                "slack": self.slack, "passed": self.passed, "violations": list(self.violations)}


def parity_residual(params: OptionParams, call: float, put: float) -> float:
    """C - P - S0 e^{-qT} + K e^{-rT}."""
    return call - put - params.discounted_spot + params.discounted_strike


def validate_put_call_parity(
    params: OptionParams,
    method: PricingMethod | str = PricingMethod.BLACK_SCHOLES,
    config: NumericalConfig | None = None,
    tolerance: float | None = None,
    z: float = DEFAULT_MC_Z,
) -> ParityResult:
    """Price a call and a put on the same inputs and check parity.

    For Monte Carlo the call and put are estimated from the *same* simulated
    paths, so C - P is the estimate of the linear payoff S_T - K, and the
    residual's standard error is that payoff's standard error.
    """
    method = parse_method(method)
    config = config or NumericalConfig()
    abs_tol = default_abs_tolerance(params) if tolerance is None else require_finite(tolerance, "tolerance")
    forward = params.discounted_spot - params.discounted_strike

    if method is PricingMethod.MONTE_CARLO:
        est = monte_carlo.call_put_forward(params, config)
        call, put = est["call"].price, est["put"].price
        resid = parity_residual(params, call, put)
        se = est["forward"].std_error
        tol = z * se + abs_tol
        passed = abs(resid) <= tol
        z_score = resid / se if se > 0 else None
        diag = (f"MC residual {resid:+.3e} vs allowance {tol:.3e} ({z:g} SE + {abs_tol:.1e}); "
                f"C-P={call - put:.6f}, S0e^-qT-Ke^-rT={forward:.6f}")
        return ParityResult(method, call, put, resid, tol, passed, se, z_score,
                            diag if passed else "PARITY FAILURE: " + diag)

    call, put = call_put_prices(method, params, config)
    resid = parity_residual(params, call, put)
    passed = abs(resid) <= abs_tol
    diag = (f"residual {resid:+.3e} vs tolerance {abs_tol:.1e}; "
            f"C-P={call - put:.10f}, S0e^-qT-Ke^-rT={forward:.10f}")
    return ParityResult(method, call, put, resid, abs_tol, passed,
                        diagnostics=diag if passed else "PARITY FAILURE: " + diag)


def price_bounds(params: OptionParams) -> tuple[float, float]:
    """No-arbitrage (lower, upper) bounds for the option in ``params``."""
    s, k = params.discounted_spot, params.discounted_strike
    if params.is_call:
        return max(s - k, 0.0), s
    return max(k - s, 0.0), k


def validate_price_bounds(
    params: OptionParams,
    price: float,
    std_error: float | None = None,
    tolerance: float | None = None,
    z: float = DEFAULT_MC_Z,
) -> BoundsResult:
    """Check lower <= price <= upper, allowing ``z * std_error`` for MC prices."""
    price = require_finite(price, "price")
    lower, upper = price_bounds(params)
    abs_tol = default_abs_tolerance(params) if tolerance is None else require_finite(tolerance, "tolerance")
    slack = abs_tol + (z * std_error if std_error else 0.0)
    violations = []
    if price < lower - slack:
        violations.append(f"price {price:.10g} below lower bound {lower:.10g} (slack {slack:.2e})")
    if price > upper + slack:
        violations.append(f"price {price:.10g} above upper bound {upper:.10g} (slack {slack:.2e})")
    return BoundsResult(price, lower, upper, slack, not violations, tuple(violations))
