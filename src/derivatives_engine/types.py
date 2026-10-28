"""Core data types: option parameters, numerical configuration and results.

Conventions (documented in docs/mathematics.md):

* rates, dividend yields and volatilities are annualised decimals (0.05 = 5%),
* maturity is in years,
* Vega is per 1.00 change in decimal volatility (``vega_per_pct`` divides by 100),
* Theta is per year of calendar time, Theta = -dV/dT (``theta_per_day`` divides by 365),
* runtimes are recorded in milliseconds and labelled as such.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .validation import (
    ValidationError,
    parse_enum,
    require_finite,
    require_int,
    require_non_negative,
    require_positive,
    require_seed,
)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class PricingMethod(str, Enum):
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"


class GreekMethod(str, Enum):
    ANALYTICAL = "analytical"
    FINITE_DIFFERENCE = "finite_difference"


_METHOD_ALIASES = {"bs": "black_scholes", "blackscholes": "black_scholes",
                   "crr": "binomial", "tree": "binomial", "mc": "monte_carlo"}
_GREEK_ALIASES = {"fd": "finite_difference", "finite_differences": "finite_difference",
                  "numerical": "finite_difference", "closed_form": "analytical"}


def parse_option_type(value: object) -> OptionType:
    return parse_enum(value, OptionType, "option_type", {"c": "call", "p": "put"})


def parse_method(value: object) -> PricingMethod:
    return parse_enum(value, PricingMethod, "pricing method", _METHOD_ALIASES)


def parse_greek_method(value: object) -> GreekMethod:
    return parse_enum(value, GreekMethod, "greek_method", _GREEK_ALIASES)


def _json_float(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


@dataclass(frozen=True)
class OptionParams:
    """A European option contract plus the market inputs needed to price it."""

    spot: float
    strike: float
    rate: float
    volatility: float
    maturity: float
    dividend_yield: float = 0.0
    option_type: OptionType | str = OptionType.CALL

    def __post_init__(self) -> None:
        for name in ("spot", "strike", "rate", "volatility", "maturity", "dividend_yield"):
            object.__setattr__(self, name, require_finite(getattr(self, name), name))
        require_positive(self.spot, "spot")
        require_positive(self.strike, "strike")
        require_non_negative(self.volatility, "volatility")
        require_positive(self.maturity, "maturity")
        object.__setattr__(self, "option_type", parse_option_type(self.option_type))

    # -- convenience -----------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OptionParams":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValidationError(f"Unknown option parameter(s): {sorted(unknown)}.")
        return cls(**dict(data))

    def replace(self, **changes: Any) -> "OptionParams":
        """Return a validated copy with some fields changed."""
        return dataclasses.replace(self, **changes)

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    @property
    def discount_factor(self) -> float:
        return math.exp(-self.rate * self.maturity)

    @property
    def dividend_discount(self) -> float:
        return math.exp(-self.dividend_yield * self.maturity)

    @property
    def discounted_spot(self) -> float:
        """S0 * exp(-qT): present value of the asset delivered at maturity."""
        return self.spot * self.dividend_discount

    @property
    def discounted_strike(self) -> float:
        """K * exp(-rT)."""
        return self.strike * self.discount_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "spot": self.spot,
            "strike": self.strike,
            "rate": self.rate,
            "volatility": self.volatility,
            "maturity": self.maturity,
            "dividend_yield": self.dividend_yield,
            "option_type": self.option_type.value,
        }


@dataclass(frozen=True)
class NumericalConfig:
    """Numerical settings shared by the tree and Monte Carlo engines.

    Attributes:
        tree_steps: CRR tree steps N (binomial only).
        n_paths: number of simulated terminal prices M (Monte Carlo only).
            With ``antithetic=True`` this counts both members of each pair
            and must be even.
        seed: seed for ``numpy.random.default_rng``. ``None`` draws fresh
            entropy (not reproducible; a warning is attached to results).
        batch_size: maximum number of paths held in memory at once.
        antithetic: use antithetic variates (variance reduction, FR-12).
        control_variate: use the discounted terminal price, whose
            expectation S0*exp(-qT) is known, as a control variate (FR-12).
        confidence_level: nominal level of the normal-approximation CI.
    """

    tree_steps: int = 500
    n_paths: int = 100_000
    seed: int | None = None
    batch_size: int = 1_000_000
    antithetic: bool = False
    control_variate: bool = False
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_steps", require_int(self.tree_steps, "tree_steps", 1))
        object.__setattr__(self, "n_paths", require_int(self.n_paths, "n_paths", 1))
        object.__setattr__(self, "seed", require_seed(self.seed))
        object.__setattr__(self, "batch_size", require_int(self.batch_size, "batch_size", 1))
        for flag in ("antithetic", "control_variate"):
            if not isinstance(getattr(self, flag), bool):
                raise ValidationError(f"{flag} must be a bool.")
        level = require_finite(self.confidence_level, "confidence_level")
        if not 0.0 < level < 1.0:
            raise ValidationError(f"confidence_level must be in (0, 1), got {level}.")
        object.__setattr__(self, "confidence_level", level)
        if self.antithetic:
            if self.n_paths < 2 or self.n_paths % 2:
                raise ValidationError(
                    f"n_paths must be an even number >= 2 with antithetic variates, got {self.n_paths}."
                )
            if self.batch_size < 2:
                raise ValidationError("batch_size must be >= 2 with antithetic variates.")
        if self.control_variate and self.independent_samples < 3:
            raise ValidationError(
                "control_variate needs at least 3 independent samples to estimate its coefficient."
            )

    @property
    def independent_samples(self) -> int:
        """Number of i.i.d. estimator units (pairs when antithetic)."""
        return self.n_paths // 2 if self.antithetic else self.n_paths

    def replace(self, **changes: Any) -> "NumericalConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


GREEK_UNITS = {
    "delta": "per 1 unit of spot",
    "gamma": "per 1 unit of spot, squared",
    "vega": "per 1.00 change in decimal volatility",
    "theta": "per year of calendar time (-dV/dT)",
}


@dataclass(frozen=True)
class Greeks:
    """Delta, Gamma, Vega and Theta with provenance metadata."""

    delta: float
    gamma: float
    vega: float
    theta: float
    method: GreekMethod
    pricing_method: PricingMethod
    bumps: Mapping[str, float] = field(default_factory=dict)
    schemes: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def vega_per_pct(self) -> float:
        """Vega per one volatility percentage point (display convention)."""
        return self.vega / 100.0

    @property
    def theta_per_day(self) -> float:
        """Approximate daily Theta, theta / 365 (labelled approximation)."""
        return self.theta / 365.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.delta, self.gamma, self.vega, self.theta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": _json_float(self.delta),
            "gamma": _json_float(self.gamma),
            "vega": _json_float(self.vega),
            "theta": _json_float(self.theta),
            "vega_per_pct": _json_float(self.vega_per_pct),
            "theta_per_day_approx": _json_float(self.theta_per_day),
            "method": self.method.value,
            "pricing_method": self.pricing_method.value,
            "bumps": dict(self.bumps),
            "schemes": dict(self.schemes),
            "units": dict(GREEK_UNITS),
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        return (f"Greeks(delta={self.delta:.6f}, gamma={self.gamma:.6f}, "
                f"vega={self.vega:.6f}, theta={self.theta:.6f}, method={self.method.value})")


@dataclass(frozen=True)
class PricingResult:
    """Structured output of :func:`derivatives_engine.price_option` (spec 4.2)."""

    params: OptionParams
    method: PricingMethod
    price: float
    config: Mapping[str, Any]
    runtime: float
    total_runtime: float
    runtime_unit: str = "ms"
    greeks: Greeks | None = None
    benchmark_price: float | None = None
    abs_error: float | None = None
    rel_error: float | None = None
    std_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float | None = None
    validation_passed: bool | None = None
    warnings: tuple[str, ...] = ()
    backend: str = "python"

    @property
    def option_type(self) -> OptionType:
        return self.params.option_type  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        ci = None
        if self.confidence_interval is not None:
            ci = [_json_float(self.confidence_interval[0]), _json_float(self.confidence_interval[1])]
        return {
            "option_type": self.params.option_type.value,  # type: ignore[union-attr]
            "method": self.method.value,
            "backend": self.backend,
            "price": _json_float(self.price),
            "greeks": self.greeks.to_dict() if self.greeks else None,
            "params": self.params.to_dict(),
            "config": dict(self.config),
            "diagnostics": {
                f"runtime_{self.runtime_unit}": _json_float(self.runtime),
                f"total_runtime_{self.runtime_unit}": _json_float(self.total_runtime),
                "runtime_unit": self.runtime_unit,
                "benchmark_price": _json_float(self.benchmark_price),
                "benchmark_error": _json_float(self.abs_error),
                "relative_error": _json_float(self.rel_error),
                "std_error": _json_float(self.std_error),
                "confidence_interval": ci,
                "confidence_level": self.confidence_level,
                "validation_passed": self.validation_passed,
                "warnings": list(self.warnings),
            },
        }
