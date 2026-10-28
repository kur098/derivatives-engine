"""Derivatives Pricing and Greeks Engine.

European call/put pricing with Black–Scholes, a CRR binomial tree and Monte
Carlo simulation; analytical and finite-difference Greeks; put–call parity and
no-arbitrage validation; reproducible benchmarks.

Example::

    from derivatives_engine import OptionParams, price_option, calculate_greeks

    params = OptionParams(spot=100, strike=100, rate=0.05, volatility=0.2,
                          maturity=1.0, option_type="call")
    result = price_option(params, method="monte_carlo", n_paths=200_000, seed=42)
    greeks = calculate_greeks(params, method="black_scholes", greek_method="analytical")
"""

__version__ = "1.0.0"

from .types import (  # noqa: E402
    GREEK_UNITS,
    GreekMethod,
    Greeks,
    NumericalConfig,
    OptionParams,
    OptionType,
    PricingMethod,
    PricingResult,
)
from .validation import NumericalError, ValidationError  # noqa: E402
from .greeks import calculate_greeks  # noqa: E402
from .parity import (  # noqa: E402
    BoundsResult,
    ParityResult,
    price_bounds,
    validate_price_bounds,
    validate_put_call_parity,
)
from .api import compare_methods, price_option  # noqa: E402
from .benchmarking import run_benchmarks  # noqa: E402

__all__ = [
    "__version__",
    "OptionParams", "NumericalConfig", "OptionType", "PricingMethod", "GreekMethod",
    "Greeks", "PricingResult", "GREEK_UNITS",
    "ValidationError", "NumericalError",
    "price_option", "calculate_greeks", "compare_methods", "run_benchmarks",
    "validate_put_call_parity", "validate_price_bounds", "price_bounds",
    "ParityResult", "BoundsResult",
]
