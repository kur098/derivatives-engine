"""Common pricer interface (architectural principle 2.1.1: model independence).

Every method is exposed as a ``Pricer``: a callable ``OptionParams -> float``
bound to a fixed numerical configuration. Greeks, validation and benchmarking
only depend on this interface.
"""

from __future__ import annotations

from typing import Callable

from . import binomial, black_scholes, monte_carlo
from .types import NumericalConfig, OptionParams, PricingMethod, parse_method

Pricer = Callable[[OptionParams], float]


def make_pricer(method: PricingMethod | str, config: NumericalConfig | None = None) -> Pricer:
    """Return a pricing function for ``method`` with ``config`` frozen in."""
    method = parse_method(method)
    config = config or NumericalConfig()
    if method is PricingMethod.BLACK_SCHOLES:
        return black_scholes.price
    if method is PricingMethod.BINOMIAL:
        steps = config.tree_steps
        return lambda params: binomial.price(params, steps)
    return lambda params: monte_carlo.price(params, config)


def call_put_prices(method: PricingMethod | str, params: OptionParams,
                    config: NumericalConfig | None = None) -> tuple[float, float]:
    """(call, put) for the contract in ``params`` using one method."""
    pricer = make_pricer(method, config)
    return (pricer(params.replace(option_type="call")),
            pricer(params.replace(option_type="put")))
