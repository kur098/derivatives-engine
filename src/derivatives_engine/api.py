"""Public pricing API (spec 12): ``price_option`` and ``compare_methods``."""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

from . import binomial, black_scholes, monte_carlo
from .greeks import calculate_greeks
from .parity import validate_price_bounds
from .types import (GreekMethod, NumericalConfig, OptionParams, PricingMethod, PricingResult,
                    parse_greek_method, parse_method)
from .validation import ValidationError, require_finite

#: Default denominator floor epsilon for relative errors (spec 7.3).
DEFAULT_REL_ERROR_FLOOR = 1e-8
_NEAR_ZERO_PRICE = 1e-6

ParamsLike = OptionParams | Mapping[str, Any]


def _coerce_params(params: ParamsLike) -> OptionParams:
    if isinstance(params, OptionParams):
        return params
    if isinstance(params, Mapping):
        return OptionParams.from_mapping(params)
    raise ValidationError(f"params must be OptionParams or a mapping, got {type(params).__name__}.")


def _build_config(config: NumericalConfig | None, overrides: Mapping[str, Any]) -> NumericalConfig:
    config = config or NumericalConfig()
    return config.replace(**overrides) if overrides else config


def _method_config(method: PricingMethod, config: NumericalConfig) -> dict[str, Any]:
    """Only the configuration fields that affect ``method``."""
    if method is PricingMethod.BINOMIAL:
        return {"tree_steps": config.tree_steps}
    if method is PricingMethod.MONTE_CARLO:
        return {k: v for k, v in config.to_dict().items() if k != "tree_steps"}
    return {}


def price_option(
    params: ParamsLike,
    method: PricingMethod | str = PricingMethod.BLACK_SCHOLES,
    config: NumericalConfig | None = None,
    *,
    greeks: bool | GreekMethod | str = False,
    bumps: Mapping[str, float] | None = None,
    benchmark: bool = True,
    validate: bool = True,
    rel_error_floor: float = DEFAULT_REL_ERROR_FLOOR,
    **config_overrides: Any,
) -> PricingResult:
    """Price a European option and return a structured :class:`PricingResult`.

    Args:
        params: :class:`OptionParams` or a mapping of its fields.
        method: ``"black_scholes"``, ``"binomial"`` or ``"monte_carlo"``.
        config: numerical configuration; keyword overrides such as
            ``tree_steps=1000``, ``n_paths=10**6``, ``seed=7`` are applied on top.
        greeks: ``False`` (default), ``True`` (method's default Greek method),
            ``"analytical"`` or ``"finite_difference"``.
        bumps: finite-difference bump overrides (see ``calculate_greeks``).
        benchmark: compute absolute/relative error against Black–Scholes.
        validate: check no-arbitrage bounds (statistical for Monte Carlo).
        rel_error_floor: epsilon in |V - V_BS| / max(|V_BS|, epsilon).

    ``runtime`` is the pricing-kernel time; ``total_runtime`` also includes
    Greeks, benchmarking and validation. Both are in ``runtime_unit`` (ms).
    """
    t_total = time.perf_counter_ns()
    params = _coerce_params(params)
    method = parse_method(method)
    config = _build_config(config, config_overrides)
    rel_error_floor = require_finite(rel_error_floor, "rel_error_floor")
    warnings: list[str] = []

    # -- pricing kernel ------------------------------------------------
    std_error = ci = level = None
    t0 = time.perf_counter_ns()
    if method is PricingMethod.BLACK_SCHOLES:
        value = black_scholes.price(params)
    elif method is PricingMethod.BINOMIAL:
        value = binomial.price(params, config.tree_steps)
    else:
        est = monte_carlo.estimate(params, config)
        value, std_error, ci, level = est.price, est.std_error, est.confidence_interval, est.confidence_level
    kernel_ms = (time.perf_counter_ns() - t0) / 1e6

    if black_scholes.is_deterministic(params):
        warnings.append("zero volatility: priced as discounted deterministic payoff")
    if method is PricingMethod.MONTE_CARLO:
        if config.seed is None:
            warnings.append("no random seed set: Monte Carlo result is not reproducible")
        if config.independent_samples < 1000:
            warnings.append("fewer than 1000 independent samples: normal-approximation CI is unreliable")

    # -- Greeks ----------------------------------------------------------
    greek_result = None
    if greeks is not False and greeks is not None:
        greek_method = None if greeks is True else parse_greek_method(greeks)
        greek_result = calculate_greeks(params, method, greek_method, config, bumps)

    # -- benchmark -------------------------------------------------------
    bs_price = abs_err = rel_err = None
    if benchmark and method is not PricingMethod.BLACK_SCHOLES:
        bs_price = black_scholes.price(params)
        abs_err = abs(value - bs_price)
        rel_err = abs_err / max(abs(bs_price), rel_error_floor)
        if abs(bs_price) < _NEAR_ZERO_PRICE:
            warnings.append("benchmark price is near zero: relative error is not meaningful")

    # -- validation ------------------------------------------------------
    passed = None
    if validate:
        bounds = validate_price_bounds(params, value, std_error=std_error)
        passed = bounds.passed
        warnings.extend(bounds.violations)

    total_ms = (time.perf_counter_ns() - t_total) / 1e6
    return PricingResult(
        params=params, method=method, price=value, config=_method_config(method, config),
        runtime=kernel_ms, total_runtime=total_ms, runtime_unit="ms",
        greeks=greek_result, benchmark_price=bs_price, abs_error=abs_err, rel_error=rel_err,
        std_error=std_error, confidence_interval=ci, confidence_level=level,
        validation_passed=passed, warnings=tuple(warnings),
    )


def compare_methods(
    params: ParamsLike,
    config: NumericalConfig | None = None,
    methods: Iterable[PricingMethod | str] = tuple(PricingMethod),
    *,
    greeks: bool | GreekMethod | str = False,
    **config_overrides: Any,
) -> list[PricingResult]:
    """Price one contract with several methods (prices, errors and runtimes).

    Use :func:`derivatives_engine.reporting.format_comparison` to print a table.
    """
    params = _coerce_params(params)
    config = _build_config(config, config_overrides)
    return [price_option(params, m, config, greeks=greeks) for m in methods]
