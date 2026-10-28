"""AT-07: invalid inputs and configurations raise informative errors."""

import math

import numpy as np
import pytest

from derivatives_engine import (NumericalConfig, OptionParams, OptionType, PricingMethod,
                                ValidationError, calculate_greeks, price_option)
from derivatives_engine.types import parse_greek_method, parse_method

BASE = dict(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, maturity=1.0)


@pytest.mark.parametrize("field,value,match", [
    ("spot", 0.0, "spot must be > 0"),
    ("spot", -1.0, "spot must be > 0"),
    ("strike", 0.0, "strike must be > 0"),
    ("volatility", -0.1, "volatility must be >= 0"),
    ("maturity", 0.0, "maturity must be > 0"),
    ("maturity", -1.0, "maturity must be > 0"),
    ("spot", math.nan, "spot must not be NaN"),
    ("rate", math.inf, "rate must be finite"),
    ("dividend_yield", -math.inf, "dividend_yield must be finite"),
    ("volatility", "0.2", "volatility must be a real number"),
    ("spot", True, "spot must be a real number"),
    ("strike", None, "strike must be a real number"),
])
def test_invalid_params(field, value, match):
    with pytest.raises(ValidationError, match=match):
        OptionParams(**{**BASE, field: value})


def test_negative_rate_and_dividend_allowed():
    p = OptionParams(**{**BASE, "rate": -0.02, "dividend_yield": -0.01})
    assert p.rate == -0.02 and p.dividend_yield == -0.01


def test_zero_volatility_allowed():
    assert OptionParams(**{**BASE, "volatility": 0.0}).volatility == 0.0


def test_numpy_scalars_accepted():
    p = OptionParams(spot=np.float64(100), strike=np.int64(95), rate=0.0,
                     volatility=np.float32(0.2), maturity=1)
    assert isinstance(p.spot, float) and p.strike == 95.0


@pytest.mark.parametrize("value", ["CALL", " call ", "c", OptionType.CALL])
def test_option_type_parsing(value):
    assert OptionParams(**BASE, option_type=value).option_type is OptionType.CALL


def test_unsupported_option_type():
    with pytest.raises(ValidationError, match="Unsupported option_type 'straddle'"):
        OptionParams(**BASE, option_type="straddle")


def test_unsupported_method():
    with pytest.raises(ValidationError, match="Unsupported pricing method 'heston'"):
        parse_method("heston")
    with pytest.raises(ValidationError):
        price_option(OptionParams(**BASE), method="american_tree")


def test_method_aliases():
    assert parse_method("BS") is PricingMethod.BLACK_SCHOLES
    assert parse_method("mc") is PricingMethod.MONTE_CARLO
    assert parse_method("crr") is PricingMethod.BINOMIAL
    assert parse_greek_method("fd").value == "finite_difference"


def test_unknown_mapping_keys():
    with pytest.raises(ValidationError, match="Unknown option parameter"):
        OptionParams.from_mapping({**BASE, "vol": 0.2})


@pytest.mark.parametrize("kwargs,match", [
    ({"tree_steps": 0}, "tree_steps must be >= 1"),
    ({"tree_steps": 10.0}, "tree_steps must be an integer"),
    ({"n_paths": 0}, "n_paths must be >= 1"),
    ({"n_paths": True}, "n_paths must be an integer"),
    ({"seed": -1}, "seed must be >= 0"),
    ({"seed": 1.5}, "seed must be an integer"),
    ({"batch_size": 0}, "batch_size must be >= 1"),
    ({"confidence_level": 1.0}, "confidence_level must be in"),
    ({"antithetic": True, "n_paths": 1001}, "must be an even number"),
    ({"control_variate": True, "n_paths": 2}, "at least 3 independent samples"),
    ({"antithetic": "yes"}, "antithetic must be a bool"),
])
def test_invalid_config(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        NumericalConfig(**kwargs)


def test_config_overrides_validated_in_price_option():
    with pytest.raises(ValidationError, match="tree_steps"):
        price_option(OptionParams(**BASE), "binomial", tree_steps=-5)


def test_analytical_greeks_rejected_for_numerical_methods():
    with pytest.raises(ValidationError, match="only available for black_scholes"):
        calculate_greeks(OptionParams(**BASE), "binomial", "analytical")


def test_invalid_bumps():
    with pytest.raises(ValidationError, match="Unknown bump key"):
        calculate_greeks(OptionParams(**BASE), "black_scholes", "fd", bumps={"rho": 0.1})
    with pytest.raises(ValidationError, match="must be > 0"):
        calculate_greeks(OptionParams(**BASE), "black_scholes", "fd", bumps={"vol": 0.0})


def test_params_are_immutable():
    p = OptionParams(**BASE)
    with pytest.raises(AttributeError):
        p.spot = 50.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        p.replace(spot=-1)
