"""FR-08/FR-09: public API, result schema and reporting."""

import json

import pytest

from derivatives_engine import (NumericalConfig, OptionParams, PricingMethod, compare_methods,
                                price_option)
from derivatives_engine.reporting import format_comparison


def test_spec_example(ref_call):
    result = price_option(ref_call, method="black_scholes")
    assert result.price == pytest.approx(10.4506, abs=1e-4)
    assert result.method is PricingMethod.BLACK_SCHOLES
    assert result.validation_passed is True
    assert result.abs_error is None  # BS is its own benchmark


def test_accepts_mapping():
    r = price_option({"spot": 100, "strike": 100, "rate": 0.05, "volatility": 0.2,
                      "maturity": 1.0, "option_type": "put"})
    assert r.price == pytest.approx(5.5735, abs=1e-4)


def test_result_schema_is_json_serialisable(ref_call):
    r = price_option(ref_call, "monte_carlo", n_paths=20_000, seed=1, greeks=True)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["option_type"] == "call" and d["method"] == "monte_carlo"
    assert set(d["greeks"]) >= {"delta", "gamma", "vega", "theta", "bumps", "method"}
    diag = d["diagnostics"]
    assert diag["runtime_unit"] == "ms" and diag["runtime_ms"] > 0
    assert diag["benchmark_error"] is not None and diag["std_error"] > 0
    assert len(diag["confidence_interval"]) == 2 and diag["confidence_level"] == 0.95
    assert d["config"]["seed"] == 1 and d["config"]["n_paths"] == 20_000


def test_binomial_result_records_config_and_error(ref_call):
    r = price_option(ref_call, "binomial", NumericalConfig(tree_steps=200))
    assert r.config == {"tree_steps": 200}
    assert r.abs_error == pytest.approx(abs(r.price - r.benchmark_price))
    assert r.rel_error == pytest.approx(r.abs_error / r.benchmark_price)
    assert r.std_error is None and r.confidence_interval is None


def test_warnings(ref_call):
    r = price_option(ref_call, "monte_carlo", n_paths=500)
    assert any("not reproducible" in w for w in r.warnings)
    assert any("fewer than 1000" in w for w in r.warnings)
    z = price_option(ref_call.replace(volatility=0.0), "binomial")
    assert any("zero volatility" in w for w in z.warnings)
    tiny = price_option(OptionParams(100, 400, 0.01, 0.1, 0.1), "binomial", tree_steps=50)
    assert any("near zero" in w for w in tiny.warnings)


def test_greek_options(ref_call):
    assert price_option(ref_call).greeks is None
    assert price_option(ref_call, greeks="analytical").greeks.method.value == "analytical"
    fd = price_option(ref_call, greeks="fd", bumps={"spot_rel": 0.01})
    assert fd.greeks.bumps["spot"] == pytest.approx(1.0)


def test_compare_methods(ref_call):
    results = compare_methods(ref_call, NumericalConfig(tree_steps=200, n_paths=20_000, seed=4))
    assert [r.method.value for r in results] == ["black_scholes", "binomial", "monte_carlo"]
    table = format_comparison(results)
    assert "binomial" in table and "tree_steps=200" in table


def test_total_runtime_includes_kernel(ref_call):
    r = price_option(ref_call, "binomial", tree_steps=300, greeks=True)
    assert r.total_runtime >= r.runtime > 0
