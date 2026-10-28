"""Compare methods on one contract, and show convergence of the numerical ones.

    python examples/compare_methods.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivatives_engine import NumericalConfig, OptionParams, compare_methods, price_option  # noqa: E402
from derivatives_engine.reporting import format_comparison, format_table  # noqa: E402

params = OptionParams(spot=100, strike=95, rate=0.03, volatility=0.25, maturity=0.5,
                      dividend_yield=0.01, option_type="put")

print("== All methods, with Greeks ==")
results = compare_methods(params, NumericalConfig(tree_steps=1000, n_paths=200_000, seed=7),
                          greeks=True)
print(format_comparison(results))

print("\n== Binomial convergence ==")
rows = []
for n in (25, 50, 100, 250, 500, 1000, 2000):
    r = price_option(params, "binomial", tree_steps=n)
    rows.append({"tree_steps": n, "price": r.price, "abs_error": r.abs_error,
                 "runtime_ms": r.runtime})
print(format_table(rows))

print("\n== Monte Carlo convergence (seed 7) ==")
rows = []
for m in (1_000, 10_000, 100_000, 1_000_000):
    for vr in (False, True):
        r = price_option(params, "monte_carlo", n_paths=m, seed=7, antithetic=vr,
                         control_variate=vr)
        rows.append({"n_paths": m, "variance_reduction": vr, "price": r.price,
                     "abs_error": r.abs_error, "std_error": r.std_error,
                     "error/SE": r.abs_error / r.std_error, "runtime_ms": r.runtime})
print(format_table(rows))
