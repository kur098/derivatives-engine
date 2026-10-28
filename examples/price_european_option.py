"""Price one European option with every method, inspect Greeks and validate.

    python examples/price_european_option.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivatives_engine import (  # noqa: E402
    NumericalConfig,
    OptionParams,
    calculate_greeks,
    price_option,
    validate_price_bounds,
    validate_put_call_parity,
)

params = OptionParams(
    spot=100.0,
    strike=100.0,
    rate=0.05,
    volatility=0.20,
    maturity=1.0,
    dividend_yield=0.0,
    option_type="call",
)

# 1. Analytical reference (spec 12.1 example).
result = price_option(params, method="black_scholes")
greeks = calculate_greeks(params, method="black_scholes", greek_method="analytical")
print("Black–Scholes price:", round(result.price, 6))
print(greeks)
print(f"  vega per vol point = {greeks.vega_per_pct:.4f}, "
      f"theta per day (approx.) = {greeks.theta_per_day:.4f}")

# 2. Numerical methods with explicit, reproducible configuration.
config = NumericalConfig(tree_steps=1000, n_paths=500_000, seed=42,
                         antithetic=True, control_variate=True)
for method in ("binomial", "monte_carlo"):
    r = price_option(params, method, config, greeks="finite_difference")
    line = (f"{method:12s} price={r.price:.6f}  |err|={r.abs_error:.2e}  "
            f"kernel={r.runtime:.2f} ms")
    if r.std_error is not None:
        lo, hi = r.confidence_interval
        line += f"  SE={r.std_error:.2e}  95% CI=[{lo:.4f}, {hi:.4f}]"
    print(line)
    print("   ", r.greeks, "schemes:", dict(r.greeks.schemes))

# 3. Validation.
for method in ("black_scholes", "binomial", "monte_carlo"):
    parity = validate_put_call_parity(params, method, config)
    print(f"parity [{method}]: passed={parity.passed}  residual={parity.residual:+.2e}")
print("bounds:", validate_price_bounds(params, result.price).passed)

# 4. Full structured result (spec 4.2 output contract).
print(json.dumps(price_option(params, "monte_carlo", config, greeks=True).to_dict(), indent=2))
