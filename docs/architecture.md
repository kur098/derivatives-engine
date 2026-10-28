# Architecture

```text
cli.py / api.py (price_option, compare_methods)
        │
types.py + validation.py  ── OptionParams, NumericalConfig, enums, ValidationError
        │
pricers.py  ── common interface: make_pricer(method, config) -> (OptionParams -> float)
   ├── black_scholes.py   analytical price + Greeks (reference)
   ├── binomial.py        CRR tree
   └── monte_carlo.py     terminal-price simulation, SE/CI, variance reduction
        │
greeks.py   ── analytical or finite differences over any Pricer
parity.py   ── put–call parity, no-arbitrage bounds
benchmarking.py ── scenario grids, timings, metadata  →  reporting.py (tables, CSV/JSON, plots)
```

## Principles (spec 2.1) and how they are met

| Principle | Implementation |
|---|---|
| Model independence | Every method is a `Pricer`; Greeks, parity and benchmarks only use that interface. |
| Reproducibility | `NumericalConfig.seed` drives `default_rng`; seeds are recorded in every result and benchmark row. A warning is attached when no seed is set. |
| Reference-based validation | Black–Scholes is the benchmark for `abs_error`/`rel_error`, Greek errors and convergence. |
| Separation of concerns | One module per concern, each with its own test file. |
| Backend interchangeability | Results and benchmark rows carry a `backend` field (`"python"`); a native backend can implement the same `Pricer` signature and be compared through the same tables. |

## Data flow of `price_option`

1. Coerce/validate inputs (`OptionParams`, `NumericalConfig` + keyword overrides).
2. Time the pricing kernel only (`runtime`).
3. Optionally compute Greeks.
4. Optionally compute error against Black–Scholes.
5. Check no-arbitrage bounds (with MC slack).
6. Return a frozen `PricingResult`; `total_runtime` covers steps 1–5.

## Errors

* `ValidationError` (subclass of `ValueError`): invalid inputs, configurations or
  tree parameters.
* `NumericalError` (subclass of `RuntimeError`): undefined quantities such as
  d₁ at zero volatility when requested directly.

## Not included

The optional C++/pybind11 backend (spec FR-10, a *Should*) is not part of this
Python package. `test_cpp_parity.py` and `cpp/` would plug in via the `Pricer`
interface and the `backend` column.
