# API reference

All names below are importable from `derivatives_engine`.

## Types

### `OptionParams(spot, strike, rate, volatility, maturity, dividend_yield=0.0, option_type="call")`
Frozen, validated. `option_type` accepts `"call"`/`"put"`/`"c"`/`"p"` or `OptionType`.
Helpers: `replace(**changes)`, `from_mapping(dict)`, `to_dict()`, `discount_factor`,
`dividend_discount`, `discounted_spot`, `discounted_strike`, `is_call`.

### `NumericalConfig(tree_steps=500, n_paths=100_000, seed=None, batch_size=1_000_000, antithetic=False, control_variate=False, confidence_level=0.95)`
Frozen, validated. With `antithetic=True`, `n_paths` counts both members of each pair and must be even.

### `PricingResult`
`params, method, price, config, runtime, total_runtime, runtime_unit ("ms"), greeks,
benchmark_price, abs_error, rel_error, std_error, confidence_interval, confidence_level,
validation_passed, warnings, backend`. `to_dict()` gives the JSON schema of spec §4.2
(`diagnostics.runtime_ms`, `diagnostics.benchmark_error`, ...).

### `Greeks`
`delta, gamma, vega, theta, method, pricing_method, bumps, schemes, notes`;
properties `vega_per_pct`, `theta_per_day`; `as_tuple()`, `to_dict()` (includes units).

## Functions

### `price_option(params, method="black_scholes", config=None, *, greeks=False, bumps=None, benchmark=True, validate=True, rel_error_floor=1e-8, **config_overrides) -> PricingResult`
`method`: `"black_scholes" | "binomial" | "monte_carlo"` (aliases `bs`, `crr`, `mc`).
`greeks`: `False`, `True` (method default), `"analytical"`, `"finite_difference"`/`"fd"`.
Keyword overrides are applied to `config`, e.g. `price_option(p, "mc", n_paths=10**6, seed=1)`.

### `calculate_greeks(params, method="black_scholes", greek_method=None, config=None, bumps=None) -> Greeks`
`greek_method=None` → analytical for Black–Scholes, finite differences otherwise.
`bumps` keys: `spot_rel` (fraction of spot), `spot_abs`, `vol`, `time`.
Analytical Greeks for binomial/Monte Carlo raise `ValidationError`.

### `validate_put_call_parity(params, method="black_scholes", config=None, tolerance=None, z=4.0) -> ParityResult`
Fields: `method, call, put, residual, tolerance, passed, std_error, z_score, diagnostics`.

### `validate_price_bounds(params, price, std_error=None, tolerance=None, z=4.0) -> BoundsResult`
Fields: `price, lower, upper, slack, passed, violations`. `price_bounds(params)` returns `(lower, upper)`.

### `compare_methods(params, config=None, methods=all, *, greeks=False, **overrides) -> list[PricingResult]`
Format with `derivatives_engine.reporting.format_comparison(results)`.

### `run_benchmarks(scenarios=None, *, tree_steps=(25..1000), path_counts=(1e3..1e6), bump_scales=(0.01, 0.1, 1, 10), greek_tree_steps=500, greek_paths=100_000, repeats=5, warmup=1, seed=20240601, variance_reduction=True, output_dir=None, make_plots=True, progress=None) -> BenchmarkReport`
`scenarios`: list of dicts (`name` + `OptionParams` fields), a JSON path, or `None` for the built-in set.
With `output_dir`: writes `benchmark_results.json`, `pricing_benchmarks.csv`,
`greek_benchmarks.csv`, `parity_checks.csv`, four PNG plots and `benchmark_report.md`.

## Lower-level modules

* `black_scholes.price / call_put_prices / greeks / d1_d2 / is_deterministic`
* `binomial.price(params, steps) / crr_parameters / min_valid_steps / terminal_prices`
* `monte_carlo.estimate(params, config) -> MCEstimate`, `price`, `call_put_forward`, `simulate(params, config, payoffs)`
* `pricers.make_pricer(method, config)`
* `greeks.finite_difference_greeks(pricer, params, method, bumps, notes, tree_steps)`

## CLI

```text
derivatives-engine price     <contract> [--method M] [--steps N] [--paths M] [--seed S] [--antithetic] [--control-variate] [--greeks [analytical|fd]]
derivatives-engine compare   <contract> [...] [--greeks] [--json]
derivatives-engine parity    <contract> --method M [...]          # exit code 2 on failure
derivatives-engine benchmark [--out DIR] [--scenarios FILE] [--repeats R] [--seed S] [--quick] [--no-plots]

<contract> = --spot S --strike K --rate r --vol σ --maturity T [--div q] [--call|--put]
```
Exit code 1 with an `error:` message on invalid input.
