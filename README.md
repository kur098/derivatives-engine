# Derivatives Pricing and Greeks Engine

A Python library that prices European calls and puts three ways — Black–Scholes
(closed form), a Cox–Ross–Rubinstein binomial tree and Monte Carlo simulation —
computes Delta, Gamma, Vega and Theta analytically or by finite differences,
validates put–call parity and no-arbitrage bounds, and benchmarks accuracy
against runtime reproducibly.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # numpy, scipy, + pytest and matplotlib
pytest                      # 215 tests, a few seconds
```

Python ≥ 3.11. Core dependencies are NumPy and SciPy; Matplotlib is only needed
for plots.

## Quick start

```python
from derivatives_engine import OptionParams, price_option, calculate_greeks

params = OptionParams(spot=100.0, strike=100.0, rate=0.05, volatility=0.20,
                      maturity=1.0, dividend_yield=0.0, option_type="call")

result = price_option(params, method="black_scholes")
greeks = calculate_greeks(params, method="black_scholes", greek_method="analytical")
print("Price:", result.price)      # 10.450583572185565
print("Greeks:", greeks)           # delta 0.636831, gamma 0.018762, vega 37.524035, theta -6.414028

# Numerical methods: reproducible configuration, error vs BS, SE and CI.
mc = price_option(params, "monte_carlo", n_paths=500_000, seed=42,
                  antithetic=True, control_variate=True, greeks="fd")
print(mc.price, mc.std_error, mc.confidence_interval, mc.abs_error)
print(mc.to_dict())                # JSON schema from spec §4.2
```

Validation:

```python
from derivatives_engine import NumericalConfig, validate_put_call_parity, validate_price_bounds

validate_put_call_parity(params, "binomial", NumericalConfig(tree_steps=1000)).passed      # True
validate_put_call_parity(params, "monte_carlo", NumericalConfig(n_paths=10**5, seed=1))   # z·SE check
validate_price_bounds(params, result.price).passed                                         # True
```

## CLI

```bash
derivatives-engine price   --spot 100 --strike 100 --rate 0.05 --vol 0.2 --maturity 1 \
                           --method monte_carlo --paths 200000 --seed 42 --greeks
derivatives-engine compare --spot 100 --strike 95 --rate 0.03 --vol 0.25 --maturity 0.5 --put --greeks
derivatives-engine parity  --spot 100 --strike 95 --rate 0.03 --vol 0.25 --maturity 0.5 --method binomial
derivatives-engine benchmark --out reports/benchmark        # full suite, ~10 s
python benchmarks/run_benchmarks.py --quick                  # same, without installing
```

`python -m derivatives_engine ...` works too.

## What's in the box

| Spec item | Where |
|---|---|
| FR-01 input validation | `types.py`, `validation.py` — NaN/inf, signs, enums, integer checks, descriptive `ValidationError` |
| FR-02 Black–Scholes | `black_scholes.py` — dividend yield, zero-vol deterministic limit, tail-accurate N(−d) |
| FR-03 CRR tree | `binomial.py` — vectorised backward induction, rejects p ∉ [0,1] and names the minimum N |
| FR-04 Monte Carlo | `monte_carlo.py` — exact terminal sampling, batching, SE + CI, seeded `default_rng` |
| FR-05/06 Greeks | `greeks.py` — analytical BS; finite differences for any method with reported bumps/schemes, boundary handling, common random numbers, lattice-aware stencils for the tree |
| FR-07 parity & bounds | `parity.py` — absolute checks for deterministic methods, z·SE checks for MC with shared paths |
| FR-08/FR-11 benchmarks | `benchmarking.py`, `reporting.py` — kernel vs end-to-end timing, environment metadata, CSV/JSON, plots, Markdown report |
| FR-09 API | `api.py` — `price_option`, `compare_methods`, structured `PricingResult` |
| FR-12 variance reduction | antithetic variates, control variate on the discounted terminal price |

Docs: [`docs/mathematics.md`](docs/mathematics.md) (formulas, units, edge cases,
numerical choices), [`docs/architecture.md`](docs/architecture.md),
[`docs/api.md`](docs/api.md).

## Conventions

Vega is per 1.00 of decimal volatility (`vega_per_pct` divides by 100). Theta is
per year, Θ = −∂V/∂T (`theta_per_day` divides by 365, labelled an
approximation). Runtimes are in milliseconds and labelled `runtime_unit`.

## Benchmark results

From `reports/benchmark/` (7 scenarios, Python 3.13, NumPy 2.5, 2-vCPU Linux VM;
regenerate with `derivatives-engine benchmark`). Errors are medians over
scenarios against Black–Scholes.

| Method | Configuration | Median abs error | Median kernel time |
|---|---|---|---|
| Binomial | N = 100 | 2.0e-2 | 0.20 ms |
| Binomial | N = 1,000 | 2.0e-3 | 2.4 ms |
| Monte Carlo, plain | M = 10⁶ | 2.0e-2 (SE 1.2e-2) | 17 ms |
| Monte Carlo, antithetic + CV | M = 10⁶ | 1.3e-3 (SE 2.7e-3) | 13 ms |

Finite-difference Greeks at default bumps, median relative error vs analytical:
Black–Scholes ≤ 2e-6, binomial (N=500) ≤ 2.3e-3, Monte Carlo (10⁵ paths,
common random numbers) 0.3–1.3%, with Gamma the noisiest. Put–call parity holds
to ~1e-12 for the tree and within sampling error for Monte Carlo in every
scenario. See `reports/benchmark/benchmark_report.md` and the plots alongside it.

These timings describe this pure-Python/NumPy backend on one machine; they are
not acceleration claims and not statements about executable market prices.

## Notes and limitations

* **No C++ backend.** The optional native backend (spec FR-10, *Should*) is not
  included. Results and benchmark rows carry a `backend` field, and any
  implementation of the `Pricer` signature can be compared through the same tables.
* **CRR oscillation.** Tree prices converge as O(1/N) but not monotonically.
  Greeks on the tree therefore use bumps that keep the strike in the same position
  relative to the nodes; naive small bumps would give badly wrong Gamma. When
  the strike is 3–7 node spacings from spot, tree Vega/Theta can still carry
  ~1% lattice noise.
* **Monte Carlo Greeks** use finite differences with common random numbers.
  Gamma of a kinked payoff remains noisy; use more paths or variance reduction.
* **MC confidence intervals** describe sampling uncertainty; a single run can
  fall outside its interval.
* Out of scope (spec §1.3): American/exotic options, stochastic or local
  volatility, calibration, implied-vol surfaces, trading.
