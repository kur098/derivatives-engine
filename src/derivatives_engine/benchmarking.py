"""Reproducible accuracy / convergence / runtime benchmarks (spec 7.4, 8).

Protocol: fixed scenario set -> warm-up -> repeated timed runs -> median and
IQR runtimes -> error against Black–Scholes -> repeat over tree sizes, path
counts and variance-reduction variants -> CSV/JSON + plots + Markdown report.

Two timings are recorded per configuration:

* ``kernel_*``: the pricing function alone,
* ``end_to_end_*``: ``price_option`` including input validation, the
  Black–Scholes benchmark and bound checks (API latency).

Only the Python backend exists in this package; the ``backend`` column is
there so a native backend can be added to the same tables later.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import __version__, black_scholes, reporting
from .api import price_option
from .greeks import DEFAULT_BUMPS, calculate_greeks
from .parity import validate_put_call_parity
from .pricers import make_pricer
from .types import NumericalConfig, OptionParams, PricingMethod
from .validation import ValidationError, require_int

DEFAULT_TREE_STEPS = (25, 50, 100, 250, 500, 1000)
DEFAULT_PATH_COUNTS = (1_000, 10_000, 100_000, 1_000_000)
DEFAULT_BUMP_SCALES = (0.01, 0.1, 1.0, 10.0)

DEFAULT_SCENARIOS: list[dict[str, Any]] = [
    {"name": "atm_call_1y", "spot": 100, "strike": 100, "rate": 0.05, "volatility": 0.20,
     "maturity": 1.0, "dividend_yield": 0.0, "option_type": "call"},
    {"name": "atm_put_1y", "spot": 100, "strike": 100, "rate": 0.05, "volatility": 0.20,
     "maturity": 1.0, "dividend_yield": 0.0, "option_type": "put"},
    {"name": "itm_call_dividend", "spot": 100, "strike": 90, "rate": 0.03, "volatility": 0.25,
     "maturity": 0.5, "dividend_yield": 0.02, "option_type": "call"},
    {"name": "otm_put", "spot": 100, "strike": 80, "rate": 0.04, "volatility": 0.30,
     "maturity": 1.0, "dividend_yield": 0.0, "option_type": "put"},
    {"name": "short_maturity_call", "spot": 100, "strike": 102, "rate": 0.05,
     "volatility": 0.20, "maturity": 1 / 52, "dividend_yield": 0.0, "option_type": "call"},
    {"name": "high_vol_call_2y", "spot": 100, "strike": 110, "rate": 0.05, "volatility": 0.60,
     "maturity": 2.0, "dividend_yield": 0.01, "option_type": "call"},
    {"name": "negative_rate_put", "spot": 100, "strike": 100, "rate": -0.01,
     "volatility": 0.15, "maturity": 2.0, "dividend_yield": 0.0, "option_type": "put"},
]


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    iqr_ms: float
    min_ms: float
    repeats: int


def time_callable(fn: Callable[[], Any], repeats: int = 5, warmup: int = 1) -> tuple[TimingStats, Any]:
    """Run ``fn`` ``warmup`` times untimed, then ``repeats`` times timed."""
    repeats = require_int(repeats, "repeats", 1)
    warmup = require_int(warmup, "warmup", 0)
    result = None
    for _ in range(warmup):
        result = fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    q1, q3 = (np.percentile(samples, [25, 75]) if len(samples) > 1 else (samples[0], samples[0]))
    return TimingStats(statistics.median(samples), float(q3 - q1), min(samples), repeats), result


def environment_metadata() -> dict[str, Any]:
    """Hardware / software context recorded with every benchmark run (spec 8.2)."""
    import scipy
    meta = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "backend": "python",
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "compiler": "n/a (pure Python/NumPy backend)",
    }
    try:
        import matplotlib
        meta["matplotlib"] = matplotlib.__version__
    except ImportError:  # pragma: no cover
        meta["matplotlib"] = None
    return meta


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    """Load scenarios from JSON: a list of objects with ``name`` + OptionParams fields."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    if not isinstance(data, list) or not data:
        raise ValidationError(f"{path}: expected a non-empty list of scenarios.")
    for i, s in enumerate(data):
        if "name" not in s:
            raise ValidationError(f"{path}: scenario {i} has no 'name'.")
        _scenario_params(s)  # validate early
    return data


def _scenario_params(scenario: Mapping[str, Any]) -> OptionParams:
    return OptionParams.from_mapping({k: v for k, v in scenario.items() if k != "name"})


@dataclass
class BenchmarkReport:
    metadata: dict[str, Any]
    settings: dict[str, Any]
    pricing_records: list[dict[str, Any]] = field(default_factory=list)
    greek_records: list[dict[str, Any]] = field(default_factory=list)
    parity_records: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "settings": self.settings,
                "pricing": self.pricing_records, "greeks": self.greek_records,
                "parity": self.parity_records, "files": self.files}


def _pricing_record(name: str, params: OptionParams, method: PricingMethod, variant: str,
                    config: NumericalConfig, repeats: int, warmup: int,
                    bs_price: float) -> dict[str, Any]:
    pricer_cfg = config
    if method is PricingMethod.MONTE_CARLO:
        from . import monte_carlo
        kernel = lambda: monte_carlo.estimate(params, pricer_cfg)  # noqa: E731
    else:
        pricer = make_pricer(method, config)
        kernel = lambda: pricer(params)  # noqa: E731
    k_stats, out = time_callable(kernel, repeats, warmup)
    e_stats, result = time_callable(lambda: price_option(params, method, config), repeats, warmup)
    price = out.price if hasattr(out, "price") else out
    se = getattr(out, "std_error", None)
    abs_err = abs(price - bs_price)
    rec = {
        "scenario": name, "option_type": params.option_type.value, "method": method.value,
        "variant": variant, "backend": "python",
        "tree_steps": config.tree_steps if method is PricingMethod.BINOMIAL else None,
        "n_paths": config.n_paths if method is PricingMethod.MONTE_CARLO else None,
        "seed": config.seed if method is PricingMethod.MONTE_CARLO else None,
        "repeats": repeats, "warmup": warmup,
        "kernel_median_ms": k_stats.median_ms, "kernel_iqr_ms": k_stats.iqr_ms,
        "kernel_min_ms": k_stats.min_ms,
        "end_to_end_median_ms": e_stats.median_ms, "end_to_end_iqr_ms": e_stats.iqr_ms,
        "price": price, "bs_price": bs_price, "abs_error": abs_err,
        "rel_error": abs_err / max(abs(bs_price), 1e-8),
        "mc_std_error": se,
        "z_score": (price - bs_price) / se if se else None,
        "validation_passed": result.validation_passed,
    }
    return rec


def run_benchmarks(
    scenarios: Sequence[Mapping[str, Any]] | str | Path | None = None,
    *,
    tree_steps: Iterable[int] = DEFAULT_TREE_STEPS,
    path_counts: Iterable[int] = DEFAULT_PATH_COUNTS,
    bump_scales: Iterable[float] = DEFAULT_BUMP_SCALES,
    greek_tree_steps: int = 500,
    greek_paths: int = 100_000,
    repeats: int = 5,
    warmup: int = 1,
    seed: int = 20_240_601,
    variance_reduction: bool = True,
    output_dir: str | Path | None = None,
    make_plots: bool = True,
    progress: Callable[[str], None] | None = None,
) -> BenchmarkReport:
    """Run the benchmark suite and optionally write CSV/JSON/plots/report.

    Args:
        scenarios: list of scenario dicts, a JSON path, or None for defaults.
        tree_steps / path_counts: convergence grids.
        bump_scales: multipliers of the default finite-difference bumps.
        greek_tree_steps / greek_paths: configuration for numerical Greeks.
        repeats / warmup: timing protocol.
        seed: Monte Carlo seed (fixed => reproducible).
        variance_reduction: also run Monte Carlo with antithetic + control variate.
        output_dir: if given, write all artefacts there.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    elif isinstance(scenarios, (str, Path)):
        scenarios = load_scenarios(scenarios)
    tree_steps, path_counts = list(tree_steps), list(path_counts)
    bump_scales = [float(s) for s in bump_scales]
    say = progress or (lambda msg: None)

    report = BenchmarkReport(
        metadata=environment_metadata(),
        settings={"tree_steps": tree_steps, "path_counts": path_counts,
                  "bump_scales": bump_scales, "greek_tree_steps": greek_tree_steps,
                  "greek_paths": greek_paths, "repeats": repeats, "warmup": warmup,
                  "seed": seed, "variance_reduction": variance_reduction,
                  "runtime_unit": "ms", "scenarios": [dict(s) for s in scenarios]},
    )

    mc_variants = [("plain", {})]
    if variance_reduction:
        mc_variants.append(("antithetic+cv", {"antithetic": True, "control_variate": True}))

    for scenario in scenarios:
        name = scenario["name"]
        params = _scenario_params(scenario)
        bs_price = black_scholes.price(params)
        say(f"[{name}] pricing")
        report.pricing_records.append(_pricing_record(
            name, params, PricingMethod.BLACK_SCHOLES, "analytical", NumericalConfig(),
            repeats, warmup, bs_price))
        for n in tree_steps:
            report.pricing_records.append(_pricing_record(
                name, params, PricingMethod.BINOMIAL, "crr", NumericalConfig(tree_steps=n),
                repeats, warmup, bs_price))
        for m in path_counts:
            for variant, extra in mc_variants:
                m_eff = m + (m % 2) if extra.get("antithetic") else m
                cfg = NumericalConfig(n_paths=m_eff, seed=seed, **extra)
                report.pricing_records.append(_pricing_record(
                    name, params, PricingMethod.MONTE_CARLO, variant, cfg, repeats, warmup,
                    bs_price))

        say(f"[{name}] greeks")
        analytical = black_scholes.greeks(params)
        greek_cfgs = {
            PricingMethod.BLACK_SCHOLES: NumericalConfig(),
            PricingMethod.BINOMIAL: NumericalConfig(tree_steps=greek_tree_steps),
            PricingMethod.MONTE_CARLO: NumericalConfig(n_paths=greek_paths, seed=seed),
        }
        for method, cfg in greek_cfgs.items():
            for scale in bump_scales:
                base = DEFAULT_BUMPS[method]
                bumps = {k: v * scale for k, v in base.items()}
                if bumps["spot_rel"] >= 0.5:
                    continue
                stats, g = time_callable(
                    lambda: calculate_greeks(params, method, "finite_difference", cfg, bumps),
                    repeats=1, warmup=0)
                for gname, est, ref in zip(("delta", "gamma", "vega", "theta"),
                                           g.as_tuple(), analytical.as_tuple()):
                    err = abs(est - ref)
                    report.greek_records.append({
                        "scenario": name, "method": method.value, "bump_scale": scale,
                        "greek": gname, "estimate": est, "analytical": ref,
                        "abs_error": err,
                        "rel_error": err / abs(ref) if abs(ref) > 1e-12 else None,
                        "scheme": g.schemes.get(gname), "bumps": dict(g.bumps),
                        "runtime_ms": stats.median_ms,
                        "tree_steps": cfg.tree_steps if method is PricingMethod.BINOMIAL else None,
                        "n_paths": cfg.n_paths if method is PricingMethod.MONTE_CARLO else None,
                    })

        say(f"[{name}] parity")
        for method, cfg in greek_cfgs.items():
            pr = validate_put_call_parity(params, method, cfg)
            report.parity_records.append({"scenario": name, **pr.to_dict()})

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report.files["json"] = str(reporting.write_json(out / "benchmark_results.json", report.to_dict()))
        report.files["pricing_csv"] = str(reporting.write_csv(out / "pricing_benchmarks.csv",
                                                              report.pricing_records))
        report.files["greeks_csv"] = str(reporting.write_csv(out / "greek_benchmarks.csv",
                                                             report.greek_records))
        report.files["parity_csv"] = str(reporting.write_csv(out / "parity_checks.csv",
                                                             report.parity_records))
        if make_plots:
            say("plots")
            report.files["binomial_plot"] = str(reporting.plot_binomial_convergence(
                report.pricing_records, out / "binomial_convergence.png"))
            report.files["mc_plot"] = str(reporting.plot_mc_convergence(
                report.pricing_records, out / "mc_convergence.png"))
            report.files["runtime_plot"] = str(reporting.plot_error_vs_runtime(
                report.pricing_records, out / "error_vs_runtime.png"))
            report.files["greeks_plot"] = str(reporting.plot_greek_bump_stability(
                report.greek_records, out / "greek_bump_stability.png"))
        report.files["report"] = str(write_markdown_report(report, out / "benchmark_report.md"))
        # Re-write JSON so it lists every artefact.
        reporting.write_json(out / "benchmark_results.json", report.to_dict())
    return report


def write_markdown_report(report: BenchmarkReport, path: str | Path) -> Path:
    """Human-readable summary of a benchmark run."""
    md = report.metadata
    lines = [
        "# Benchmark report", "",
        f"Generated {md['timestamp_utc']} · derivatives-engine {md['package_version']} · "
        f"backend `{md['backend']}`", "",
        "## Environment", "",
        f"- Python {md['python']} ({md['python_implementation']}), NumPy {md['numpy']}, "
        f"SciPy {md['scipy']}",
        f"- {md['os']} · {md['machine']} · {md['cpu_count']} logical CPUs",
        f"- Thread env: {md['thread_env']}",
        f"- Timing: {report.settings['warmup']} warm-up + {report.settings['repeats']} "
        "timed runs, median reported (ms). Kernel = pricing function only; "
        "end-to-end = `price_option` incl. validation and benchmark.", "",
    ]

    def table(rows, cols):
        lines.append("```text")
        lines.append(reporting.format_table(rows, cols, digits=6))
        lines.append("```")
        lines.append("")

    pr = report.pricing_records
    for title, method in (("Binomial convergence", "binomial"),
                          ("Monte Carlo convergence", "monte_carlo")):
        lines += [f"## {title}", ""]
        keys = ("tree_steps",) if method == "binomial" else ("variant", "n_paths")
        agg: dict[tuple, list] = {}
        for r in pr:
            if r["method"] == method:
                agg.setdefault(tuple(r[k] for k in keys), []).append(r)
        rows = []
        for key, rs in sorted(agg.items()):
            row = dict(zip(keys, key))
            row["median_abs_error"] = statistics.median(r["abs_error"] for r in rs)
            row["max_abs_error"] = max(r["abs_error"] for r in rs)
            if method == "monte_carlo":
                row["median_std_error"] = statistics.median(r["mc_std_error"] for r in rs)
                row["max_|z|"] = max(abs(r["z_score"]) for r in rs if r["z_score"] is not None)
            row["median_kernel_ms"] = statistics.median(r["kernel_median_ms"] for r in rs)
            row["median_e2e_ms"] = statistics.median(r["end_to_end_median_ms"] for r in rs)
            rows.append(row)
        table(rows, list(rows[0].keys()) if rows else None)

    lines += ["## Finite-difference Greeks at default bumps (relative error vs analytical)", ""]
    rows = []
    for method in ("black_scholes", "binomial", "monte_carlo"):
        for g in ("delta", "gamma", "vega", "theta"):
            errs = [r["rel_error"] for r in report.greek_records
                    if r["method"] == method and r["greek"] == g and r["bump_scale"] == 1.0
                    and r["rel_error"] is not None]
            if errs:
                rows.append({"method": method, "greek": g,
                             "median_rel_error": statistics.median(errs),
                             "max_rel_error": max(errs)})
    table(rows, ["method", "greek", "median_rel_error", "max_rel_error"])

    lines += ["## Put–call parity", ""]
    table([{"scenario": r["scenario"], "method": r["method"], "residual": r["residual"],
            "tolerance": r["tolerance"], "passed": r["passed"]} for r in report.parity_records],
          ["scenario", "method", "residual", "tolerance", "passed"])

    plots = [(k, v) for k, v in report.files.items() if v.endswith(".png")]
    if plots:
        lines += ["## Plots", ""]
        lines += [f"![{k}]({Path(v).name})" for k, v in plots]
        lines.append("")
    lines += ["## Caveats", "",
              "- Errors are measured against the Black–Scholes closed form under identical "
              "assumptions; they are not claims about executable market prices.",
              "- Monte Carlo errors are single seeded draws; compare them with the reported "
              "standard error rather than reading them as deterministic accuracy.",
              "- Runtimes depend on hardware, BLAS threading and system load; no native "
              "backend is included, so no acceleration factors are claimed.", ""]
    path = Path(path)
    path.write_text("\n".join(lines))
    return path
