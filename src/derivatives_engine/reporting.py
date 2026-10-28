"""Tables, CSV/JSON output and Matplotlib plots (FR-11, spec 8.3).

Matplotlib is imported lazily so the core library works without it.
Charts use one shared categorical order (identity never rides on color
alone: every series also has its own marker and a legend).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .types import PricingResult

#: Fixed categorical order (assigned by entity, never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                 "#8f5bd6", "#c4453c"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_INK, _MUTED, _GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


# --------------------------------------------------------------------------- tables
def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None,
                 digits: int = 6) -> str:
    """Render dict rows as a fixed-width text table."""
    if not rows:
        return "(no rows)"
    columns = list(columns or rows[0].keys())
    cells = [[_fmt(r.get(c), digits) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    line = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join([line, sep, *body])


def comparison_rows(results: Iterable[PricingResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        cfg = r.config
        row: dict[str, Any] = {
            "method": r.method.value,
            "config": ", ".join(f"{k}={v}" for k, v in cfg.items()
                                if k in ("tree_steps", "n_paths", "seed")) or "-",
            "price": r.price,
            "abs_error": r.abs_error,
            "rel_error": r.rel_error,
            "std_error": r.std_error,
            "runtime_ms": r.runtime,
        }
        if r.greeks is not None:
            row.update(delta=r.greeks.delta, gamma=r.greeks.gamma,
                       vega=r.greeks.vega, theta=r.greeks.theta)
        rows.append(row)
    return rows


def format_comparison(results: Iterable[PricingResult]) -> str:
    """Text table of prices, errors and runtimes from ``compare_methods``."""
    return format_table(comparison_rows(results))


# --------------------------------------------------------------------------- files
def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                             for k, v in row.items()})
    return path


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _clean(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(data), indent=2, default=_json_default))
    return path


# --------------------------------------------------------------------------- plots
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": _MUTED, "axes.labelcolor": _INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": _INK,
        "xtick.color": _MUTED, "ytick.color": _MUTED, "axes.grid": True,
        "grid.color": _GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
        "lines.markersize": 6, "figure.dpi": 120, "savefig.bbox": "tight",
    })
    return plt


def _style(i: int) -> dict[str, Any]:
    return {"color": SERIES_COLORS[i % len(SERIES_COLORS)],
            "marker": SERIES_MARKERS[i % len(SERIES_MARKERS)],
            "markeredgecolor": "white", "markeredgewidth": 1.0}


def _floor(x: float) -> float:
    """Keep exact zeros visible on log axes."""
    return max(x, 1e-16)


def plot_binomial_convergence(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    by_scenario: dict[str, list] = defaultdict(list)
    for r in records:
        if r["method"] == "binomial":
            by_scenario[r["scenario"]].append(r)
    for i, (name, rows) in enumerate(sorted(by_scenario.items())):
        rows.sort(key=lambda r: r["tree_steps"])
        ax.plot([r["tree_steps"] for r in rows], [_floor(r["abs_error"]) for r in rows],
                label=name, **_style(i))
    if by_scenario:
        steps = sorted({r["tree_steps"] for rows in by_scenario.values() for r in rows})
        ref0 = median(_floor(r["abs_error"]) for rows in by_scenario.values() for r in rows
                      if r["tree_steps"] == steps[0])
        ax.plot(steps, [ref0 * steps[0] / n for n in steps], color=_MUTED, linestyle="--",
                linewidth=1.2, label="O(1/N) reference")
    ax.set(xscale="log", yscale="log", xlabel="Tree steps N",
           ylabel="|binomial − Black–Scholes|", title="CRR binomial convergence")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_mc_convergence(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    variants = sorted({r["variant"] for r in records if r["method"] == "monte_carlo"})
    scenarios = sorted({r["scenario"] for r in records if r["method"] == "monte_carlo"})
    for ax, variant in zip(axes, variants):
        for i, name in enumerate(scenarios):
            rows = sorted((r for r in records if r["method"] == "monte_carlo"
                           and r["variant"] == variant and r["scenario"] == name),
                          key=lambda r: r["n_paths"])
            x = [r["n_paths"] for r in rows]
            st = _style(i)
            ax.plot(x, [_floor(r["abs_error"]) for r in rows], label=f"{name} |error|", **st)
            ax.plot(x, [_floor(r["mc_std_error"]) for r in rows], color=st["color"],
                    linestyle=":", linewidth=1.5)
        ax.set(xscale="log", yscale="log", xlabel="Simulated paths M",
               title=f"Monte Carlo ({variant})")
    axes[0].set_ylabel("|MC − Black–Scholes|  (dotted: standard error)")
    if len(variants) < len(axes):
        for ax in axes[len(variants):]:
            ax.set_visible(False)
    axes[0].legend(fontsize=7, ncol=1)
    fig.suptitle("Monte Carlo convergence: error tracks the O(1/√M) standard error",
                 fontsize=11, color=_INK)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_error_vs_runtime(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Median kernel runtime vs median absolute error across scenarios, per configuration."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        if r["method"] == "black_scholes":
            continue
        size = r.get("tree_steps") or r.get("n_paths")
        groups[(r["method"], r["variant"], size)].append(r)
    series: dict[tuple, list] = defaultdict(list)
    for (method, variant, size), rows in groups.items():
        series[(method, variant)].append(
            (size, median(r["kernel_median_ms"] for r in rows),
             median(_floor(r["abs_error"]) for r in rows)))
    for i, (key, pts) in enumerate(sorted(series.items())):
        pts.sort()
        ax.plot([p[1] for p in pts], [p[2] for p in pts], label=" / ".join(key), **_style(i))
        ax.annotate(f"{pts[-1][0]:,}", (pts[-1][1], pts[-1][2]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=_MUTED, va="center")
    ax.set(xscale="log", yscale="log", xlabel="Median kernel runtime (ms)",
           ylabel="Median |error| vs Black–Scholes",
           title="Accuracy vs runtime (median over scenarios)")
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_greek_bump_stability(greek_records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Median relative error of FD Greeks vs bump scale, per pricing method."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    greeks = ["delta", "gamma", "vega", "theta"]
    methods = sorted({r["method"] for r in greek_records})
    for ax, g in zip(axes.flat, greeks):
        for i, m in enumerate(methods):
            by_scale: dict[float, list] = defaultdict(list)
            for r in greek_records:
                if r["method"] == m and r["greek"] == g and r["rel_error"] is not None:
                    by_scale[r["bump_scale"]].append(r["rel_error"])
            scales = sorted(by_scale)
            ax.plot(scales, [_floor(median(by_scale[s])) for s in scales], label=m, **_style(i))
        ax.set(xscale="log", yscale="log", title=g.capitalize())
        ax.set_ylabel("median relative error")
    for ax in axes[1]:
        ax.set_xlabel("bump scale (× default bump)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Finite-difference Greeks vs analytical Black–Scholes", fontsize=11, color=_INK)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
