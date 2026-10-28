"""Command-line interface: ``derivatives-engine {price,compare,parity,benchmark}``.

Examples::

    derivatives-engine price --spot 100 --strike 100 --rate 0.05 --vol 0.2 --maturity 1 \\
        --method monte_carlo --paths 200000 --seed 42 --greeks
    derivatives-engine compare --spot 100 --strike 95 --rate 0.03 --vol 0.25 --maturity 0.5 --put
    derivatives-engine parity --method binomial --steps 1000 ...
    derivatives-engine benchmark --out results/ [--quick] [--scenarios benchmarks/scenarios.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import reporting
from .api import compare_methods, price_option
from .benchmarking import run_benchmarks
from .parity import validate_put_call_parity
from .types import NumericalConfig, OptionParams
from .validation import NumericalError, ValidationError


def _add_contract_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("contract and market")
    g.add_argument("--spot", type=float, required=True)
    g.add_argument("--strike", type=float, required=True)
    g.add_argument("--rate", type=float, required=True, help="continuously compounded, decimal")
    g.add_argument("--vol", "--volatility", dest="volatility", type=float, required=True)
    g.add_argument("--maturity", type=float, required=True, help="years")
    g.add_argument("--div", "--dividend-yield", dest="dividend_yield", type=float, default=0.0)
    kind = g.add_mutually_exclusive_group()
    kind.add_argument("--call", dest="option_type", action="store_const", const="call")
    kind.add_argument("--put", dest="option_type", action="store_const", const="put")
    p.set_defaults(option_type="call")


def _add_numeric_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("numerical configuration")
    g.add_argument("--steps", type=int, default=500, help="binomial tree steps")
    g.add_argument("--paths", type=int, default=100_000, help="Monte Carlo paths")
    g.add_argument("--seed", type=int, default=None, help="Monte Carlo seed")
    g.add_argument("--antithetic", action="store_true")
    g.add_argument("--control-variate", action="store_true")


def _params(a: argparse.Namespace) -> OptionParams:
    return OptionParams(spot=a.spot, strike=a.strike, rate=a.rate, volatility=a.volatility,
                        maturity=a.maturity, dividend_yield=a.dividend_yield,
                        option_type=a.option_type)


def _config(a: argparse.Namespace) -> NumericalConfig:
    return NumericalConfig(tree_steps=a.steps, n_paths=a.paths, seed=a.seed,
                           antithetic=a.antithetic, control_variate=a.control_variate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="derivatives-engine",
                                     description="European option pricing and Greeks engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("price", help="price one option and print the JSON result")
    _add_contract_args(p)
    _add_numeric_args(p)
    p.add_argument("--method", default="black_scholes",
                   help="black_scholes | binomial | monte_carlo")
    p.add_argument("--greeks", nargs="?", const=True, default=False,
                   help="also compute Greeks (optionally: analytical | finite_difference)")

    c = sub.add_parser("compare", help="compare all three methods on one contract")
    _add_contract_args(c)
    _add_numeric_args(c)
    c.add_argument("--greeks", action="store_true")
    c.add_argument("--json", action="store_true", help="print JSON instead of a table")

    v = sub.add_parser("parity", help="check put–call parity for a method")
    _add_contract_args(v)
    _add_numeric_args(v)
    v.add_argument("--method", default="black_scholes")

    b = sub.add_parser("benchmark", help="run the reproducible benchmark suite")
    b.add_argument("--out", default="benchmark_results", help="output directory")
    b.add_argument("--scenarios", default=None, help="scenario JSON file (default: built-in)")
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--seed", type=int, default=20_240_601)
    b.add_argument("--quick", action="store_true",
                   help="small grids for a fast smoke run (max 1e5 paths, 250 steps)")
    b.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "price":
            result = price_option(_params(args), args.method, _config(args), greeks=args.greeks)
            print(json.dumps(result.to_dict(), indent=2))
        elif args.command == "compare":
            results = compare_methods(_params(args), _config(args), greeks=args.greeks)
            if args.json:
                print(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                print(reporting.format_comparison(results))
                for r in results:
                    for w in r.warnings:
                        print(f"warning [{r.method.value}]: {w}", file=sys.stderr)
        elif args.command == "parity":
            res = validate_put_call_parity(_params(args), args.method, _config(args))
            print(json.dumps(res.to_dict(), indent=2))
            return 0 if res.passed else 2
        elif args.command == "benchmark":
            kwargs = {}
            if args.quick:
                kwargs = dict(tree_steps=(25, 50, 100, 250), path_counts=(1_000, 10_000, 100_000),
                              greek_paths=20_000, greek_tree_steps=200)
            report = run_benchmarks(args.scenarios, repeats=args.repeats, seed=args.seed,
                                    output_dir=args.out, make_plots=not args.no_plots,
                                    progress=lambda m: print(m, file=sys.stderr), **kwargs)
            print(f"wrote {len(report.files)} files to {args.out}:")
            for kind, path in report.files.items():
                print(f"  {kind:14s} {path}")
    except (ValidationError, NumericalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
