"""Run the full benchmark suite from the repository root.

    python benchmarks/run_benchmarks.py                 # full grids -> benchmark_results/
    python benchmarks/run_benchmarks.py --quick         # fast smoke run
    python benchmarks/run_benchmarks.py --out results/ --repeats 7

Equivalent to ``derivatives-engine benchmark --scenarios benchmarks/scenarios.json``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # allow running without installation

from derivatives_engine.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--scenarios" not in args:
        args += ["--scenarios", str(ROOT / "benchmarks" / "scenarios.json")]
    sys.exit(main(["benchmark", *args]))
