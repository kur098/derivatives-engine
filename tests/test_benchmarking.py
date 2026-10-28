"""AT-10: benchmark reporting, plus CLI smoke tests."""

import csv
import json

import pytest

from derivatives_engine import run_benchmarks
from derivatives_engine.benchmarking import load_scenarios, time_callable
from derivatives_engine.cli import main
from derivatives_engine.validation import ValidationError

SCEN = [{"name": "atm", "spot": 100, "strike": 100, "rate": 0.05, "volatility": 0.2,
         "maturity": 1.0, "option_type": "call"},
        {"name": "otm_put", "spot": 100, "strike": 85, "rate": 0.02, "volatility": 0.3,
         "maturity": 0.5, "option_type": "put"}]


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    report = run_benchmarks(SCEN, tree_steps=(10, 50), path_counts=(1_000, 4_000),
                            bump_scales=(0.1, 1.0), greek_tree_steps=100, greek_paths=4_000,
                            repeats=2, warmup=1, seed=7, output_dir=out, make_plots=True)
    return report, out


def test_records_contain_required_fields(small_run):
    report, _ = small_run
    rec = report.pricing_records
    # 1 BS + 2 binomial + 2 MC sizes x 2 variants per scenario.
    assert len(rec) == 2 * (1 + 2 + 4)
    for r in rec:
        for key in ("kernel_median_ms", "kernel_iqr_ms", "end_to_end_median_ms", "abs_error",
                    "rel_error", "backend", "method", "scenario"):
            assert key in r
        assert r["kernel_median_ms"] >= 0
    mc = [r for r in rec if r["method"] == "monte_carlo"]
    assert all(r["seed"] == 7 and r["mc_std_error"] > 0 for r in mc)
    assert report.greek_records and all(r["scheme"] for r in report.greek_records)
    assert all(r["passed"] for r in report.parity_records)


def test_metadata(small_run):
    md = small_run[0].metadata
    for key in ("python", "numpy", "scipy", "os", "cpu_count", "thread_env", "timestamp_utc"):
        assert key in md


def test_files_written(small_run):
    report, out = small_run
    for name in ("benchmark_results.json", "pricing_benchmarks.csv", "greek_benchmarks.csv",
                 "parity_checks.csv", "benchmark_report.md", "binomial_convergence.png",
                 "mc_convergence.png", "error_vs_runtime.png", "greek_bump_stability.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    data = json.loads((out / "benchmark_results.json").read_text())
    assert data["settings"]["seed"] == 7 and "report" in data["files"]
    rows = list(csv.DictReader((out / "pricing_benchmarks.csv").open()))
    assert len(rows) == len(report.pricing_records)


def test_reproducible_prices(tmp_path):
    kw = dict(tree_steps=(10,), path_counts=(2_000,), bump_scales=(1.0,), greek_paths=2_000,
              greek_tree_steps=20, repeats=1, warmup=0, seed=3, variance_reduction=False)
    a = run_benchmarks(SCEN[:1], **kw)
    b = run_benchmarks(SCEN[:1], **kw)
    assert [r["price"] for r in a.pricing_records] == [r["price"] for r in b.pricing_records]


def test_time_callable():
    stats, out = time_callable(lambda: 42, repeats=3, warmup=2)
    assert out == 42 and stats.repeats == 3 and stats.median_ms >= 0


def test_load_scenarios(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(SCEN))
    assert load_scenarios(path)[0]["name"] == "atm"
    path.write_text(json.dumps([{"name": "bad", "spot": -1, "strike": 1, "rate": 0,
                                 "volatility": 0.2, "maturity": 1}]))
    with pytest.raises(ValidationError):
        load_scenarios(path)


def test_shipped_scenarios_file_is_valid():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "scenarios.json"
    assert len(load_scenarios(path)) >= 5


# ----------------------------------------------------------------------- CLI
CONTRACT = ["--spot", "100", "--strike", "100", "--rate", "0.05", "--vol", "0.2", "--maturity", "1"]


def test_cli_price(capsys):
    assert main(["price", *CONTRACT, "--method", "mc", "--paths", "5000", "--seed", "1",
                 "--greeks"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "monte_carlo" and out["greeks"]["delta"] > 0


def test_cli_compare_and_parity(capsys):
    assert main(["compare", *CONTRACT, "--put", "--paths", "5000", "--seed", "1"]) == 0
    assert "monte_carlo" in capsys.readouterr().out
    assert main(["parity", *CONTRACT, "--method", "binomial", "--steps", "100"]) == 0


def test_cli_reports_validation_errors(capsys):
    bad = ["--spot", "-5", *CONTRACT[2:]]
    assert main(["price", *bad]) == 1
    assert "spot must be > 0" in capsys.readouterr().err
