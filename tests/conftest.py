import pytest

from derivatives_engine import OptionParams


@pytest.fixture
def ref_call() -> OptionParams:
    """Spec 13.1 reference contract."""
    return OptionParams(spot=100.0, strike=100.0, rate=0.05, volatility=0.2,
                        maturity=1.0, dividend_yield=0.0, option_type="call")


@pytest.fixture
def ref_put(ref_call: OptionParams) -> OptionParams:
    return ref_call.replace(option_type="put")


# A spread of well-conditioned contracts used across test modules.
CASES = [
    OptionParams(100, 100, 0.05, 0.20, 1.0, 0.0, "call"),
    OptionParams(100, 100, 0.05, 0.20, 1.0, 0.0, "put"),
    OptionParams(100, 90, 0.03, 0.25, 0.5, 0.02, "call"),
    OptionParams(100, 80, 0.04, 0.30, 1.0, 0.0, "put"),
    OptionParams(100, 110, 0.05, 0.60, 2.0, 0.01, "call"),
    OptionParams(100, 100, -0.01, 0.15, 2.0, 0.0, "put"),
]
CASE_IDS = ["atm_call", "atm_put", "itm_call_div", "otm_put", "high_vol_call", "neg_rate_put"]
