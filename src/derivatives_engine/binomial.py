"""Cox–Ross–Rubinstein binomial tree for European options (spec 5.2).

    dt = T/N,  u = exp(sigma*sqrt(dt)),  d = 1/u,
    p  = (exp((r-q) dt) - d) / (u - d),
    V_{i,j} = exp(-r dt) [p V_{i+1,j+1} + (1-p) V_{i+1,j}],

with European payoffs at maturity and no early exercise. The backward
induction is vectorised over each time slice; memory is O(N).

The tree is only arbitrage-free when 0 <= p <= 1, which is equivalent to
N >= T (r-q)^2 / sigma^2. Invalid trees are rejected with a message that
states the minimum step count instead of being silently used.

Zero volatility is handled separately: the CRR lattice collapses (u = d = 1,
p undefined), so the deterministic discounted payoff is returned, which is
the exact limit of the tree price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs
from .types import OptionParams
from .validation import ValidationError, require_int


@dataclass(frozen=True)
class CRRParameters:
    dt: float
    u: float
    d: float
    p: float
    discount: float


def min_valid_steps(params: OptionParams) -> int:
    """Smallest N for which the CRR probability lies in [0, 1]."""
    if params.volatility == 0.0:
        return 1
    carry = params.rate - params.dividend_yield
    return max(1, math.ceil(params.maturity * (carry / params.volatility) ** 2))


def crr_parameters(params: OptionParams, steps: int) -> CRRParameters:
    """Compute and validate the CRR lattice parameters."""
    steps = require_int(steps, "tree_steps", 1)
    dt = params.maturity / steps
    a = params.volatility * math.sqrt(dt)
    u, d = math.exp(a), math.exp(-a)
    growth = math.exp((params.rate - params.dividend_yield) * dt)
    if u == d:
        raise ValidationError("CRR tree is degenerate (u == d); volatility too small for a lattice.")
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValidationError(
            f"CRR risk-neutral probability p={p:.6g} lies outside [0, 1] for N={steps} "
            f"(|r-q|*sqrt(dt) exceeds sigma). Use at least "
            f"{min_valid_steps(params)} steps for these inputs."
        )
    return CRRParameters(dt=dt, u=u, d=d, p=p, discount=math.exp(-params.rate * dt))


def terminal_prices(params: OptionParams, steps: int) -> np.ndarray:
    """Terminal asset prices S0 u^j d^(N-j), j = 0..N (ascending)."""
    a = params.volatility * math.sqrt(params.maturity / steps)
    j = np.arange(steps + 1, dtype=np.float64)
    return params.spot * np.exp(a * (2.0 * j - steps))


def price(params: OptionParams, steps: int = 500) -> float:
    """CRR price of the European option in ``params`` using ``steps`` steps."""
    steps = require_int(steps, "tree_steps", 1)
    if bs.is_deterministic(params):
        return bs.price(params)

    crr = crr_parameters(params, steps)
    s_t = terminal_prices(params, steps)
    if params.is_call:
        values = np.maximum(s_t - params.strike, 0.0)
    else:
        values = np.maximum(params.strike - s_t, 0.0)

    pu = crr.discount * crr.p
    pd = crr.discount * (1.0 - crr.p)
    for _ in range(steps):
        # values[j] holds the node with j up-moves; its up child is values[j+1].
        values = pu * values[1:] + pd * values[:-1]
    return float(values[0])
