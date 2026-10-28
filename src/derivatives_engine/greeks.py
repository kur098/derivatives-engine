"""Greeks engine: analytical (Black–Scholes) and finite differences (any method).

Finite-difference schemes (spec 6.3)::

    Delta ~ [V(S+h) - V(S-h)] / 2h
    Gamma ~ [V(S+h) - 2V(S) + V(S-h)] / h^2
    Vega  ~ [V(s+h) - V(s-h)] / 2h
    Theta ~ -[V(T+h) - V(T-h)] / 2h      (calendar-time decay, per year)

When a central difference would leave the valid domain (sigma - h < 0,
T - h <= 0) the bump is reduced to a tenth of the distance to the boundary; if the
input sits on the boundary (sigma = 0) or an absolute spot bump exceeds S, a
second-order one-sided forward scheme is used. If a bumped valuation is itself
invalid (e.g. a CRR tree with p outside [0, 1]) that Greek is returned as NaN
with an explanatory note. Schemes, bump sizes and notes are always reported.

Monte Carlo Greeks use common random numbers: every bumped valuation reuses
the same seed and hence the same normal draws. If no seed was configured a
seed is generated once, used for all bumps and recorded.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from . import black_scholes
from .pricers import Pricer, make_pricer
from .types import (GreekMethod, Greeks, NumericalConfig, OptionParams, PricingMethod,
                    parse_greek_method, parse_method)
from .validation import ValidationError, require_finite

#: Default bumps per pricing method. ``spot_rel`` is a fraction of spot;
#: ``vol`` and ``time`` are absolute (decimal vol, years).
#: Tree and Monte Carlo prices are not smooth in their inputs at fine scales
#: (lattice re-alignment / payoff kink), so they need larger bumps than the
#: analytical model, trading truncation error for stability.
DEFAULT_BUMPS: dict[PricingMethod, dict[str, float]] = {
    PricingMethod.BLACK_SCHOLES: {"spot_rel": 1e-3, "vol": 1e-4, "time": 1e-4},
    PricingMethod.BINOMIAL: {"spot_rel": 1e-3, "vol": 1e-3, "time": 1e-3},
    PricingMethod.MONTE_CARLO: {"spot_rel": 2e-2, "vol": 1e-2, "time": 1e-2},
}

_BUMP_KEYS = {"spot_rel", "spot_abs", "vol", "time"}


def resolve_bumps(method: PricingMethod, params: OptionParams,
                  bumps: Mapping[str, float] | None = None) -> dict[str, float]:
    """Merge user bumps with defaults and convert to absolute sizes."""
    merged = dict(DEFAULT_BUMPS[method])
    if bumps:
        unknown = set(bumps) - _BUMP_KEYS
        if unknown:
            raise ValidationError(f"Unknown bump key(s) {sorted(unknown)}; allowed {sorted(_BUMP_KEYS)}.")
        merged.update({k: require_finite(v, f"bump '{k}'") for k, v in bumps.items()})
    h_s = merged.pop("spot_abs", None) or merged["spot_rel"] * params.spot
    out = {"spot": h_s, "spot_rel": h_s / params.spot, "vol": merged["vol"], "time": merged["time"]}
    for key in ("spot", "vol", "time"):
        if not out[key] > 0:
            raise ValidationError(f"Bump size for {key} must be > 0, got {out[key]}.")
    return out


def _first_and_second(pricer: Pricer, params: OptionParams, field: str, h: float,
                      v0: float, central_ok: bool) -> tuple[float, float, str]:
    """First derivative, second derivative and scheme name along ``field``."""
    x = getattr(params, field)
    if central_ok:
        up = pricer(params.replace(**{field: x + h}))
        dn = pricer(params.replace(**{field: x - h}))
        return (up - dn) / (2 * h), (up - 2 * v0 + dn) / (h * h), "central"
    v1 = pricer(params.replace(**{field: x + h}))
    v2 = pricer(params.replace(**{field: x + 2 * h}))
    first = (-3 * v0 + 4 * v1 - v2) / (2 * h)  # second-order forward
    second = (v0 - 2 * v1 + v2) / (h * h)
    return first, second, "forward"


def _nonuniform(pricer: Pricer, params: OptionParams, field: str, x_dn: float, x_up: float,
                v0: float) -> tuple[float, float]:
    """Second-order three-point first/second derivatives on a non-uniform stencil."""
    x = getattr(params, field)
    v_up = pricer(params.replace(**{field: x_up}))
    v_dn = pricer(params.replace(**{field: x_dn}))
    h1, h2 = x - x_dn, x_up - x
    denom = h1 * h2 * (h1 + h2)
    first = (h1 * h1 * v_up - h2 * h2 * v_dn + (h2 * h2 - h1 * h1) * v0) / denom
    second = 2.0 * (h1 * v_up - (h1 + h2) * v0 + h2 * v_dn) / denom
    return first, second


#: Strike distance from spot, in lattice spacings, above which binomial Vega and
#: Theta switch to strike-phase-aligned bumps (see _crr_stencils).
PHASE_ALIGN_MIN_SPACINGS = 7.0


def _crr_stencils(params: OptionParams, steps: int, target_rel: float) -> dict[str, tuple]:
    """Bump stencils that keep CRR discretisation error in phase.

    The CRR price error oscillates with the position of the strike relative to
    the terminal nodes, j* = N/2 + c with c = ln(K/S) / (2 sigma sqrt(dt)).
    A naive bump moves that position by a fraction of a node and injects the
    oscillation into the finite difference. Instead:

    * Spot is bumped by u^(+-2k): the whole lattice shifts by k nodes while the
      strike stays in the same relative position (always used).
    * Volatility and maturity are bumped so that c changes by exactly +-1
      node: sigma' = sigma C/(C -+ 1) and T' = T (C/(C -+ 1))^2 with C = |c|.
      These bumps get large when the strike is close to spot, so they are only
      used when C >= PHASE_ALIGN_MIN_SPACINGS; closer to the money the phase
      moves slowly and the ordinary central bump is used.
    """
    a = params.volatility * np.sqrt(params.maturity / steps)
    k = max(1, int(np.ceil(np.log1p(target_rel) / (2 * a))))
    stencils: dict[str, tuple] = {
        "spot": (params.spot * float(np.exp(-2 * k * a)), params.spot * float(np.exp(2 * k * a)), k)}
    c_abs = abs(float(np.log(params.strike / params.spot)) / (2 * a))
    if c_abs >= PHASE_ALIGN_MIN_SPACINGS:
        lo, hi = c_abs / (c_abs + 1.0), c_abs / (c_abs - 1.0)
        stencils["volatility"] = (params.volatility * lo, params.volatility * hi)
        stencils["maturity"] = (params.maturity * lo * lo, params.maturity * hi * hi)
    return stencils


def finite_difference_greeks(pricer: Pricer, params: OptionParams, method: PricingMethod,
                             bumps: Mapping[str, float] | None = None,
                             notes: tuple[str, ...] = (),
                             tree_steps: int | None = None) -> Greeks:
    """Finite-difference Greeks of an arbitrary pricer.

    If ``tree_steps`` is given (CRR pricer with positive volatility), bumps are
    chosen to keep the lattice error in phase; see :func:`_crr_stencils`.
    """
    h = resolve_bumps(method, params, bumps)
    v0 = pricer(params)
    stencils = (_crr_stencils(params, tree_steps, h["spot_rel"])
                if tree_steps is not None and params.volatility > 0 else {})
    if bumps and "spot_abs" in bumps:
        stencils.pop("spot", None)

    if "spot" in stencils:
        s_dn, s_up, k = stencils["spot"]
        delta, gamma = _nonuniform(pricer, params, "spot", s_dn, s_up, v0)
        h.update(spot=s_up - params.spot, spot_up=s_up, spot_down=s_dn, lattice_spacings=float(k))
        s_scheme = "lattice_aligned_central"
    else:
        delta, gamma, s_scheme = _first_and_second(
            pricer, params, "spot", h["spot"], v0, params.spot - h["spot"] > 0)

    extra = list(notes)

    def bumped_derivative(field: str, key: str, label: str, strictly_positive: bool):
        """Vega / dV/dT with boundary handling; ValidationError -> documented NaN."""
        x = getattr(params, field)
        try:
            if field in stencils:
                lo, hi = stencils[field]
                value, _ = _nonuniform(pricer, params, field, lo, hi, v0)
                h.update({f"{key}_up": hi, f"{key}_down": lo})
                return value, "strike_phase_aligned"
            step = h[key]
            feasible = (x - step > 0) if strictly_positive else (x - step >= 0)
            if not feasible and x > 0:
                # Shrink the bump to stay inside the domain rather than go one-sided.
                step = x / 10
                h[key] = step
                extra.append(f"{label}: bump reduced to {step:.3g} to keep the central "
                             "difference inside the valid domain")
                feasible = True
            value, _, scheme = _first_and_second(pricer, params, field, step, v0, feasible)
            if scheme == "forward":
                extra.append(f"{label}: central difference infeasible at {field}={x:g}, "
                             "used one-sided forward scheme")
            return value, scheme
        except ValidationError as exc:
            extra.append(f"{label}: not available, a bumped valuation was invalid ({exc})")
            return float("nan"), "unavailable"

    vega, v_scheme = bumped_derivative("volatility", "vol", "Vega", strictly_positive=False)
    dv_dt, t_scheme = bumped_derivative("maturity", "time", "Theta", strictly_positive=True)
    if s_scheme == "forward":
        extra.append("Delta/Gamma: central difference infeasible, used one-sided forward scheme")
    return Greeks(
        delta=delta, gamma=gamma, vega=vega, theta=-dv_dt,
        method=GreekMethod.FINITE_DIFFERENCE, pricing_method=method,
        bumps=h,
        schemes={"delta": s_scheme, "gamma": s_scheme, "vega": v_scheme, "theta": t_scheme},
        notes=tuple(extra),
    )


def calculate_greeks(
    params: OptionParams,
    method: PricingMethod | str = PricingMethod.BLACK_SCHOLES,
    greek_method: GreekMethod | str | None = None,
    config: NumericalConfig | None = None,
    bumps: Mapping[str, float] | None = None,
) -> Greeks:
    """Delta, Gamma, Vega and Theta for ``params``.

    Args:
        params: the option.
        method: pricing method whose prices are differentiated.
        greek_method: ``"analytical"`` (Black–Scholes only) or
            ``"finite_difference"``. Defaults to analytical for Black–Scholes
            and finite differences otherwise.
        config: numerical configuration for tree / Monte Carlo pricers.
        bumps: override bump sizes, keys ``spot_rel`` | ``spot_abs`` | ``vol`` | ``time``.
    """
    method = parse_method(method)
    if greek_method is None:
        greek_method = (GreekMethod.ANALYTICAL if method is PricingMethod.BLACK_SCHOLES
                        else GreekMethod.FINITE_DIFFERENCE)
    greek_method = parse_greek_method(greek_method)
    config = config or NumericalConfig()

    if greek_method is GreekMethod.ANALYTICAL:
        if method is not PricingMethod.BLACK_SCHOLES:
            raise ValidationError(
                f"Analytical Greeks are only available for black_scholes, not {method.value}; "
                "use greek_method='finite_difference'.")
        return black_scholes.greeks(params)

    notes: tuple[str, ...] = ()
    if method is PricingMethod.MONTE_CARLO:
        if config.seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
            config = config.replace(seed=seed)
            notes = (f"no seed configured; generated seed {seed} shared by all bumps",)
        notes += (f"common random numbers: seed {config.seed} reused for every bump",)
    tree_steps = config.tree_steps if method is PricingMethod.BINOMIAL else None
    return finite_difference_greeks(make_pricer(method, config), params, method, bumps, notes,
                                    tree_steps=tree_steps)
