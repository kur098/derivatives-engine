# Mathematics and conventions

## Assumptions

European exercise; geometric Brownian motion with constant volatility σ,
constant continuously compounded rate r and continuous dividend yield q; no
transaction costs. r and q may be negative. σ ≥ 0, T > 0, S₀ > 0, K > 0.

## Units

| Quantity | Unit |
|---|---|
| r, q, σ | annualised decimals (0.05 = 5%) |
| T | years |
| Delta | per 1 unit of spot |
| Gamma | per 1 unit of spot, squared |
| Vega | per 1.00 change in decimal volatility (`vega_per_pct` = vega / 100) |
| Theta | per year of calendar time, Θ = −∂V/∂T (`theta_per_day` = Θ / 365, a labelled approximation) |
| Runtime | milliseconds (`runtime_unit` is recorded in every result) |

## Black–Scholes (reference)

d₁ = [ln(S₀/K) + (r − q + σ²/2)T] / (σ√T), d₂ = d₁ − σ√T

C = S₀e^{−qT}N(d₁) − Ke^{−rT}N(d₂), P = Ke^{−rT}N(−d₂) − S₀e^{−qT}N(−d₁)

N is SciPy's `ndtr`; N(−d) is evaluated directly rather than as 1 − N(d) so
that deep in/out-of-the-money values keep their precision. Prices are clamped
at 0 to remove round-off negatives.

**Zero volatility.** When σ√T < 10⁻¹², the price is the discounted
deterministic payoff max(S₀e^{−qT} − Ke^{−rT}, 0) (mirror for puts). Greeks
use the σ → 0 limit: N(d) becomes an indicator (½ exactly at the forward),
n(d₁) → 0, so Γ = ν = 0 and Θ reduces to the carry terms.

Analytical Greeks follow spec §6.2 exactly.

## CRR binomial tree

Δt = T/N, u = e^{σ√Δt}, d = 1/u, p = (e^{(r−q)Δt} − d)/(u − d).
Backward induction V = e^{−rΔt}[pV_up + (1−p)V_down] from European payoffs,
vectorised over each slice (O(N) memory, O(N²) work).

**Validity.** 0 ≤ p ≤ 1 ⇔ N ≥ T(r−q)²/σ². Smaller trees raise
`ValidationError` naming the minimum N. σ = 0 is priced via the deterministic
payoff, the exact limit of the tree.

**Parity.** For any valid tree, E[S_T] = S₀e^{(r−q)T} exactly, so CRR satisfies
put–call parity to floating-point precision.

**Convergence.** Error is O(1/N) but oscillates with the position of the strike
relative to the terminal nodes (and with N's parity), so convergence is not
monotone.

## Monte Carlo

S_T = S₀ exp[(r − q − σ²/2)T + σ√T Z]. A single draw per path is exact for
European payoffs. Estimator: mean of discounted payoffs X; SE = s_X/√M; CI =
estimate ± z_{(1+level)/2}·SE (normal approximation). Moments are accumulated
across batches with Chan's pairwise update.

**Antithetic variates.** Each draw Z is paired with −Z; the pair average is
one i.i.d. unit, so SE is computed over M/2 pair averages.

**Control variate.** C = e^{−rT}S_T with known mean S₀e^{−qT}. Estimate
X̄ − β̂(C̄ − E[C]), β̂ = Cov(X, C)/Var(C), residual variance with n − 2 degrees
of freedom. Applied to the linear payoff S_T − K it reproduces the forward
exactly, so MC parity residuals become ≈ 0.

**Reproducibility.** `numpy.random.default_rng(seed)`; draws are consumed
sequentially, so results do not depend on `batch_size`. Draws do not depend on
the option inputs, so re-pricing with the same seed gives common random
numbers for finite differences.

## Finite-difference Greeks

Central differences per spec §6.3. Default bumps:

| Method | spot | σ | T |
|---|---|---|---|
| Black–Scholes | 0.1% of S₀ | 10⁻⁴ | 10⁻⁴ y |
| Binomial | ≥ 0.1% of S₀, rounded up to whole lattice spacings | 10⁻³ | 10⁻³ y |
| Monte Carlo | 2% of S₀ | 10⁻² | 10⁻² y |

Monte Carlo bumps are larger because the payoff kink makes small-bump
differences (especially Gamma) noisy even with common random numbers.

**Domain boundaries.** If σ − h < 0 or T − h ≤ 0 the bump is reduced to a tenth
of the distance to the boundary. If σ = 0 (or an absolute spot bump ≥ S₀) a
second-order one-sided forward scheme is used. If a bumped valuation is itself
invalid (e.g. an invalid CRR tree) that Greek is NaN with a note.

**Tree-aware stencils.** Naive bumps move the strike relative to the CRR nodes
and inject the lattice oscillation into the difference (a 0.1% spot bump on a
500-step tree can give a Gamma several times too large). The engine instead:

* bumps spot by u^{±2k}: the lattice shifts by k whole nodes and the strike's
  relative position is unchanged, then applies second-order non-uniform
  three-point formulas;
* when the strike is ≥ 7 node spacings from spot, bumps σ and T so the strike
  moves by exactly ±1 node: σ′ = σC/(C ∓ 1), T′ = T(C/(C ∓ 1))², with
  C = |ln(K/S)|/(2σ√Δt). Closer to the money the phase moves slowly and
  ordinary bumps are used.

The scheme used for each Greek is recorded in `Greeks.schemes`.

## Validation

Parity residual ε = C − P − S₀e^{−qT} + Ke^{−rT}; bounds per spec §7.2.
Deterministic methods use an absolute tolerance 10⁻⁹·max(1, S₀, K). Monte Carlo
checks allow z·SE (default z = 4) plus that floor, using shared paths for C and P.

Relative error: |V − V_BS| / max(|V_BS|, ε), ε = 10⁻⁸ by default; results warn
when V_BS < 10⁻⁶ because the ratio is then not meaningful.
