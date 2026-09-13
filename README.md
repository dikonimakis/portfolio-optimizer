# Efficient Frontier Portfolio Optimizer

An implementation of **Modern Portfolio Theory (Markowitz, 1952)** with production-grade numerics:
long-only portfolios are solved **exactly** by active-set quadratic programming, estimates are built
from real price history with data-driven shrinkage, portfolios respect real position limits, inputs are
validated for internal consistency, and estimation error is both mitigated and **measured** rather than
hidden.

## Quick start

```bash
# everything: 460 assertions, the browser bundle, and every documented number
npm run verify

# or individually
npm test                    # the eight test suites
npm run verify:bundle       # executes the browser build
npm run verify:findings     # regenerates every number in this file

# fetch real price history (your own machine, your own key)
# stooq needs no key in principle, but as of 2026-09 its public endpoint
# returns an anti-bot JavaScript challenge instead of data for any non-browser
# client — alphavantage (free key from alphavantage.co) is what actually works:
node fetch-market-data.js --source=alphavantage --tickers=SPY,AGG,GLD --key=YOUR_KEY --out=prices.csv

# or crypto, via Tiingo (free token from tiingo.com/account/api/token) —
# inferFrequency auto-detects the 365-day calendar, annualizing correctly
node fetch-market-data.js --source=tiingocrypto --tickers=btcusd,ethusd --key=YOUR_TOKEN --out=crypto.csv

# estimate model inputs from it
node -e "const D=require('./data-pipeline.js'), fs=require('fs');
  const e=D.estimateFromCSV(fs.readFileSync('prices.csv','utf8'));
  console.log(e.names);
  console.log('annual return:', e.mu.map(v=>(v*100).toFixed(1)+'%'));
  console.log('annual vol:   ', e.vols.map(v=>(v*100).toFixed(1)+'%'));
  console.log('LW shrinkage: ', e.shrinkage.delta.toFixed(3));
  e.quality.warnings.forEach(w=>console.log('warning:', w));"
```

Or load the CSV directly into the dashboard, which runs the same pipeline in the browser.

## Files

| File | Purpose |
|---|---|
| `portfolio-core.js` | Optimization. Active-set QP solver with bounds and group constraints, closed-form frontier, shrinkage, sensitivity analysis. Zero dependencies. |
| `data-pipeline.js` | Estimation. Price CSV to annualized mu/Sigma, including data-driven Ledoit-Wolf shrinkage. Zero dependencies. |
| `fetch-market-data.js` | Downloads price history into the CSV format the pipeline expects. Run locally with your own key. |
| `portfolio-core.test.js` | 171 assertions |
| `data-pipeline.test.js` | 68 assertions |
| `backtest.js` | Walk-forward out-of-sample testing. Zero dependencies. |
| `scenarios.js` | Generators for non-stationary markets and a controllable-informativeness leading signal. |
| `fetch-news-sentiment.js` | Downloads per-ticker news sentiment into the JSON format the sentiment strategy expects. Run locally with your own key. |
| `fetch-market-data.test.js` | 67 assertions |
| `backtest.test.js` | 50 assertions |
| `scenarios.test.js` | 38 assertions |
| `black-litterman.test.js` | 29 assertions |
| `sentiment-value.test.js` | 9 assertions |
| `fetch-news-sentiment.test.js` | 28 assertions |
| `portfolio-optimizer.jsx` | React dashboard: efficient frontier plus an out-of-sample backtest tab. |
| `reproduce-findings.js` | Regenerates every number quoted in this file and flags any that no longer match. |
| `bundle-runtime-check.js` | Executes the bundled modules exactly as the browser does, catching wiring bugs the syntax check cannot. |
| `run-real-backtest.js` | Ad hoc script (not part of the test suite): runs the walk-forward backtest comparison on real fetched price history rather than generated scenarios. See section 3m. |
| `ASSESSMENT.md` | Honest evaluation of what this is worth as a product. |

## Verification status

**460 assertions across eight suites, all passing**, plus a bundle runtime check that executes the
browser build end to end and a reproduction script that regenerates all 45 numbers quoted below.

- The QP satisfies full KKT optimality — two-sided bound conditions plus non-negative group multipliers and complementary slackness — across 41 risk-tolerance levels and 800 randomized problems, half of them with random, sometimes overlapping group constraints
- Golden-section search over the constrained frontier reproduces the closed-form maximum-Sharpe solution to 1e-6 when bounds do not bind — an independent check of both methods against each other
- Exact QP beats the best of 20,000 Monte Carlo draws by 1.77% Sharpe; no random portfolio out of 20,000 beats it
- Infeasible mandates are refused with an explanation rather than answered with a plausible-looking wrong portfolio — and, since a Phase 1 was added, *possible* mandates are no longer refused by mistake
- Ledoit-Wolf shrinkage cuts covariance estimation error by 46% on a 40-observation sample, and its intensity falls from 0.226 to 0.001 as the sample grows from 40 to 10,000
- When observations are fewer than assets the sample covariance is singular; shrinkage restores invertibility so the optimizer can run at all
- Every provider response shape is tested, including rate limits, invalid keys, missing adjusted closes, and malformed CSV
- Both fetch scripts have now been run against live APIs, not just recorded fixtures (2026-09). Confirmed working end to end: Alpha Vantage for price history and news sentiment, and Tiingo (added as a third `--source`) for full multi-year adjusted price history. Real findings from that testing: Stooq's public CSV endpoint now returns an anti-bot JavaScript proof-of-work challenge instead of data for any non-browser client (a server-side block this project does not attempt to defeat); Alpha Vantage's free tier no longer serves `outputsize=full` or adjusted closes on `TIME_SERIES_DAILY`, and appears to ignore the `--limit` parameter on `NEWS_SENTIMENT` (10 and 200 both returned exactly 50 articles); and Tiingo's daily-prices endpoint silently returns only the single latest quote, not full history, unless a `startDate` is supplied — `fetch-market-data.js` now always sends one. See the header comments in `fetch-market-data.js` and `fetch-news-sentiment.js` for what changed and how the scripts adapted.
- The browser bundle is executed, not merely parsed: CSV import, estimation, all three backtest strategies, constraint enforcement at every rebalance, and equity-curve construction. This caught a downsampling bug that a syntax check cannot see.
- The backtester is tested for look-ahead bias directly: injecting a price spike at time T leaves every return and every weight before T bit-identical

### What the tests caught

**0. A convincing result that did not survive replication.** Covered in full in section 3h: the first
backtest showed equal weighting beating every optimized strategy, matching a famous published finding.
Twenty independent replications showed the ranking was noise. Listed first because it is the failure
mode least likely to be caught by a test that merely passes.

Kept in full, because a test suite that never embarrasses its author is not being run hard enough.

**1. Broken random number generator.** The Monte Carlo comparison and sensitivity analysis used a textbook linear congruential generator. In JavaScript, multiplying a 31-bit state by a constant near 2^30 exceeds the 2^53 limit of exact integer arithmetic, so low-order bits were silently discarded and the draws were badly non-uniform. It surfaced as sample means failing to converge to known true values. Replaced with mulberry32, which uses `Math.imul`.

**2. A guard in the wrong layer.** `pricesToReturns` refused to return fewer than two observations, enforcing a requirement belonging to the covariance estimators, and made a legitimate partial-data case fail with a misleading error.

**3. Infinite recursion from a careless refactor.** Extracting the provider parsers into pure functions was done with a text substitution that matched inside the new function rather than at the call site, so `buildCSV` called itself forever. The integration test hit it immediately. The file was rewritten cleanly rather than patched further — patching a mangled file compounds the mangling.

**4. A second measurement bug that produced a convincing false finding.** This is the instructive one. The sensitivity analysis reported that position limits *tripled* weight turnover under estimation error, and a persuasive mechanism was available: caps force more assets into the portfolio, so more of them react to noise, and more binding constraints means the binding set flips more often. Every part of that story is plausible. It was also wrong. The baseline inside `sensitivityAnalysis` was being computed *without* the bounds, so constrained portfolios were being compared against an unconstrained portfolio they could never match. With the baseline corrected, limits modestly *reduce* turnover (6.2% versus 7.5%) as well as improving diversification. A mechanism that explains a number is not evidence that the number is right.

Two test *expectations* were also wrong and were corrected rather than loosened: one demanded accuracy better than the statistical standard error permits, and one generated synthetic data whose covariance exactly matched the shrinkage target, making a shrinkage intensity of 1.0 the correct answer rather than a failure.

**5. A dashboard feature that never actually worked, missed by its own verification script.** `portfolio-optimizer.jsx` claimed to inline `data-pipeline.js` and `backtest.js` alongside `portfolio-core.js` (section 3j). It inlined only the latter. CSV import and the entire out-of-sample tab threw `estimateFromCSV is not defined` on first use — not a regression, never functional. `bundle-runtime-check.js`, whose stated purpose is exactly to catch this kind of wiring bug, never caught it, because it independently re-implements the same three files as its own copy and tests that copy, never reading `portfolio-optimizer.jsx` at all. Its "BUNDLE RUNTIME OK" was verifying itself. Found by live-testing the actual dashboard in a browser — the one verification path nothing in the automated suite covered. Fixed by properly inlining both files (which required removing a Node/browser `require` indirection layer in `backtest.js` designed for running standalone, not for a scope where everything is already inlined together — left in place, it silently redeclared two of `portfolio-core.js`'s real functions as broken proxies) and by teaching `bundle-runtime-check.js` to extract and execute `portfolio-optimizer.jsx`'s real source via Node's `vm` module, so a future divergence between the dashboard and its own verification script fails loudly instead of passing quietly.

**6. Silently wrong annualization for a 24/7 market.** Added when testing whether crypto (via a new Tiingo source, `--source=tiingocrypto`) could be fed through the pipeline "without destroying the math." It couldn't, silently: `inferFrequency` treated any daily-cadence series as a 252-day weekday market unconditionally — the only calendar this project had ever tested against. Fed real BTC/ETH price history, the pipeline reported annualized return and volatility using 252 instead of the correct 365, understating BTC's true annualized return by 15 percentage points (33.5% reported vs. 48.5% correct) and volatility by over 11 points (56.5% vs. 67.9%) on live data. A median-gap-of-1-day alone cannot distinguish "weekday market" from "trades every day," so the fix instead checks what *fraction* of gaps are exactly 1 day: a weekday market has a 3-day weekend gap roughly one week in five, a 24/7 calendar has almost none. Guarded to require at least 20 gaps before applying, so a short sample that simply has not crossed a weekend yet is not misread as crypto. This one function is duplicated three times in this codebase (`data-pipeline.js`, `bundle-runtime-check.js`'s copy, and `portfolio-optimizer.jsx`'s inlined copy) and all three needed the identical fix — itself a small case of the risk bug #5 above is really about.

The fix itself briefly broke three of the numbers quoted in section 3f, caught immediately by `reproduce-findings.js` exactly as it is meant to: several existing tests and `reproduce-findings.js`'s own position-limit fixture generate synthetic price series over fully-consecutive calendar dates (weekends included) to represent ordinary stock/bond data, which the new fraction-of-1-day-gaps check correctly, if inconveniently, identified as looking like a 24/7 market. The right fix was the data, not the detector: those fixtures now generate real weekday-only dates, which is what they were always supposed to represent.

## 1. Problem setup

Let there be `n` assets. Define:

- `μ` — n×1 vector of expected returns
- `σ_i` — volatility (standard deviation) of asset `i`
- `ρ_ij` — correlation between assets `i` and `j`
- `Σ` — n×n covariance matrix, `Σ_ij = σ_i · σ_j · ρ_ij`
- `w` — n×1 vector of portfolio weights
- `1` — n×1 vector of ones

Portfolio expected return: `r = wᵀμ`
Portfolio variance: `σ² = wᵀΣw`

**The Markowitz problem:** for a target return `r`, find the weights that minimize risk:

```
minimize    (1/2) wᵀΣw
subject to  wᵀμ = r        (target return)
            wᵀ1 = 1        (fully invested)
```

No `w ≥ 0` constraint here — this is the *unconstrained* (short-selling allowed) version, which has a closed-form solution. Solving it for every `r` traces out the efficient frontier.

## 2. Closed-form derivation

Lagrangian:

```
L(w, λ, γ) = (1/2) wᵀΣw − λ(wᵀμ − r) − γ(wᵀ1 − 1)
```

First-order condition:

```
∂L/∂w = Σw − λμ − γ1 = 0   ⟹   w = Σ⁻¹(λμ + γ1)
```

Substituting back into the two constraints gives a 2×2 linear system in `(λ, γ)`. Define the three scalars:

```
A = 1ᵀΣ⁻¹1        B = 1ᵀΣ⁻¹μ        C = μᵀΣ⁻¹μ        D = AC − B²
```

Then:

```
C·λ + B·γ = r
B·λ + A·γ = 1
```

Solving (Cramer's rule):

```
λ = (Ar − B)/D          γ = (C − Br)/D
```

So the efficient portfolio for target return `r` is:

```
w(r) = Σ⁻¹[ (C − Br)/D · 1 + (Ar − B)/D · μ ]
```

and its variance, obtained from `σ²(r) = wᵀΣw = λr + γ`:

```
σ²(r) = (A·r² − 2B·r + C) / D
```

This is the parabola (in `r`, `σ²` space) — or hyperbola in `(σ, r)` space — that the app plots as the gold **efficient frontier line**. The code sweeps `r` across a range and keeps only points where `σ²(r) ≥ 0`.

### Global minimum-variance (GMV) portfolio

Minimizing `σ²(r)` over `r` (`dσ²/dr = 0`) gives `r_gmv = B/A`, and substituting back:

```
w_gmv = Σ⁻¹1 / A          σ²_gmv = 1/A
```

This is the leftmost point of the frontier — the lowest-risk portfolio achievable, regardless of return.

### Tangency (maximum Sharpe ratio) portfolio

The Sharpe ratio is `S(w) = (wᵀμ − r_f) / √(wᵀΣw)`. Because `S` is scale-invariant in `w` (multiplying `w` by a constant doesn't change it), maximizing it subject to `1ᵀw = 1` reduces to finding the direction that maximizes the ratio, then normalizing:

```
w_tan = Σ⁻¹(μ − r_f·1) / [ 1ᵀΣ⁻¹(μ − r_f·1) ]
```

Geometrically, this is the point where a straight line from `(0, r_f)` — the **Capital Allocation Line** — is tangent to the frontier. It has the steepest possible risk/return slope of any achievable combination of the risky assets.

## 3. Exact long-only optimization (active-set QP)

The closed form above allows negative weights. Real long-only mandates require `w >= 0`, which turns the
problem into a quadratic program with inequality constraints — no closed form exists. Earlier versions of
this project approximated it by sampling random portfolios; that is now replaced by an exact solver.

**The general problem solved:**

```
min (1/2) x'Sx - c'x    s.t.  a'x = b,  x >= 0
```

**KKT conditions.** With multiplier `gamma` on the equality constraint and `s >= 0` on the bounds:

```
Sx - c - gamma*a - s = 0,   s_i * x_i = 0,   a'x = b,   x >= 0
```

Because `S` is positive semi-definite the problem is convex, so **any KKT point is the global optimum**.

**Active-set method.** Maintain a "free set" F of indices allowed to be positive; the rest are pinned at 0.
On F the inequality constraints vanish, leaving a linear system:

```
[ S_FF  -a_F ] [x_F  ]   [c_F]
[ a_F'    0  ] [gamma] = [ b ]
```

Solve it, then repair violations: if a free variable came out negative, drop it from F; if a pinned variable
has a negative multiplier `s_i = (Sx)_i - c_i - gamma*a_i`, release it back into F. Iterate until both
primal and dual feasibility hold — at which point the KKT conditions are satisfied exactly and the solution
is optimal. Terminates in finitely many steps.

**Three problems, one solver:**

| Portfolio | c | a | b |
|---|---|---|---|
| Minimum variance | 0 | 1 | 1 |
| Efficient at risk tolerance tau | tau*mu | 1 | 1 |
| **Maximum Sharpe** | 0 | mu - rf | 1 |

The maximum-Sharpe row uses the **Cornuejols-Tutuncu transformation**: the Sharpe ratio is scale-invariant
in `w`, so instead of maximizing a ratio (not a QP), solve `min y'Sy s.t. (mu-rf)'y = 1, y >= 0` and
recover `w = y / sum(y)`. This is exact — no search over risk-aversion parameters, no simulation.

If no asset has an expected return above the risk-free rate the problem is infeasible; the solver reports
this rather than returning a meaningless portfolio.

## 3b. Input validation

Users can type any numbers into a correlation matrix, but not every such matrix is possible. If A and B are
strongly positively correlated and B and C are too, A and C cannot be strongly negatively correlated. Such
a matrix is not positive semi-definite, implies negative portfolio variance, and yields weights that look
precise but are meaningless.

`validateCorrelation` runs a Jacobi eigenvalue decomposition and rejects any matrix with a negative
eigenvalue. `repairCorrelation` clips negative eigenvalues to a small positive floor, reconstructs the
matrix, and rescales it to unit diagonal — giving the nearest valid correlation matrix. The UI blocks
optimization until the matrix is valid, and offers one-click repair.

## 3c. Shrinkage

Markowitz optimization is famously an "error maximizer" (Michaud, 1989): it loads onto whichever asset has
the most overestimated expected return. Two standard corrections are implemented.

**Covariance shrinkage** toward a constant-correlation target (the Ledoit-Wolf target `F`):

```
Sigma_shrunk = delta*F + (1-delta)*Sigma,   F_ij = rbar * sigma_i * sigma_j  (i != j)
```

*Honest caveat:* the data-optimal `delta` in Ledoit-Wolf requires the raw return series to estimate. This
tool takes summary statistics as input, so `delta` is user-controlled rather than data-derived.

**Expected-return shrinkage** via Jorion (1986) Bayes-Stein, pulling returns toward the GMV return:

```
mu_shrunk = (1-phi)*mu + phi*mu_gmv*1,   phi = (n+2) / [(n+2) + T*(mu - mu_gmv*1)' Sigma^-1 (mu - mu_gmv*1)]
```

`T` is the number of observations behind the estimates — smaller samples shrink harder. The method declines
to shrink when `T <= n+2`, where the estimator is not defined.

## 3d. Sensitivity analysis

Rather than presenting weights with false precision, the tool measures their fragility: it re-optimizes
several hundred times with expected returns perturbed by +/-p%, then reports how much of the portfolio moves
(median and worst-case turnover) plus the range each weight occupied. On the default assumptions, +/-10%
error in returns moves a median of 3.1% of the portfolio; +/-30% moves 11.4%.

This is deliberately a headline feature. A tool that shows only a single optimal answer overstates what
mean-variance optimization can deliver.

## 3e. From price history to model inputs (`data-pipeline.js`)

Manual assumptions were the weakest part of the earlier build: the model's precision was entirely
notional if the numbers going in were guesses. This module estimates them from actual price series.

**Pipeline:** CSV → returns → periodic estimates → annualized `mu`, `Sigma`, `rho`.

**Log versus simple returns.** Log returns are time-additive, so the mean scales linearly with the
number of periods per year. Simple returns compound: `(1+r)^k - 1`. Both are offered; the annualization
differs correctly between them.

**Missing data.** Any period where an asset lacks a usable price is dropped for *all* assets.
Forward-filling would be easier but understates volatility and correlation, quietly flattering the
portfolio. Dropped periods are counted and reported.

**Frequency inference.** Annualization needs periods-per-year. Rather than assume daily, the median
gap between dates is measured and mapped to daily/weekly/monthly/quarterly/annual.

**True Ledoit-Wolf shrinkage.** This is the substantive gain from having raw returns. Section 3c noted
that the data-optimal shrinkage intensity could not be computed from summary statistics — with a return
series it can:

```
delta* = (pi - rho) / gamma / T
```

where `pi` sums the asymptotic variances of the sample covariance entries, `rho` the asymptotic
covariances between sample and target estimation error, and `gamma` the squared Frobenius distance
between them. The user no longer picks a shrinkage parameter; the data picks it.

Measured behaviour on synthetic data with a known covariance:

| Observations | Chosen `delta` |
|---|---|
| 40 | 0.226 |
| 250 | 0.035 |
| 1,500 | 0.006 |
| 10,000 | 0.001 |

Shrinkage is strong when data is scarce and fades as the sample covariance becomes reliable — which is
precisely the intended behaviour, and it is derived, not tuned.

**Alternative estimators.** Sample covariance (unbiased, `1/(T-1)`) and EWMA (RiskMetrics, default decay
0.94) are also selectable. EWMA weights recent observations more heavily, which matters when volatility
regimes shift.

**Data quality checks.** The module reports rather than silently absorbs: missing prices, stale
(unchanged) prices that understate volatility, zero or negative prices, single-period moves above 25%
that suggest unadjusted splits, and insufficient observations relative to the number of assets.

**Short history is called out explicitly.** Integration testing produced a Sharpe ratio of **4.09** from
60 observations, falling to **0.475** from 3,000 observations of the same underlying process. A Sharpe of
4 does not exist in real markets; it was pure estimation error. The binding constraint on expected
returns is calendar time, not observation count — sampling one year more finely does not improve the mean
estimate at all. Any history under three years now triggers a warning that quantifies the standard error
and states plainly that a resulting Sharpe above roughly 1.5 should be read as an artefact.

## 3f. Position limits (`l <= w <= u`)

Textbook mean-variance optimization routinely answers "put everything in one asset". On the test data
in this repository, the unconstrained maximum-Sharpe portfolio allocates **100% to a single holding**.
No real mandate permits that, so the solver was generalized from `w >= 0` to full two-sided bounds.

**The solver.** The active-set method now tracks three states per variable — free, at lower bound, at
upper bound — and the KKT conditions become, writing the gradient residual `r_i = (Sx)_i - c_i - gamma*a_i`:

```
free (l_i < x_i < u_i):   r_i = 0
at lower bound:           r_i >= 0
at upper bound:           r_i <= 0
```

Pinned variables are substituted into the right-hand side and the reduced system is solved on the free
set only. Every iteration either pins a variable violating a bound or releases one whose multiplier has
the wrong sign, so the method still terminates in finitely many steps at an exact KKT point.

**Infeasibility is detected before solving, not discovered afterwards.** Minimums summing above 100%, or
maximums summing below it, make a fully invested portfolio impossible. Both are checked up front and
refused with an explanation naming the problem. A solver that returns a confident-looking portfolio for an
impossible mandate is worse than one that returns nothing.

**Maximum Sharpe needs a different method under limits.** The Cornuejols-Tutuncu transformation relies on
the Sharpe ratio being scale-invariant in `w`. A position limit `w_i <= u_i` becomes `y_i <= u_i * 1'y`
after the transformation — a general linear inequality, not a bound on `y` — so the trick no longer
applies. With limits the optimum is found by golden-section search over risk tolerance along the
constrained frontier. The efficient frontier is concave, so the Sharpe ratio (the slope of the line from
`(0, rf)` to a frontier point) is unimodal along it, which is precisely what golden-section search
requires. It converges to tolerance rather than closing in one step, but it is not a heuristic, and the
test suite confirms it reproduces the closed-form answer to 1e-6 whenever bounds do not bind.

**Measured effect** on the repository's test assumptions:

| Mandate | Sharpe | Assets held | Largest position |
|---|---|---|---|
| Unconstrained | 0.309 | 1 of 5 | 100% |
| Max 30% | 0.106 | 4 of 5 | 30% |
| Max 25%, min 5% | 0.079 | 5 of 5 | 25% |

Sharpe falls as constraints tighten, exactly as theory demands — the unconstrained optimum is by
definition unbeatable. That the constrained number is lower is not a defect. The unconstrained portfolio
concentrates its entire value in one estimate of one expected return, and the sensitivity panel shows how
little that estimate can be trusted.

## 3g. Group constraints (`Gx <= h`)

Per-asset limits cannot express "at most 60% in equities" when equities span several holdings. That is a
constraint on a *sum*, so the solver was extended from box constraints to general linear inequalities.

**Specification** is by intent, not by matrix:

```js
groups: [{ name: "Equity", assets: [0, 2, 3], max: 0.60, min: 0.20 }]
```

A maximum becomes the row `sum(x_i for i in group) <= max`; a minimum becomes `-sum(x_i) <= -min`, since
every row must face the same direction.

**The solver** now maintains a working set `W` of group rows currently held as equalities, alongside the
existing per-variable active sets. With multiplier `gamma` on the budget constraint and `lambda >= 0` on
the active rows, the gradient residual becomes

```
r = Sx - c - gamma*a + G_W' lambda
```

and optimality requires `r = 0` on free variables, `r >= 0` at a lower bound, `r <= 0` at an upper bound,
`lambda >= 0` on every row in `W`, and complementary slackness on each. The KKT system grows to
`(free + 1 + |W|)` equations. Each iteration adds the most violated group row to `W`, or releases the row
whose multiplier has gone negative, or pins/frees a variable at its bound. Redundant active rows can make
the system singular; those are dropped and re-verified against the final answer, so a constraint is never
quietly abandoned.

**Infeasibility is diagnosed in the user's own terms** before any solving happens:

- a group capped below the sum of its members' own minimum allocations
- a group floor above what its members' position limits can supply
- a group minimum above its own maximum
- disjoint group caps that cover every asset yet sum below 100%

Each produces a sentence naming the group and the arithmetic, rather than a solver error code. In the randomized stress test of 400 generated mandates: **246 solved exactly, 111 refused as genuinely
impossible, and 43 returned an approximate but fully feasible answer** (see below). No mandate was
answered with a portfolio that violated it.

**Measured effect** on the repository's estimated inputs, where the unconstrained optimum is 100% in a
single holding:

| Mandate | Sharpe | Resulting portfolio |
|---|---|---|
| Unconstrained | 0.309 | Intl 100% |
| Equities <= 40% | 0.093 | Bonds 31%, Intl 40%, Gold 29% |
| Defensive >= 35% | 0.215 | Bonds 14%, Intl 65%, Gold 21% |
| Balanced mandate | 0.102 | spread across all five |

Maximum Sharpe under group constraints uses the same golden-section search as the bounded case, for the
same reason: the Cornuejols-Tutuncu transformation turns `w_i <= u_i` into `y_i <= u_i * 1'y`, a general
inequality in `y`, so the closed form no longer applies.

### When the exact method does not converge

The active-set method is exact, but on degenerate problems — where redundant active constraints repeatedly
produce singular KKT systems — it can fail to settle. Two earlier behaviours here were defects:

1. **Rows dropped for singularity were blocked permanently.** The iteration then terminated at a point
   violating them and reported *"These limits cannot be satisfied"* for mandates that were satisfiable by
   hand. Fixing this alone moved 51 of 400 random mandates from wrongly-refused to solved.
2. **The first fix replaced that with a bare feasibility point**, which satisfies every constraint and
   optimizes nothing. Safe, but a weak answer presented as an answer.

The current fallback runs projected gradient descent from the verified feasible point, projecting back
onto the constraint set after each step, which converges to the constrained minimum for a convex
objective. Measured against the exact solver on 120 problems where both succeed:

| | Objective gap vs exact |
|---|---|
| Median | −2.2 × 10⁻¹⁰ % (machine precision) |
| 90th percentile | 0.075% |
| Worst | 29.7% |

It matches the exact optimum outright about half the time and is within a fraction of a percent for 90% of
problems, but the worst case is genuinely poor. It is a safety net, not a substitute, and results produced
this way are flagged `approximate: true` rather than presented as optimal. It also never beats the exact
solver, which is an independent check that the exact solver is finding the true optimum.

## 3h. Out-of-sample testing (`backtest.js`)

Everything above is in-sample: the optimizer is scored on the same data it was fitted to. That is not
evidence it works. This module re-estimates and re-optimizes at each rebalance using only information
available at that moment, then measures what the portfolio actually earned afterwards.

**Walk-forward design.** At each date `t` the estimation window is strictly `[t - lookback, t - 1]`; the
return earned is at `t`. Between rebalances weights drift with the assets rather than being silently
reset. Log returns are used for estimation, simple returns for profit and loss, where they are exactly
correct. Transaction costs are charged on every unit of weight traded.

**Look-ahead bias is tested for, not asserted away.** It is the failure that quietly invalidates most
backtests, so there is a direct test: inject a threefold price spike at period 700 and confirm that every
return and every weight before the spike enters the estimation window is bit-identical to the untampered
run — while confirming later results *do* change, so the test cannot pass vacuously.

**Metrics** are geometric, not arithmetic. Returns of +10% then −10% leave you at 0.99, and the code is
tested on exactly that case. Also reported: annualized volatility, Sharpe, Sortino, peak-to-trough maximum
drawdown, win rate, average turnover, and total costs paid.

### What the backtest actually showed, and the correction

The first comparison run produced a striking table: equal weighting beat every optimized strategy, and
maximum Sharpe posted a *negative* out-of-sample Sharpe ratio with 18.7% turnover and a 42% drawdown.

| Strategy | Return | Vol | Sharpe | Max DD | Turnover |
|---|---|---|---|---|---|
| Equal weight (1/N) | 6.0% | 13.1% | 0.308 | −27.8% | 2.1% |
| Min variance, shrunk | 3.4% | 8.9% | 0.153 | −25.4% | 2.1% |
| Max Sharpe, shrunk | 0.8% | 15.6% | −0.075 | −42.1% | 18.7% |

That result appears to reproduce a well-known finding in the literature — DeMiguel, Garlappi and Uppal
(2009), that naive diversification is hard to beat. It was tempting to report it as such.

**It was one draw.** Re-running the same comparison across 20 independently generated universes:

| Assets | Lookback | 1/N | Min variance | Max Sharpe | 1/N best in |
|---|---|---|---|---|---|
| 5 | 250 | 0.369 | 0.301 | 0.217 | 4 of 20 |
| 5 | 500 | 0.255 | 0.247 | 0.287 | 2 of 20 |
| 5 | 1000 | 0.202 | 0.290 | 0.412 | 3 of 20 |
| 10 | 500 | 0.496 | 0.529 | 0.360 | 7 of 20 |
| 10 | 1000 | 0.586 | 0.735 | 0.605 | 5 of 20 |

(median out-of-sample Sharpe)

1/N is best in a minority of universes, and the optimizer overtakes it as the estimation window
lengthens — exactly what the estimation-error explanation predicts. Across universes the maximum-Sharpe
strategy ranged from **−0.589 to 1.491**. A single backtest number from that distribution carries almost
no information. Both the multi-universe comparison and the lookback effect are now assertions in the test
suite, so the claim is checked rather than asserted.

**One caveat larger than the result.** This data is generated from a stationary process with constant
`mu` and `Sigma`. Real markets are not stationary, and non-stationarity punishes estimation far harder
than anything measured here. This test is therefore *generous* to the optimizer. It does not establish
that mean-variance optimization beats equal weighting on real data — the published evidence on real data
is considerably less kind — and no backtest in this repository should be read as a claim about the future.

## 3i. Stress testing the assumptions (`scenarios.js`)

Section 3h ended with a caveat: the backtest data was stationary by construction, which makes it a
generous test. Leaving that as a caveat would be leaving the main weakness unexamined, so this module
generates markets that break the assumptions directly.

Each generator is verified to actually exhibit the property it is named for, rather than merely being
named after it:

| Scenario | What it breaks | Verified behaviour |
|---|---|---|
| Regime switching | constant `mu` and `Sigma` | crises 9.6% of periods; volatility 1.13% → 3.42%; correlation 0.24 → 0.85 |
| Volatility clustering | independent draws | squared-return autocorrelation 0.161 vs 0.002 stationary |
| Structural break | constant `mu` | asset ranking fully reversed at the midpoint |
| Correlation breakdown | constant `Sigma` | correlation 0.20 → 0.89 during drawdowns |

**Median out-of-sample Sharpe, 12 trials per scenario** (descriptive only — see below):

| Scenario | 1/N | Min variance | Max Sharpe | Optimizer edge |
|---|---|---|---|---|
| Stationary | 0.331 | 0.530 | 0.323 | +0.199 |
| Regime switching | 0.473 | 0.477 | 0.510 | +0.037 |
| Volatility clustering | 0.506 | 0.691 | 0.469 | +0.185 |
| Structural break | 0.461 | 0.550 | 0.458 | +0.089 |
| Correlation breakdown | 0.363 | 0.491 | 0.353 | +0.128 |

The optimizer's edge over equal weighting is largest on stationary data (+0.199) and collapses under
regime switching (+0.037). That is the caveat from section 3h, now measured rather than asserted.

### The finding that came out of a failed hypothesis

The expected result was that a structural break — reversing every asset's expected return — would punish
the estimate-dependent strategy hardest. It did not. Maximum Sharpe scored 0.458 after a break versus
0.323 on stationary data: no damage at all.

Investigating why produced something more useful than the hypothesis:

```
spread in true annual returns:   8.2 percentage points
standard error of the estimate: 13.5 percentage points
signal-to-noise:                0.61
```

**The noise in a 500-day mean estimate is larger than the entire spread being estimated.** The break
caused no measurable damage because the estimator was never tracking expected returns in the first
place — it was tracking noise, and reversing the truth does not disturb noise. Amplifying the signal
twelvefold makes the break bite exactly as predicted (2.008 Sharpe of damage), which confirms the
mechanism rather than excusing it.

This is Merton's (1980) point, measured inside this system: the precision of a mean estimate depends on
the *calendar length* of the sample, not the number of observations in it. Sampling a year more finely
does not improve it. Covariance behaves completely differently — its relative error falls as
`sqrt(2/T)`, about 6.3% at the same 500 observations.

That asymmetry — covariance estimable to a few percent, means not estimable at all — is the reason
minimum variance, which ignores expected returns entirely, leads maximum Sharpe across these tests.

### A methodological correction, the third of its kind

The 12-trial table above shows maximum Sharpe *ahead* of minimum variance under regime switching
(0.510 vs 0.477). An earlier version of the test suite asserted a ranking from that table. Re-running
with 60 **paired** trials — both strategies on identical generated data, which removes the
between-universe variance — reverses it:

| Scenario | Min variance − Max Sharpe | t | Min variance won |
|---|---|---|---|
| Stationary | +0.0947 | 2.84 | 38 of 60 |
| Regime switching | +0.0435 | 1.61 | 34 of 60 |

The regime-switching result was noise. The stationary result is real. The comparison table in the test
suite is now explicitly marked descriptive, and every ranking claim carries a paired t-statistic.

This is the third time in this project that an apparent ordering failed to survive more trials: first the
1/N result in section 3h, then the position-limit turnover finding in the bug log, now this. The pattern
is consistent enough to be the main methodological lesson of the repository — **a number produced by
correct code, measured correctly, can still support a false conclusion.**

## 3j. The dashboard

Two views over the same model.

**Efficient frontier** — assets, correlations with live positive-semi-definite validation, shrinkage
controls, position and group limits, the constrained and unconstrained frontiers, optimal portfolios,
and the sensitivity panel.

**Out-of-sample test** — available once a price CSV is loaded. Runs the walk-forward backtest from
section 3h with the position and group limits applied at every rebalance, and charts growth of 1 against
equal weighting. The comparison table is followed by a panel headed *"Read this table with suspicion"*,
which states the −0.59 to +1.49 Sharpe range that the same strategies produced across simulated
histories. Presenting the number without that range would misrepresent what a single backtest can tell
anyone.

Everything runs in the browser. `portfolio-core.js`, `data-pipeline.js`, and `backtest.js` are the same
files the test suites exercise in Node — the dashboard inlines them rather than reimplementing anything.

**A real defect this claim used to be wrong about.** Until 2026-09, that last sentence was aspirational,
not true: `portfolio-optimizer.jsx` inlined `portfolio-core.js` but never actually inlined
`data-pipeline.js` or `backtest.js` — CSV import and the entire out-of-sample tab threw
`estimateFromCSV is not defined` the moment either was used. This went undetected because
`bundle-runtime-check.js` never read `portfolio-optimizer.jsx` at all: it independently re-implements the
same three files as its own hand-maintained copy and tests *that*, so its "BUNDLE RUNTIME OK" kept
passing throughout — proving its own duplicate was wired correctly while saying nothing about the real
dashboard file. Found by live-testing the dashboard in an actual browser (simulating a file upload via a
synthetic `DataTransfer`, since setting a file input's value is blocked), not by any existing test. Fixed
by properly inlining both files into `portfolio-optimizer.jsx` (stripping the Node/browser `require`
indirection in `backtest.js`'s copy, which does not apply inside a single already-inlined scope, and
fixing two identifier collisions it caused), and by adding a section to `bundle-runtime-check.js` that
extracts and executes `portfolio-optimizer.jsx`'s actual source via Node's `vm` module, so this
specific class of bug — the file claiming to test something that isn't the real file — cannot recur
silently again.

## 3k. Transaction costs inside the objective

Charging costs after optimizing lets the optimizer produce a portfolio it cannot afford to reach. The
backtest measured maximum Sharpe turning over 14.8% of the portfolio per rebalance against 2.3% for
minimum variance — at any realistic cost, paid out of returns.

**Formulation.** The penalty `cost * sum|w - wPrev|` is convex but not smooth, so the trade splits into
non-negative buy and sell legs:

```
w = wPrev + b - s,   b >= 0,  s >= 0,   sum|w - wPrev| = 1'b + 1's
```

giving a quadratic program in `x = [b; s]` with `2n` variables. With `A = [I, -I]`, the quadratic term is
`(1/2) x'(A'SA)x` and the linear term `[A'(S wPrev - tau*mu) + cost*1]'x`. Per-asset bounds and group
rows carry over as inequalities in `(b, s)`.

`A'SA` is singular by construction — rank `n` in `2n` dimensions — which is harmless at the optimum, where
no asset is both bought and sold, so the active set pins one leg of each pair and the reduced system has
full rank. A small ridge is added so intermediate iterates cannot hit an exactly singular system.

**Does it actually help?** Measured, paired, on identical generated data:

| Cost | Strategy | Net Sharpe change | t | Turnover |
|---|---|---|---|---|
| 10 bps | Min variance | +0.021 | 1.9 | 2.4% → 0.6% |
| 10 bps | Max Sharpe | −0.002 | −0.1 | 15.9% → 13.6% |
| 200 bps | Max Sharpe | +0.412 | 14.1 | 15.9% → 6.0% |

Turnover falls dramatically in every case. The *net return* benefit scales with the cost: unambiguous at
200 basis points, marginal at 10, and absent for maximum Sharpe at 10 where the cost is small next to the
estimation noise already dominating that strategy.

The 10 bps minimum-variance result sits at t ≈ 1.9, just short of conventional significance. The trial
count was not raised until it crossed a threshold — that would be choosing the answer. The test asserts
the direction and prints the t-statistic.

### The solver defect this uncovered

Building this exposed a real limitation in the active-set method. It began each solve from a no-trade
point, which satisfies the bounds and budget but not necessarily the general rows of `G`. When a portfolio
is re-optimized against limits its **current holdings already breach** — a mandate tightens, or the
portfolio drifts out of compliance — five or more rows were violated at the outset and the method had no
route back. Worse, rows dropped for producing a singular KKT system were blocked permanently, so the
iteration terminated at a point violating them and reported *"These limits cannot be satisfied"* for
problems that were plainly satisfiable by hand.

Two fixes:

- **`findFeasiblePoint`** — cyclic projection onto each constraint set in turn (POCS). Every set is closed
  and convex, so the iteration converges to their intersection whenever one exists, and the result is
  verified rather than assumed.
- **Rows are no longer blocked permanently.** They are released for retry with a bounded attempt count,
  and if any abandoned row still binds at termination, the solver checks whether a feasible point exists
  at all before declaring the mandate impossible.

Genuinely impossible mandates are still refused — there is a test confirming the fallback does not mask
real infeasibility.

## 3l. Incorporating an external signal (Black-Litterman, sentiment)

A natural extension once real data and shrinkage exist: fold in a subjective view — most concretely, a
news-sentiment score — without letting it overwrite the carefully-estimated historical mean. Black-
Litterman (1992) is the standard, principled way to do this.

**The formula**, with `mu0` a prior (here, the shrunk historical mean), `P`/`Q`/`Omega` encoding the
views and their uncertainty, and `tau` a conventional scale factor (0.05, Idzorek 2005):

```
posterior_mu = mu0 + tau*Sigma*P' * inv(P*tau*Sigma*P' + Omega) * (Q - P*mu0)
```

**A design choice, made after finding a bug in the alternative.** The textbook formula also derives a
posterior covariance by adding back a term for residual uncertainty in the mean estimate. An earlier
version here did that — and failed a basic consistency check: a view with near-infinite uncertainty is
mathematically equivalent to no view at all, yet the two produced different covariances (0.0225 vs
0.023625 on a test case). Rather than patch that discontinuity, the implementation was simplified:
`blackLitterman` now adjusts **only the mean** and returns the input covariance unchanged. This also
avoids re-introducing uncertainty into a covariance this project already spent section 3e getting right
via Ledoit-Wolf — replacing a data-derived shrinkage intensity with a much less rigorous heuristic would
have been a step backward dressed up as a step forward.

**`sentimentViews(mu0, scores, articleCounts, opts)`** turns a per-asset sentiment score into Black-
Litterman inputs: one absolute view per covered asset, `Q_i = mu0_i + score_i * tiltPerUnit`, with
uncertainty `baseVariance / articleCount` (more articles behind a score, more confidence — the usual
`1/sqrt(n)` shrinkage of a noisy estimate).

### A second, more interesting bug — caught by the experiment, not the unit tests

The first version of `sentimentViews` built the view as `Q_i = score_i * tiltPerUnit` alone, discarding
`mu0` entirely. Every unit test passed, because they checked the formula's internal arithmetic, not
whether the result made directional sense. Running an ordinary integration check — a stock with an 8%
historical prior and a genuinely bullish sentiment score of 0.36 — surfaced it immediately: the posterior
mean moved from **8.00% down to 7.86%**. Bullish news pulling the estimate down.

The mechanism: with `tiltPerUnit` calibrated so a maximal score corresponds to a plausible tilt (about
15%), a score of 0.36 asserts an absolute view of roughly 5.4% — which is a *downgrade* for any asset
whose prior already exceeded that, regardless of the news being positive. Fixed by making the view
additive to the prior (`mu0_i + score_i * tiltPerUnit`), so a positive score always moves the estimate up
from wherever it already was. This is the kind of error that is invisible from inside a unit test suite
that only checks a formula against itself; it only became visible by connecting the output to a number
with real-world meaning.

### Does it actually help? Measured, not assumed

`withLeadingSignal` (in `scenarios.js`) generates a market where a genuine, noisy leading signal exists,
with its **informativeness** (correlation to what actually happens next) as an explicit, controllable
parameter — 0 is pure noise, 1 is a perfect leading indicator. This turns "does sentiment help" from a
question answered by intuition into one answered by a properly-powered paired experiment (`maxsharpe`
vs. `maxsharpeSentiment`, same generated market, net of trading costs):

| Informativeness | Sharpe change | t |
|---|---|---|
| 0.00 (pure noise) | −0.0119 | −2.88 |
| 0.15 (plausible real-world guess) | −0.0095 | −2.31 |
| 0.90 (near-perfect signal) | +0.0108 | 3.15 |

**A signal has to be genuinely good before it is worth using.** Pure noise measurably *costs* Sharpe (the
tilt still churns the portfolio at zero expected benefit). A plausible estimate for how informative a real
news-sentiment score actually is — well below the 0.61 signal-to-noise ratio section 3i measured for
returns estimated from clean historical *prices*, since sentiment is a noisier, indirect proxy for the
same latent quantity — lands at 0.15, which is still net negative. Only near-perfect informativeness
(0.90) is clearly worthwhile.

**This number was underpowered before it was corrected.** An initial run at N=40–60 trials gave
inconsistent signs at informativeness 0 depending on which seed range was drawn (t ranging from −2.10 to
+0.18). Rather than pick the favorable draw, trial counts were increased until the estimate stabilized
(N=150), which is what produced the table above. Interior points (0.3, 0.6) are reported at lower power
for shape only and are not individually asserted to be significant.

### `fetch-news-sentiment.js`

A CLI, structured exactly like `fetch-market-data.js` for the same reason: the response-parsing logic
(`parseNewsSentimentJSON`, `aggregateSentiment`) is pure and unit-tested against recorded response
shapes; only the HTTPS call itself is untested here, for lack of network access to a news provider. Uses
Alpha Vantage's `NEWS_SENTIMENT` endpoint rather than a bag-of-words scorer built from scratch — a
homemade sentiment model would be low quality and would dress up noise as a measurement; a real scoring
service, however imperfect, is a more honest foundation. Aggregation is **relevance-weighted**: an
article barely mentioning a ticker counts for less than one squarely about it.

## 3m. Validation against real market data

Sections 3h and 3i validate the optimizer entirely against generated data — either a stationary random
process or the deliberately non-stationary scenarios in `scenarios.js`. Both are honest about what that
does and does not establish (3h: "no backtest in this repository should be read as a claim about the
future"). This section is the first validation against actual historical prices, made possible once
`fetch-market-data.js` was verified against a live provider (section "Verification status" above) rather
than only recorded fixtures.

**Data.** Five ETFs proxying the same asset classes as the dashboard's illustrative defaults — SPY (US
stocks), AGG (US bonds), EFA (international stocks), VNQ (REITs), GLD (gold) — fetched via
`fetch-market-data.js --source=tiingo --adjusted`, common history from **2004-11-18 to 2026-09-11 (~21.8
years, 5,487 trading days)**, bounded by GLD's 2004 inception. This span includes the 2008 financial
crisis and the 2020 crash, which the shorter window used in an earlier draft of this test did not.

**Method.** `backtest.js`'s existing walk-forward machinery (`compareStrategies`), unchanged — monthly
rebalancing, 10bps transaction costs, 2% risk-free rate — run at three lookback windows (250, 500, 1000
days) rather than one, specifically to check whether any result was an artifact of a single hyperparameter
choice, since real history cannot be independently replicated the way `scenarios.js` can.

| Lookback | 1/N Sharpe | Min-variance Sharpe | Max Sharpe | Max Sharpe, capped 30% |
|---|---|---|---|---|
| 250 days | 0.465 | 0.271 | 0.504 | **0.625** |
| 500 days | 0.419 | 0.273 | 0.414 | **0.512** |
| 1000 days | 0.593 | 0.391 | 0.754 | **0.774** |

**Capped maximum Sharpe had the highest risk-adjusted return at every lookback tested**, and its margin
over 1/N widened as the lookback lengthened (longer estimation windows give more stable inputs). **Minimum
variance was the weakest strategy by Sharpe at every lookback** — the opposite ranking from section 3i's
synthetic-scenario finding that minimum variance leads maximum Sharpe on average. The most plausible
explanation is not a contradiction of that finding but a difference in what the two tests measure: over
this specific 22-year window, equities and REITs meaningfully outearned bonds and gold, a slow-moving
directional signal that minimum variance ignores by construction, whereas the synthetic scenarios have no
such persistent real-world drift built in. That explanation is plausible, not verified — it would need its
own test to confirm, and is stated here as an open question, not a finding.

### A result that reversed itself once more data was used — the fourth time this has happened

An earlier pass at this test used only 2010–2026 data (500-day lookback) — a real-data fetch, but one that
happened to start after the 2008 crisis and used a single lookback choice. It showed 1/N winning
(Sharpe 0.601) over capped maximum Sharpe (0.564), the opposite of the table above. Extending to the full
available common history and testing multiple lookbacks reversed the ranking. This is now the fourth time
in this project that an initial ordering did not survive additional scrutiny — after the 1/N-beats-everything
result in section 3h, the position-limit turnover finding in the bug log, and the regime-switching ranking
in section 3i. Four occurrences of the same failure mode across a project this size is itself the more
important data point: a single backtest window, however real, is not evidence on its own.

**What this does and does not establish.** This is one realized historical path for one specific five-asset
universe — not the 20–60 independently generated, paired-and-t-tested universes section 3i uses for
synthetic data. Real history cannot be resampled that way. Treat this as one genuine, real data point that
complicates a naive "just use 1/N" takeaway, not as proof that constrained mean-variance optimization beats
naive diversification in general. Consistent with section 3h: no backtest in this repository, real or
generated, should be read as a claim about the future.

Reproduce with `node fetch-market-data.js --source=tiingo --tickers=SPY,AGG,EFA,VNQ,GLD --key=YOUR_TOKEN
--adjusted --out=prices_real_full.csv` followed by `node run-real-backtest.js prices_real_full.csv`.

## Reproducing the numbers

Every quantitative claim in this file is regenerated by:

```bash
node reproduce-findings.js
```

It recomputes all 45 documented figures and flags any that no longer match, so documentation drift becomes
a test failure rather than something a reader discovers later. It has already earned its place three
times: once by catching that the solver fix had changed the group-constraint outcomes (162 refusals became
111 because problems previously refused in error are now solved), once by catching a bug in itself — a
random draw moved inside a `map` consumed one number per asset instead of one per trial, desynchronising
the sequence and producing figures that looked plausible and were wrong — and once by confirming the
sentiment-view semantics fix in section 3l changed the measured numbers (the harmful-noise effect shrank
from an initial, less-powered estimate to −0.0119, and the beneficial-signal effect grew to +0.0108), so
the documented figures could be updated to match rather than left stale.

### A gap the dashboard itself had

Every non-convergent result described above (the 43-of-400 fallback, the turnover-cost augmented QP,
the golden-section search) returned a fully feasible portfolio — but nothing downstream ever checked
whether it was optimal or merely feasible. `minVarianceLongOnly`, `efficientLongOnly`,
`efficientWithTurnoverCost`, and the constrained path of `maxSharpeLongOnly` all discarded `solveQP`'s
`converged` flag at the point of return. A user could receive an approximate answer with zero indication
it differed from an exact one.

Fixed by tagging `.approximate` directly onto the returned weight array at every wrapper (arrays are
objects in JavaScript, so this is free and breaks no existing caller that treats the result as a plain
array), rolling it up across `longOnlyFrontier`, and surfacing it in the dashboard as a visible
**"approximate"** badge on the affected result card, with an explanation of what that means and why.
One naming collision came with it: the golden-section search's own `converged: true` (the *search*
always converges in a fixed number of steps) had been silently absorbing whichever candidate's QP-level
`approximate` flag it carried — those are different claims and are no longer conflated.

A test forces this path deterministically (a known seed/trial that hits the fallback) rather than
hoping a random sweep happens to trigger it, and confirms the approximate result is nonetheless fully
feasible.

## 4. Code walkthrough (`portfolio-core.js`)

| Function | Role |
|---|---|
| `solveLinear` / `invertMatrix` | Gaussian elimination with partial pivoting. Throws on singular input rather than returning garbage. |
| `symmetricEigen` | Jacobi rotation method for symmetric matrices. Used for PSD validation and matrix repair. |
| `buildCovariance` | `Sigma_ij = sigma_i * sigma_j * rho_ij`. |
| `validateCorrelation` | Checks symmetry, unit diagonal, range, and positive semi-definiteness. Returns the minimum eigenvalue and a plain-language list of problems. |
| `repairCorrelation` | Eigenvalue clipping plus rescaling to unit diagonal — the nearest valid correlation matrix. |
| `shrinkCovariance` | Constant-correlation shrinkage (section 3c). |
| `bayesSteinMu` | Jorion (1986) expected-return shrinkage. |
| `analyticFrontier` | Closed-form scalars A, B, C, D plus GMV and tangency portfolios (section 2). |
| **`solveQP`** | The bounded active-set QP solver (sections 3 and 3f). Every constrained portfolio routes through it. |
| `refineByProjectedGradient` | Fallback when the active-set method does not converge: descends toward the optimum from a feasible point, projecting back after each step. |
| `blackLitterman` / `sentimentViews` | Blends a subjective view (e.g. sentiment) with the prior mean (section 3l). Adjusts only the mean; the covariance is left as estimated. |
| `findFeasiblePoint` | Phase 1: cyclic projection onto the constraint sets, so the solver can start inside the feasible region even when the current holdings are outside it. |
| `efficientWithTurnoverCost` / `maxSharpeWithTurnoverCost` | Optimization with the transaction cost inside the objective (section 3k). |
| `normalizeBounds` | Accepts per-asset arrays or `{minWeight, maxWeight}` shorthand plus `groups`, and reports whether the constraints are trivial so the exact maximum-Sharpe path can still be used. |
| `buildGroupConstraints` | Turns group specifications into the inequality rows `Gx <= h`, with plain-language labels. |
| `checkGroupFeasibility` | Diagnoses impossible mandates before solving, naming the group and the arithmetic. |
| `checkKKT` | Independently verifies the two-sided KKT conditions at a solution. Used by the test suite, not by the app — the solver does not grade its own work. |
| `minVarianceLongOnly` / `efficientLongOnly` / `maxSharpeLongOnly` | Thin wrappers mapping each portfolio problem onto `solveQP`. All accept optional bounds; `maxSharpeLongOnly` reports which method it used. |
| `longOnlyFrontier` | Sweeps risk tolerance geometrically, deduplicates, and filters to genuinely efficient points (return must strictly increase with volatility). |
| `sensitivityAnalysis` | Section 3d. Uses a seeded PRNG so results are reproducible run to run. |

**Numerical robustness.** `solveQP` degrades gracefully: a degenerate free set (perfectly collinear assets)
shrinks the free set and retries rather than crashing; variables with a zero coefficient in the equality
constraint are excluded since they cannot help satisfy it; iteration is capped and the `converged` flag is
returned to the caller rather than silently ignored.

**Why no external math library.** All linear algebra is hand-implemented. This keeps the core dependency-free
and testable in plain Node, and avoids relying on library behaviour for edge cases (singular matrices,
eigenvalues of near-degenerate covariance) that matter here.

## 5. Key concepts demonstrated

- Mean-variance optimization via Lagrangian methods and matrix inversion
- Efficient frontier, global minimum-variance portfolio, tangency portfolio, Capital Allocation Line, Sharpe ratio
- Constrained vs. unconstrained optimization, and why each added constraint class needs a different solution method
- Estimation error as a first-class concern rather than a footnote
- Statistical estimation from time series: sample versus shrinkage estimators, and why the choice matters most when data is scarce
- Walk-forward validation, look-ahead bias, and why a single backtest result is not evidence
- Paired statistical testing, and the difference between a measured number and a supported conclusion
- Bayesian updating of a prior estimate (Black-Litterman), and why an external signal is only worth using once its informativeness clears a measurable threshold — not by default
- Reactive computation architecture (`useMemo`) for a live, editable financial model

## 6. Assumptions and limitations

- **Estimation risk is mitigated, not eliminated.** The frontier is highly sensitive to the input `μ` vector — small changes in assumed expected returns can produce large swings in optimal weights (the classical Markowitz "error maximization" critique, Michaud 1989). Shrinkage (section 3c) and the sensitivity panel (section 3d) address this, but no amount of post-processing rescues genuinely bad return forecasts.
- **Static covariance.** `Σ` is treated as constant. Real markets have time-varying volatility and correlation (correlations especially tend to rise sharply in crises, reducing diversification exactly when it's most needed).
- **Two moments only.** Mean-variance optimization ignores skewness and kurtosis (tail risk). It is a poor model for assets with strongly asymmetric return distributions (e.g., options, distressed debt).
- **Single period.** No taxes or rebalancing-schedule constraints beyond the fixed-interval walk-forward test; the model assumes one investment decision at a time, not a dynamic multi-period policy. (Transaction costs themselves ARE modelled — section 3k — this refers to taxes and scheduling only.)
- **Shrinkage intensity is data-optimal only when a price history is loaded.** With manually entered summary statistics there is no return series from which to derive `delta`, so the manual slider remains a judgement call.
- **Expected returns may not be estimable at all.** Section 3i measures a signal-to-noise ratio of 0.61 for expected returns at realistic parameters. Any result this tool produces that depends on ranking assets by expected return should be treated with corresponding suspicion; minimum variance, which ignores them, rests on far firmer ground.
- **Historical estimates are not forecasts.** Estimating `mu` from past returns assumes the future resembles the past. It frequently does not, and expected returns are the input the optimizer is most sensitive to. The short-history warning and sensitivity panel exist to keep this visible rather than to solve it.
- **An external signal (sentiment or otherwise) is net-negative unless it is genuinely informative.** Section 3l measured this directly rather than assuming it either way: a pure-noise signal costs about 0.012 Sharpe on average (still churns the portfolio for no benefit); a plausible estimate of real news-sentiment informativeness (0.15) is also net negative; only a signal with informativeness above roughly 0.5–0.6 clears the bar. Nothing in this project establishes that any real feed reaches that bar — that would need its own validation against the feed's actual historical accuracy.
- **Provider data is not validated for corporate actions.** Unadjusted splits and dividends are flagged heuristically (moves above 25%) but not corrected. Prefer adjusted close prices where the provider offers them.
- **No cardinality constraints.** Position limits, minimum allocations, group constraints, and transaction costs are all supported. "Hold at most 10 names" is a combinatorial constraint, not a convex one, and is not.
- **Illustrative default data.** The five default assets (US Stocks, US Bonds, Intl Stocks, REITs, Gold) use approximate long-run historical figures for demonstration, not live or sourced market data.

## 7. Tech stack

- **Core, data pipeline, backtester, and scenario generators:** plain JavaScript, zero dependencies — the same files run in Node and the browser
- **UI:** React (hooks, `useMemo` for reactive recomputation)
- **Charts:** [Recharts](https://recharts.org/)
- **Styling:** Tailwind CSS
- **Tests:** plain Node, no test framework required

## 8. Possible extensions

- Cardinality constraints, which need mixed-integer methods rather than a convex QP
- A regime-aware estimator that adapts to the scenarios in `scenarios.js`, rather than only being measured against them
- Validating a real news-sentiment feed's actual informativeness against its own historical accuracy, rather than assuming a value — section 3l measured that the break-even point is roughly 0.5–0.6, but nothing here establishes where any real feed actually falls
- Capital allocation line with a user-selected target risk level
- Factor-model covariance estimation as an alternative to constant-correlation shrinkage
- Automatic corporate-action adjustment rather than heuristic flagging
- Wiring `blackLitterman`/`sentimentViews` into the dashboard UI directly (currently API-level only, exercised through the backtest and test suites, not yet exposed as an interactive control)

## 9. Symbol glossary

| Symbol | Meaning |
|---|---|
| `n` | Number of assets |
| `w` | n×1 vector of portfolio weights |
| `μ` | n×1 vector of expected returns |
| `Σ` | n×n covariance matrix |
| `r_f` | Risk-free rate |
| `A, B, C, D` | Scalars from `1ᵀΣ⁻¹1`, `1ᵀΣ⁻¹μ`, `μᵀΣ⁻¹μ`, and `AC − B²`, used in the closed-form frontier |
| `λ, γ` | Lagrange multipliers on the return and budget constraints |
