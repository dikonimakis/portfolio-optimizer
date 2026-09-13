# Honest Assessment: What This Is Worth

Replaces the earlier roadmap, which was written before the evidence came in and no longer reflects
what the project has established about itself.

---

## The short answer on commercialisation

**As a product: weak. As a credential: strong.**

The uncomfortable part is that this project's own findings are the main argument against selling it.

## Why the findings undercut the product

Three measurements from the test suite, all reproducible by running it:

1. **Expected returns are not estimable at realistic parameters.** Signal-to-noise of 0.61 — the standard
   error of a 500-day mean estimate (13.5 percentage points annualized) exceeds the entire spread being
   estimated (8.2 points). The maximum-Sharpe portfolio, the headline output of any optimizer, rests on
   inputs that cannot be measured.

2. **The optimizer's edge over equal weighting collapses under realistic conditions.** +0.199 Sharpe on
   stationary data, +0.037 under regime switching. Real markets switch regimes.

3. **Three separate apparent findings failed replication.** Each looked convincing, each came from
   correct code and correct measurement, each was wrong.

A product whose own documentation says "the number this produces may not mean much" is honest. It is not
easy to sell.

## The market reality

| Obstacle | Detail |
|---|---|
| The maths is commoditized | Markowitz is from 1952. PyPortfolioOpt and Riskfolio-Lib are free, mature, and more featured. |
| Institutional buyers are served | Asset managers use Aladdin, Axioma, Barra, FactSet. They are not buying a JavaScript optimizer. |
| Retail is regulated | Anything resembling personalised allocation advice triggers licensing obligations that dwarf the build cost. |
| No data moat | The inputs are public prices. Anyone can obtain the same ones. |
| No distribution | The hardest part of fintech is not the model. It is reaching people who will pay. |

## What genuinely has value here

**The validation layer, not the optimizer.** Most tools show a number. This one shows the number plus how
much to trust it: sensitivity bands, data-quality warnings, short-history warnings that quantify the
standard error, refusal to answer impossible mandates, and stress tests against non-stationary markets.

That is unusual. Whether it is a *business* is a different question — it is a feature, and features are
easier to copy than to sell. The most plausible commercial shape would be validation tooling that plugs
into existing optimizers rather than a competing optimizer. That is speculation, not a plan.

## What it is unambiguously good for

A portfolio piece that demonstrates, with evidence rather than assertion:

- Convex optimization implemented from first principles — active-set QP with bounds and general linear
  inequalities, verified against KKT conditions on 800 randomized problems
- Statistical estimation, including a correct Ledoit-Wolf derivation
- Walk-forward validation with an explicit look-ahead bias test
- Paired statistical testing with t-statistics
- **The willingness to publish findings that undermine the author's own tool**

The last item is the rarest and, for a technical hire, the most informative. Anyone can build something
that appears to work. The full bug log in the README — four code defects and three failed replications,
each explained rather than quietly fixed — is stronger evidence of competence than a clean history would be.

## Recommended path

1. **Publish it as a portfolio piece.** It is ready.
2. **Do not pursue commercialisation on the current evidence.** Not because the engineering is weak, but
   because the project has honestly established that the problem it solves is harder than the solution
   admits.
3. **If commercialisation is still wanted, change the target.** The defensible claim is not "better
   portfolios" — the evidence does not support it. It is "know how much to trust your portfolio numbers."
   That is a different product, aimed at people who already run optimizers.

## What remains technically

- Cardinality constraints (mixed-integer, not convex — a genuinely different solver)
- A regime-aware estimator that adapts to the scenarios rather than only being measured against them
- Backtest validation of actual strategy performance on real historical price data, as opposed to only
  the generated stationary/non-stationary scenarios in `scenarios.js`

Two items originally listed here are done and no longer apply: transaction costs are now inside the
objective (README section 3k, not merely charged after the fact), and both `fetch-market-data.js` and
`fetch-news-sentiment.js` have been run against live APIs rather than only recorded fixtures (2026-09) —
see README's "Verification status" section for what that testing found.

None of the remaining items change the assessment above.
