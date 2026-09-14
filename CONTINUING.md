# Continuing This Project

The original `CONTINUING.md` from the prior claude.ai conversation was not included when this
project was handed off to Claude Code — it never made it into the zip. This file replaces it,
written from inside the Claude Code session that picked the project up. It does not claim to
know why the original move from claude.ai happened; it only documents what happened once the
project arrived here, and what a future session needs to know to continue.

## What this session actually did

Starting point: a working, thoroughly-tested project (494 claimed assertions, later found to
be a stale count) that had never been run against a live API, never been run as an actual
browser app, and had no version control.

In order:

1. **Installed the missing toolchain.** Neither Node.js nor git existed on this machine.
   Both installed via `winget` (needed the user's UAC approval each time).
2. **Ran `npm run verify` for the first time in this environment** and found a real,
   environment-dependent test bug: date generation using local-time `Date` construction
   piped through `.toISOString()`, which silently drops a row across a UK/BST daylight-saving
   transition. Fixed with `Date.UTC`. See README's bug log, entries above #5.
3. **Verified `fetch-market-data.js` and `fetch-news-sentiment.js` against real APIs for the
   first time.** Stooq turned out to be blocked by an anti-bot JavaScript challenge. Alpha
   Vantage's free tier had tightened since the code was written (`outputsize=full` and
   adjusted closes are now premium-only; `NEWS_SENTIMENT`'s `limit` parameter appears to be
   ignored). All documented in the file header comments and README's "Verification status."
4. **Added Tiingo as a third data source** — both equities (`--source=tiingo`) and crypto
   (`--source=tiingocrypto`) — after Alpha Vantage's free tier proved too thin for a real
   backtest. Found and fixed several more live-API-only gotchas along the way (Tiingo's daily
   endpoint silently returning one row without an explicit `startDate`; the crypto endpoint's
   different error shape; a ~5,000-row cap that truncates from the *end*, not the start).
5. **Found the dashboard's CSV import and backtest tab had never actually worked** —
   `portfolio-optimizer.jsx` claimed to inline `data-pipeline.js` and `backtest.js` but never
   did, and `bundle-runtime-check.js` (whose job is exactly to catch this) never caught it
   because it tested its own disconnected copy, not the real file. Fixed both the inlining and
   the verification gap. This is the single most significant defect found this session — a
   documented, tested-sounding feature that had silently never worked.
6. **Found that `inferFrequency` assumed every daily series was a 252-day weekday market**,
   which understated crypto's annualized return/volatility by roughly 30% when fed real BTC/ETH
   data. Fixed with a heuristic distinguishing weekday markets from 24/7 ones.
7. **Ran real walk-forward backtests on real historical data** for the first time — a 5-asset
   broad-class universe (22 years) and a 10-sector-ETF universe (11 years) — documented in
   README section 3m with the same epistemic rigor as the rest of the project (these are single
   historical paths, not replicated statistical claims; the two universes gave meaningfully
   different answers, reported honestly rather than reconciled into one story).
8. **Put the project under git** (it had never been version-controlled) and built a Vite
   harness (`dashboard-preview/`) to actually run `portfolio-optimizer.jsx` in a browser —
   which is how #5 above was discovered in the first place.
9. **Built and published a static deployment** of the dashboard as a Claude Artifact
   (`https://claude.ai/code/artifact/45e30a8a-166e-4a5f-a542-4f6fe058a4bc`), replacing the
   Tailwind Play CDN (explicitly not for production) with a properly compiled static build.
   **Confirmed rendering correctly by the user** after two rounds of blind fixes (this session
   has no authenticated browser access to the artifact platform, so every fix here was made
   without being able to see the result directly — the user's confirmation was the first actual
   look at it).
10. **Wired sentiment/Black-Litterman into the dashboard UI** — a per-asset score input, a
    tilt-per-unit control, and a live readout of the resulting posterior shift. Live-verified: a
    bullish score correctly increased that asset's optimal weight and Sharpe, and left
    minimum-variance's weights (correctly) unchanged.
11. **Shipped three real datasets as one-click dashboard samples** (broad asset classes, 10
    sector ETFs, and 4-asset crypto — all real Tiingo data, not illustrative numbers) so the
    tool is immediately usable with real market data and no API key required from a visitor.
    Also raised the dashboard's asset limit from 8 to 12 to fit the sector sample, and added a
    ticker mapping (US Stocks -> SPY, etc.) so the illustrative category labels resolve to
    something concretely buyable.

## Current verified state

- `npm run verify`: 460 assertions across 8 suites, 0 failures, bundle runtime check OK
  (including a section that executes `portfolio-optimizer.jsx`'s real source, not a copy),
  all 45 documented README claims reproduced and consistent.
- Git: 9 commits on `master`, working tree clean as of the last commit. No remote configured —
  this only exists on this machine right now.
- The dashboard runs and has been live-tested end to end: default illustrative view, all three
  real-data samples, CSV upload, sentiment tilt, and a full walk-forward backtest run — all
  confirmed working in an actual browser, and the deployed Artifact confirmed rendering
  correctly by the user directly (not just verified locally).

## What's still open

- **Cardinality constraints and a regime-aware estimator remain unbuilt.** ASSESSMENT.md's own
  position is not to chase these without a specific reason — they're listed as open, not as a
  backlog to work through by default.
- **No git remote / no CI.** Everything here is local. If this project needs to survive a
  machine loss, or needs a review workflow, that's unaddressed. The user has expressed interest
  in publishing this (portfolio piece / LinkedIn); pushing to a public GitHub repo has been
  offered but not done as of this writing.
- **Commercialization**: ASSESSMENT.md's conclusion stands and nothing this session found
  contradicts it — publish as a portfolio piece, don't pursue it as a product on current
  evidence. If revisited, the defensible angle is "know how much to trust your portfolio
  numbers," not "better portfolios."

## Working discipline established this session (worth keeping)

- Every fetch script claim was verified against the *live* API before being trusted, not
  assumed from documentation or memory — this caught real, otherwise-invisible bugs every
  single time it was done.
- When a change broke `reproduce-findings.js`, the fix was always to find and correct the real
  bug (in the code or in a test fixture), never to loosen the check.
- A single backtest result — even from real data — was never reported as a finding on its own;
  every real-data claim in this README states plainly that it's one historical path, not a
  replicated statistical result.
- When a verification step was genuinely outside this session's reach (confirming the deployed
  Artifact renders, since this session has no authenticated browser access to that platform),
  the honest move was to say so plainly and ask the user directly, rather than guessing or
  quietly asserting it probably worked.
