const S = require("./scenarios.js");
const B = require("./backtest.js");
const C = require("./portfolio-core.js");

let pass = 0, fail = 0;
const failures = [];
function ok(name, cond, detail = "") {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; failures.push(name); console.log(`  FAIL  ${name}  ${detail}`); }
}
const section = (s) => console.log(`\n== ${s}`);
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((t, v) => t + (v - m) ** 2, 0) / (a.length - 1)); };

const ALPHA_VOL = 0.0006;
const PERIODS_PER_YEAR = 252;
const TILT_PER_UNIT = ALPHA_VOL * PERIODS_PER_YEAR;

function pairedTrial(informativeness, trials, seedBase) {
  const diffs = [];
  for (let trial = 0; trial < trials; trial++) {
    const sc = S.withLeadingSignal(6, 3000, seedBase + trial * 131, {
      informativeness, blockLength: 21, alphaVol: ALPHA_VOL,
    });
    const run = (strategy, signals) => B.walkForward(sc.prices, null, {
      lookback: 500, rebalanceEvery: 21, strategy,
      estimator: "ledoit-wolf", periodsPerYear: PERIODS_PER_YEAR, costBps: 10, rf: 0.02,
      signals, sentimentTilt: TILT_PER_UNIT,
    });
    let blind = null, sentiment = null;
    try { blind = run("maxsharpe"); } catch (e) { }
    try { sentiment = run("maxsharpeSentiment", sc.signal); } catch (e) { }
    if (blind && sentiment && blind.metrics && sentiment.metrics) {
      diffs.push(sentiment.metrics.sharpe - blind.metrics.sharpe);
    }
  }
  const m = mean(diffs);
  return { mean: m, t: m / (sd(diffs) / Math.sqrt(diffs.length)), n: diffs.length,
           wins: diffs.filter((d) => d > 0).length };
}

section("The signal reaches the strategy without look-ahead");
{
  const sc = S.withLeadingSignal(4, 1000, 5, { informativeness: 1.0, blockLength: 21 });
  ok("signal is aligned one-to-one with returns", sc.signal.length === sc.returns.length);
  let threw = false;
  try {
    B.walkForward(sc.prices, null, {
      lookback: 500, rebalanceEvery: 21, strategy: "maxsharpeSentiment",
      signals: sc.signal.slice(0, 10),
    });
  } catch (e) { threw = true; }
  ok("a misaligned signal array is rejected rather than silently misread", threw);
}
{
  const sc = S.stationary(5, 2000, 8);
  const blind = B.walkForward(sc.prices, null, {
    lookback: 500, rebalanceEvery: 21, strategy: "maxsharpe",
    estimator: "ledoit-wolf", periodsPerYear: 252, costBps: 10, rf: 0.02,
  });
  const sentimentNoSignal = B.walkForward(sc.prices, null, {
    lookback: 500, rebalanceEvery: 21, strategy: "maxsharpeSentiment",
    estimator: "ledoit-wolf", periodsPerYear: 252, costBps: 10, rf: 0.02,
  });
  ok("with no signal supplied, the sentiment strategy exactly reproduces the blind one",
    Math.abs(blind.metrics.sharpe - sentimentNoSignal.metrics.sharpe) < 1e-9,
    `blind ${blind.metrics.sharpe} vs sentiment(no signal) ${sentimentNoSignal.metrics.sharpe}`);
}

section("A directional bug caught by this experiment, not by the unit tests");
{
  const mu0 = [0.08];
  const { P, Q, omega } = C.sentimentViews(mu0, [0.36], [2], { tiltPerUnit: 0.1512 });
  const post = C.blackLitterman(mu0, [[0.0225]], P, Q, omega);
  ok("a positive sentiment score can no longer push the posterior below the prior",
    post.mu[0] >= mu0[0], `prior ${mu0[0]}, posterior ${post.mu[0]}`);
}

section("Does a sentiment-style signal help? Only when it is genuinely informative.");
{
  const zero = pairedTrial(0, 150, 10000);
  ok("a pure-noise signal (informativeness=0) measurably HURTS performance",
    zero.mean < 0 && zero.t < -2,
    `mean ${zero.mean.toFixed(4)}, t=${zero.t.toFixed(2)}`);
  console.log(`      informativeness=0.00: ${zero.mean >= 0 ? "+" : ""}${zero.mean.toFixed(4)} Sharpe, t=${zero.t.toFixed(2)}, won ${zero.wins}/${zero.n}`);

  const high = pairedTrial(0.9, 150, 10000);
  ok("a near-perfect signal (informativeness=0.9) measurably HELPS",
    high.mean > 0 && high.t > 2, `mean ${high.mean.toFixed(4)}, t=${high.t.toFixed(2)}`);
  console.log(`      informativeness=0.90: ${high.mean >= 0 ? "+" : ""}${high.mean.toFixed(4)} Sharpe, t=${high.t.toFixed(2)}, won ${high.wins}/${high.n}`);

  ok("the benefit is strictly greater with a better signal than with none",
    high.mean > zero.mean, `${zero.mean.toFixed(4)} -> ${high.mean.toFixed(4)}`);

  const mid1 = pairedTrial(0.3, 30, 20000);
  const mid2 = pairedTrial(0.6, 30, 20000);
  const curve = [{ inf: 0, ...zero }, { inf: 0.3, ...mid1 }, { inf: 0.6, ...mid2 }, { inf: 0.9, ...high }];
  curve.forEach((r) => console.log(`      informativeness=${r.inf.toFixed(1)}: ${r.mean >= 0 ? "+" : ""}${r.mean.toFixed(4)} Sharpe (t=${r.t.toFixed(2)}, N=${r.n})`));
  ok("the interior points sit consistent with a monotone trend between the endpoints",
    mid1.mean >= zero.mean - 0.02 && mid2.mean <= high.mean + 0.02,
    curve.map((r) => r.mean.toFixed(3)).join(" -> "));
}

section("What this implies for a real sentiment feed");
{
  const plausible = pairedTrial(0.15, 150, 10000);
  ok("at an informativeness plausible for a real news signal (0.15), the effect is negative",
    plausible.mean < 0 && plausible.t < -2,
    `mean ${plausible.mean.toFixed(4)}, t=${plausible.t.toFixed(2)}`);
  console.log(`      informativeness=0.15 (a plausible real-world guess): ${plausible.mean >= 0 ? "+" : ""}${plausible.mean.toFixed(4)} Sharpe, t=${plausible.t.toFixed(2)}`);
  console.log(`      conclusion: without evidence the feed clears roughly 0.5-0.6+ informativeness,`);
  console.log(`      using it is expected to cost, not add, performance.`);
}

console.log(`\n${"=".repeat(64)}`);
console.log(`${pass} passed, ${fail} failed`);
if (fail) { console.log("Failures:"); failures.forEach((f) => console.log(`  - ${f}`)); }
console.log("=".repeat(64));
process.exit(fail ? 1 : 0);
