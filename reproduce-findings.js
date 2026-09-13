// ============================================================================
// reproduce-findings.js
//
// Every number quoted in README.md is regenerated here and compared against
// what the documentation claims. Documentation drifts silently as code changes;
// this makes that drift a test failure rather than a discovery someone else
// makes later.
//
//   node reproduce-findings.js
//
// A claim is FLAGGED when the regenerated value falls outside the tolerance
// recorded beside it. Flags are not necessarily bugs — they mean the README and
// the code no longer agree, and one of them needs correcting.
// ============================================================================

const C = require("./portfolio-core.js");
const D = require("./data-pipeline.js");
const B = require("./backtest.js");
const S = require("./scenarios.js");

let checked = 0, flagged = 0;
const flags = [];

function claim(section, description, documented, actual, tolerance, format = (v) => String(v)) {
  checked++;
  const ok = Math.abs(actual - documented) <= tolerance;
  if (!ok) { flagged++; flags.push(`${section}: ${description}`); }
  console.log(
    `  ${ok ? "  ok  " : " FLAG "} ${description.padEnd(52)} ` +
    `README ${format(documented).padStart(9)}   actual ${format(actual).padStart(9)}`
  );
}

function note(text) { console.log(`         ${text}`); }
function section(name) { console.log(`\n== ${name}`); }

// ---------------------------------------------------------------- fixtures
const VOLS = [0.15, 0.06, 0.17, 0.19, 0.16];
const CORR = [
  [1, -0.1, 0.85, 0.6, 0.1],
  [-0.1, 1, -0.05, 0.1, 0.2],
  [0.85, -0.05, 1, 0.55, 0.15],
  [0.6, 0.1, 0.55, 1, 0.2],
  [0.1, 0.2, 0.15, 0.2, 1],
];
const COV = C.buildCovariance(VOLS, CORR);
const MU = [0.10, 0.04, 0.08, 0.09, 0.06];
const RF = 0.02;

function mulberry(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => {
  const m = mean(a);
  return Math.sqrt(a.reduce((t, v) => t + (v - m) ** 2, 0) / (a.length - 1));
};
const median = (a) => { const s2 = [...a].sort((x, y) => x - y); return s2.length ? s2[Math.floor(s2.length / 2)] : NaN; };
const pctFmt = (v) => (v * 100).toFixed(1) + "%";
const num3 = (v) => v.toFixed(3);
const num4 = (v) => v.toFixed(4);

// ================================================================ section 3
section("Section 3 — exact QP versus Monte Carlo");
{
  const exact = C.maxSharpeLongOnly(MU, COV, RF);
  const rand = mulberry(424242);
  let bestMC = -Infinity;
  for (let k = 0; k < 20000; k++) {
    const e = Array.from({ length: 5 }, () => -Math.log(rand() + 1e-12));
    const tot = e.reduce((a, b) => a + b, 0);
    const w = e.map((x) => x / tot);
    const vol = Math.sqrt(C.quadForm(w, COV));
    const sh = (C.dot(w, MU) - RF) / vol;
    if (sh > bestMC) bestMC = sh;
  }
  const improvement = (exact.sharpe / bestMC - 1) * 100;
  claim("3", "exact QP improvement over 20,000 MC draws", 1.77, improvement, 0.30,
    (v) => v.toFixed(2) + "%");
}

// ================================================================ section 3e
section("Section 3e — Ledoit-Wolf shrinkage");
{
  function cholesky(A) {
    const n = A.length;
    const L = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s2 = A[i][j];
      for (let k = 0; k < j; k++) s2 -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s2, 1e-14)) : s2 / L[j][j];
    }
    return L;
  }
  function normalGen(rng) {
    let spare = null;
    return () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u, v, q;
      do { u = 2 * rng() - 1; v = 2 * rng() - 1; q = u * u + v * v; } while (q === 0 || q >= 1);
      const f = Math.sqrt(-2 * Math.log(q) / q);
      spare = v * f;
      return u * f;
    };
  }
  function gen(mu, cov, T, seed) {
    const n = mu.length, L = cholesky(cov), nm = normalGen(mulberry(seed)), out = [];
    for (let t = 0; t < T; t++) {
      const z = Array.from({ length: n }, () => nm());
      const row = new Array(n);
      for (let i = 0; i < n; i++) {
        let s2 = 0;
        for (let k = 0; k <= i; k++) s2 += L[i][k] * z[k];
        row[i] = mu[i] + s2;
      }
      out.push(row);
    }
    return out;
  }
  const frob = (A, Bm) => {
    let s2 = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A.length; j++) s2 += (A[i][j] - Bm[i][j]) ** 2;
    return Math.sqrt(s2);
  };

  // error reduction on a 40-observation sample
  const n = 8;
  const trueCov = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 0.0004 : 0.00008)));
  const X = gen(new Array(n).fill(0.0003), trueCov, 40, 2468);
  const lw = D.ledoitWolfShrinkage(X);
  const errSample = frob(D.sampleCovariance(X, false), trueCov);
  const errLW = frob(lw.covariance, trueCov);
  claim("3e", "shrinkage error reduction at T=40", 46, (1 - errLW / errSample) * 100, 4,
    (v) => v.toFixed(0) + "%");

  // delta by sample size, misspecified target
  const m2 = 6;
  const loadings = [[0.020, 0.004], [0.015, -0.010], [0.008, 0.012],
                    [0.030, 0.002], [0.005, -0.015], [0.022, 0.009]];
  const idio = [0.00020, 0.00008, 0.00030, 0.00012, 0.00025, 0.00006];
  const hetCov = Array.from({ length: m2 }, (_, i) =>
    Array.from({ length: m2 }, (_, j) => {
      let v = loadings[i][0] * loadings[j][0] + loadings[i][1] * loadings[j][1];
      if (i === j) v += idio[i];
      return v;
    }));
  const documented = { 40: 0.226, 250: 0.035, 1500: 0.006, 10000: 0.001 };
  for (const T of [40, 250, 1500, 10000]) {
    const d = D.ledoitWolfShrinkage(gen(new Array(m2).fill(0.0003), hetCov, T, 8642)).delta;
    claim("3e", `shrinkage intensity at T=${T}`, documented[T], d, 0.02, num3);
  }
}

// ================================================================ section 3f
section("Section 3f — position limits");
{
  // README quotes these against the ESTIMATED inputs from the integration check,
  // which are regenerated here from the same synthetic universe.
  const documented = [
    { label: "unconstrained", bounds: null, sharpe: 0.309, held: 1 },
    { label: "max 30%", bounds: { maxWeight: 0.30 }, sharpe: 0.106, held: 4 },
    { label: "max 25%, min 5%", bounds: { maxWeight: 0.25, minWeight: 0.05 }, sharpe: 0.079, held: 5 },
  ];
  note("uses the estimated inputs from the integration fixture, not the manual assumptions");
  const est = buildEstimatedFixture();
  for (const d of documented) {
    const r = C.maxSharpeLongOnly(est.mu, est.cov, 0.02, d.bounds);
    claim("3f", `Sharpe, ${d.label}`, d.sharpe, r.sharpe, 0.02, num3);
    const held = r.weights.filter((w) => w > 0.005).length;
    claim("3f", `assets held, ${d.label}`, d.held, held, 0.5, (v) => String(Math.round(v)));
  }
}

function buildEstimatedFixture() {
  // Same construction as integration-check: a 2-factor universe, estimated
  // through the pipeline, then rounded the way the UI rounds.
  const n = 5;
  const load = [[0.011, 0.003], [0.004, -0.006], [0.013, 0.005], [0.009, -0.002], [0.006, 0.008]];
  const idio = [0.00008, 0.00002, 0.00011, 0.00006, 0.00009];
  const cov = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
    let v = load[i][0] * load[j][0] + load[i][1] * load[j][1];
    if (i === j) v += idio[i];
    return v;
  }));
  const mu = [0.00045, 0.00012, 0.00050, 0.00030, 0.00020];
  const sc = S.stationary(n, 2000, 4242, {
    mu, vols: cov.map((r, i) => Math.sqrt(r[i])),
    corr: cov.map((row, i) => row.map((v, j) => v / Math.sqrt(cov[i][i] * cov[j][j]))),
  });
  // Weekday-only dates (skipping Sat/Sun), matching a real stock market
  // calendar: a fully-consecutive (weekend-inclusive) series here is
  // misread by inferFrequency as a 24/7 market (365 periods/year) rather
  // than the 252-day weekday calendar this 5-asset universe represents.
  let csv = "Date,Equity,Bonds,SmallCap,Intl,Gold\n";
  const cur = new Date(Date.UTC(2016, 0, 1));
  sc.prices.forEach((row) => {
    while (cur.getUTCDay() === 0 || cur.getUTCDay() === 6) cur.setUTCDate(cur.getUTCDate() + 1);
    csv += cur.toISOString().slice(0, 10) + "," + row.map((v) => v.toFixed(4)).join(",") + "\n";
    cur.setUTCDate(cur.getUTCDate() + 1);
  });
  const est = D.estimateFromCSV(csv, { estimator: "ledoit-wolf" });
  const uiVols = est.vols.map((v) => Math.round(v * 10000) / 100);
  const uiCorr = est.corr.map((r) => r.map((v) => Math.round(v * 1000) / 1000));
  return {
    mu: est.mu.map((m) => Math.round(m * 10000) / 10000),
    cov: C.buildCovariance(uiVols.map((v) => v / 100), uiCorr),
  };
}

// ================================================================ section 3g
section("Section 3g — group constraints, solver outcomes");
{
  const rand = mulberry(31415926);
  let solvedExactly = 0, refused = 0, approximate = 0;
  for (let t = 0; t < 400; t++) {
    const n = 3 + Math.floor(rand() * 5);
    const L = Array.from({ length: n }, () => Array.from({ length: n }, () => rand() - 0.5));
    const cov = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let v = 0;
        for (let k = 0; k < n; k++) v += L[i][k] * L[j][k];
        cov[i][j] = v;
      }
      cov[i][i] += 0.01;
    }
    const mu = Array.from({ length: n }, () => rand() * 0.2 - 0.03);
    // tau must be drawn ONCE, exactly as the test suite draws it. Inlining it
    // into the map consumed one random number per asset and desynchronised the
    // entire sequence, producing figures that looked plausible and were wrong.
    const tau = Math.pow(10, -2 + 4 * rand());
    const c = mu.map((m) => tau * m);
    const lower = Array.from({ length: n }, () => rand() * (0.8 / n));
    const upper = lower.map((lo) => lo + 0.1 + rand() * 0.8);
    const nGroups = 1 + Math.floor(rand() * 2);
    const groups = [];
    for (let gi = 0; gi < nGroups; gi++) {
      const members = [];
      for (let i = 0; i < n; i++) if (rand() < 0.5) members.push(i);
      if (members.length === 0) members.push(Math.floor(rand() * n));
      const spec = { name: `g${gi}`, assets: members };
      if (rand() < 0.7) spec.max = 0.2 + rand() * 0.8;
      if (rand() < 0.4) spec.min = rand() * 0.3;
      groups.push(spec);
    }
    try {
      if (C.checkGroupFeasibility(n, lower, upper, groups)) { refused++; continue; }
      const { G, h } = C.buildGroupConstraints(n, groups);
      const r = C.solveQP(cov, c, new Array(n).fill(1), 1, { lower, upper, G, h });
      if (r.converged) solvedExactly++; else approximate++;
    } catch (e) { refused++; }
  }
  claim("3g", "solved exactly out of 400 random mandates", 246, solvedExactly, 10, (v) => String(Math.round(v)));
  claim("3g", "refused as genuinely impossible", 111, refused, 10, (v) => String(Math.round(v)));
  claim("3g", "returned an approximate but feasible answer", 43, approximate, 10, (v) => String(Math.round(v)));
}

// ================================================================ section 3i
section("Section 3i — scenario properties");
{
  const rs = S.describe(S.regimeSwitching(5, 6000, 22));
  claim("3i", "regime switching: crisis share of periods", 0.096, rs.stressFraction, 0.02, pctFmt);
  claim("3i", "regime switching: calm volatility", 0.0113, rs.calmVol, 0.002, (v) => (v * 100).toFixed(2) + "%");
  claim("3i", "regime switching: crisis volatility", 0.0342, rs.stressVol, 0.005, (v) => (v * 100).toFixed(2) + "%");
  claim("3i", "regime switching: calm correlation", 0.24, rs.calmCorrelation, 0.04, num3);
  claim("3i", "regime switching: crisis correlation", 0.85, rs.stressCorrelation, 0.04, num3);

  const vc = S.describe(S.volatilityClustering(5, 6000, 33));
  const flat = S.describe(S.stationary(5, 6000, 33));
  claim("3i", "clustering: squared-return autocorrelation", 0.161, vc.volAutocorrelation, 0.03, num3);
  claim("3i", "stationary: squared-return autocorrelation", 0.002, flat.volAutocorrelation, 0.03, num3);

  const cb = S.describe(S.correlationBreakdown(5, 6000, 55));
  claim("3i", "breakdown: calm correlation", 0.20, cb.calmCorrelation, 0.05, num3);
  claim("3i", "breakdown: stressed correlation", 0.89, cb.stressCorrelation, 0.05, num3);
}

section("Section 3i — signal-to-noise on expected returns");
{
  const LOOKBACK = 500, DAILY_VOL = 0.012;
  const seAnnualPP = (DAILY_VOL / Math.sqrt(LOOKBACK)) * 252 * 100;
  const sc = S.structuralBreak(6, 3000, 961);
  const spreadPP = (Math.max(...sc.muBefore) - Math.min(...sc.muBefore)) * 252 * 100;
  claim("3i", "spread in true annual returns (pp)", 8.2, spreadPP, 0.3, (v) => v.toFixed(1));
  claim("3i", "standard error of the estimate (pp)", 13.5, seAnnualPP, 0.3, (v) => v.toFixed(1));
  claim("3i", "signal-to-noise ratio", 0.61, spreadPP / seAnnualPP, 0.04, (v) => v.toFixed(2));
  claim("3i", "covariance relative error at T=500", 6.3, Math.sqrt(2 / LOOKBACK) * 100, 0.2,
    (v) => v.toFixed(1) + "%");
}

section("Section 3i — paired strategy comparison");
{
  function pairedDiff(gen, a, b, trials) {
    const diffs = [];
    for (let t = 0; t < trials; t++) {
      const sc = gen(6, 3000, 900 + t * 61);
      const run = (strat) => B.walkForward(sc.prices, null, {
        lookback: 500, rebalanceEvery: 21, strategy: strat,
        estimator: "ledoit-wolf", periodsPerYear: 252, costBps: 10, rf: 0.02,
      }).metrics.sharpe;
      let x = null, y = null;
      try { x = run(a); } catch (e) { /* skip */ }
      try { y = run(b); } catch (e) { /* skip */ }
      if (x !== null && y !== null) diffs.push(x - y);
    }
    const m = mean(diffs);
    return { mean: m, t: m / (sd(diffs) / Math.sqrt(diffs.length)) };
  }
  const st = pairedDiff(S.stationary, "minvar", "maxsharpe", 60);
  claim("3i", "stationary: minvar - maxsharpe", 0.0947, st.mean, 0.02, num4);
  claim("3i", "stationary: t-statistic", 2.84, st.t, 0.6, (v) => v.toFixed(2));

  const rg = pairedDiff(S.regimeSwitching, "minvar", "maxsharpe", 60);
  claim("3i", "regime switching: minvar - maxsharpe", 0.0435, rg.mean, 0.03, num4);
  claim("3i", "regime switching: t-statistic", 1.61, rg.t, 0.7, (v) => v.toFixed(2));
}

// ================================================================ section 3k
section("Section 3k — transaction costs");
{
  const cov = [
    [0.00012, 0.00003, 0.00005, 0.00002, 0.00003, 0.00001],
    [0.00003, 0.00006, 0.00001, 0.00001, 0.00002, 0.00001],
    [0.00005, 0.00001, 0.00020, 0.00003, 0.00004, 0.00002],
    [0.00002, 0.00001, 0.00003, 0.00009, 0.00002, 0.00001],
    [0.00003, 0.00002, 0.00004, 0.00002, 0.00014, 0.00002],
    [0.00001, 0.00001, 0.00002, 0.00001, 0.00002, 0.00007],
  ];
  const mu = [0.00035, 0.00012, 0.00040, 0.00025, 0.00018, 0.00022];
  const vols = cov.map((r, i) => Math.sqrt(r[i]));
  const corr = cov.map((row, i) => row.map((v, j) => v / (vols[i] * vols[j])));

  function paired(blind, aware, costBps, trials = 25) {
    const diffs = [], tb = [], ta = [];
    for (let k = 0; k < trials; k++) {
      const sc = S.stationary(6, 3000, 900 + k * 61, { mu, vols, corr });
      const run = (strat) => B.walkForward(sc.prices, null, {
        lookback: 500, rebalanceEvery: 21, strategy: strat,
        estimator: "ledoit-wolf", periodsPerYear: 252, costBps, rf: 0.02,
      });
      let x = null, y = null;
      try { x = run(blind); } catch (e) { /* skip */ }
      try { y = run(aware); } catch (e) { /* skip */ }
      if (x && y && x.metrics && y.metrics) {
        diffs.push(y.metrics.sharpe - x.metrics.sharpe);
        tb.push(x.averageTurnover); ta.push(y.averageTurnover);
      }
    }
    const m = mean(diffs);
    return { mean: m, t: m / (sd(diffs) / Math.sqrt(diffs.length)),
             turnoverBlind: mean(tb), turnoverAware: mean(ta) };
  }

  const mv10 = paired("minvar", "minvarCost", 10);
  claim("3k", "10bps min variance: Sharpe change", 0.021, mv10.mean, 0.02, num3);
  claim("3k", "10bps min variance: turnover before", 0.024, mv10.turnoverBlind, 0.008, pctFmt);
  claim("3k", "10bps min variance: turnover after", 0.006, mv10.turnoverAware, 0.008, pctFmt);

  const ms10 = paired("maxsharpe", "maxsharpeCost", 10);
  claim("3k", "10bps max Sharpe: Sharpe change", -0.002, ms10.mean, 0.03, num3);
  claim("3k", "10bps max Sharpe: turnover before", 0.159, ms10.turnoverBlind, 0.04, pctFmt);

  const ms200 = paired("maxsharpe", "maxsharpeCost", 200);
  claim("3k", "200bps max Sharpe: Sharpe change", 0.412, ms200.mean, 0.10, num3);
  claim("3k", "200bps max Sharpe: turnover after", 0.060, ms200.turnoverAware, 0.03, pctFmt);
}

// ================================================================ section 3l
section("Section 3l — sentiment / Black-Litterman");
{
  const alphaVol = 0.0006, periodsPerYear = 252;
  const tiltPerUnit = alphaVol * periodsPerYear;
  function pairedTrial(informativeness, trials, seedBase) {
    const diffs = [];
    for (let trial = 0; trial < trials; trial++) {
      const sc = S.withLeadingSignal(6, 3000, seedBase + trial * 131, {
        informativeness, blockLength: 21, alphaVol,
      });
      const run = (strategy, signals) => B.walkForward(sc.prices, null, {
        lookback: 500, rebalanceEvery: 21, strategy,
        estimator: "ledoit-wolf", periodsPerYear, costBps: 10, rf: 0.02,
        signals, sentimentTilt: tiltPerUnit,
      });
      let blind = null, sentiment = null;
      try { blind = run("maxsharpe"); } catch (e) { /* skip */ }
      try { sentiment = run("maxsharpeSentiment", sc.signal); } catch (e) { /* skip */ }
      if (blind && sentiment && blind.metrics && sentiment.metrics) {
        diffs.push(sentiment.metrics.sharpe - blind.metrics.sharpe);
      }
    }
    const m = mean(diffs);
    return { mean: m, t: m / (sd(diffs) / Math.sqrt(diffs.length)) };
  }

  const zero = pairedTrial(0, 150, 10000);
  claim("3l", "informativeness=0: Sharpe change", -0.0119, zero.mean, 0.006, num4);
  claim("3l", "informativeness=0: t-statistic", -2.88, zero.t, 0.8, (v) => v.toFixed(2));

  const high = pairedTrial(0.9, 150, 10000);
  claim("3l", "informativeness=0.9: Sharpe change", 0.0108, high.mean, 0.006, num4);
  claim("3l", "informativeness=0.9: t-statistic", 3.15, high.t, 0.8, (v) => v.toFixed(2));

  const plausible = pairedTrial(0.15, 150, 10000);
  claim("3l", "informativeness=0.15 (plausible real feed): Sharpe change", -0.0095, plausible.mean, 0.006, num4);
  claim("3l", "informativeness=0.15: t-statistic", -2.31, plausible.t, 0.8, (v) => v.toFixed(2));
}

// ================================================================ summary
console.log(`\n${"=".repeat(78)}`);
if (flagged === 0) {
  console.log(`${checked} documented claims regenerated, all consistent with README.md`);
} else {
  console.log(`${checked} claims checked, ${flagged} no longer match README.md:`);
  flags.forEach((f) => console.log(`  - ${f}`));
  console.log("\nThe code and the documentation disagree. One of them is wrong.");
}
console.log("=".repeat(78));
process.exit(flagged ? 1 : 0);
