// ============================================================================
// scenarios.js
// Generators for market data that violates the assumptions the optimizer rests on.
//
// Mean-variance optimization assumes mu and Sigma are constant and knowable.
// The backtest in this repository used data generated exactly that way, which
// makes it a generous test. This module builds the failure modes real markets
// actually exhibit, so the optimizer can be measured where it is supposed to
// struggle rather than only where it is supposed to work.
//
//   stationary          the baseline assumption, for comparison
//   regimeSwitching     calm and crisis states with different mu, vol and correlation
//   volatilityClustering GARCH(1,1): quiet periods follow quiet, storms follow storms
//   structuralBreak     mu changes permanently partway through
//   correlationBreakdown correlations rise toward 1 exactly when diversification is needed
//
// All generators are seeded and reproducible.
// ============================================================================

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeNormal(rng) {
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

function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-14)) : s / L[j][j];
    }
  }
  return L;
}

function covFrom(vols, corr) {
  const n = vols.length;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => vols[i] * vols[j] * corr[i][j]));
}

function constantCorr(n, rho) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : rho)));
}

function drawFrom(mu, L, norm) {
  const n = mu.length;
  const z = Array.from({ length: n }, () => norm());
  const row = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k <= i; k++) s += L[i][k] * z[k];
    row[i] = mu[i] + s;
  }
  return row;
}

function toPrices(logReturns, start = 100) {
  const n = logReturns[0].length;
  const prices = [new Array(n).fill(start)];
  for (const row of logReturns) {
    const prev = prices[prices.length - 1];
    prices.push(prev.map((p, i) => p * Math.exp(row[i])));
  }
  return prices;
}

// ---------------------------------------------------------------- generators

// The assumption the model is built on: constant mu and Sigma forever.
function stationary(n, T, seed, opts = {}) {
  const rng = makeRng(seed);
  const norm = makeNormal(rng);
  const mu = opts.mu || Array.from({ length: n }, () => 0.0001 + rng() * 0.0004);
  const vols = opts.vols || Array.from({ length: n }, () => 0.008 + rng() * 0.010);
  const corr = opts.corr || constantCorr(n, 0.3);
  const L = cholesky(covFrom(vols, corr));

  const returns = [];
  for (let t = 0; t < T; t++) returns.push(drawFrom(mu, L, norm));
  return { prices: toPrices(returns), returns, regimes: new Array(T).fill(0), label: "Stationary" };
}

// Markov switching between a calm state and a crisis state. Crises are shorter,
// far more volatile, negative in drift, and much more correlated.
function regimeSwitching(n, T, seed, opts = {}) {
  const rng = makeRng(seed);
  const norm = makeNormal(rng);

  const calmMu = opts.calmMu || Array.from({ length: n }, () => 0.0003 + rng() * 0.0004);
  const calmVols = opts.calmVols || Array.from({ length: n }, () => 0.006 + rng() * 0.006);
  const crisisMult = opts.crisisVolMultiplier || 3.0;

  const calmL = cholesky(covFrom(calmVols, constantCorr(n, opts.calmCorr != null ? opts.calmCorr : 0.25)));
  const crisisVols = calmVols.map((v) => v * crisisMult);
  const crisisL = cholesky(covFrom(crisisVols, constantCorr(n, opts.crisisCorr != null ? opts.crisisCorr : 0.85)));
  const crisisMu = calmMu.map((m) => -Math.abs(m) * (opts.crisisDriftMultiplier || 4));

  const pCalmToCrisis = opts.pCalmToCrisis != null ? opts.pCalmToCrisis : 0.004;
  const pCrisisToCalm = opts.pCrisisToCalm != null ? opts.pCrisisToCalm : 0.04;

  const returns = [];
  const regimes = [];
  let inCrisis = false;
  for (let t = 0; t < T; t++) {
    if (inCrisis) { if (rng() < pCrisisToCalm) inCrisis = false; }
    else { if (rng() < pCalmToCrisis) inCrisis = true; }
    regimes.push(inCrisis ? 1 : 0);
    returns.push(drawFrom(inCrisis ? crisisMu : calmMu, inCrisis ? crisisL : calmL, norm));
  }
  return { prices: toPrices(returns), returns, regimes, label: "Regime switching" };
}

// GARCH(1,1) on a common factor: volatility persists, so calm and turbulent
// stretches cluster instead of arriving independently.
function volatilityClustering(n, T, seed, opts = {}) {
  const rng = makeRng(seed);
  const norm = makeNormal(rng);
  const mu = opts.mu || Array.from({ length: n }, () => 0.0002 + rng() * 0.0003);
  const baseVols = opts.vols || Array.from({ length: n }, () => 0.008 + rng() * 0.008);
  const corr = opts.corr || constantCorr(n, 0.3);
  const L = cholesky(covFrom(baseVols, corr));

  const omega = opts.omega != null ? opts.omega : 0.05;
  const alpha = opts.alpha != null ? opts.alpha : 0.10;
  const beta = opts.beta != null ? opts.beta : 0.85;

  const returns = [];
  const regimes = [];
  let h = 1;          // variance multiplier, long-run mean 1
  let lastShock = 0;
  for (let t = 0; t < T; t++) {
    h = omega + alpha * lastShock * lastShock + beta * h;
    const scale = Math.sqrt(h);
    const row = drawFrom(mu, L, norm).map((r, i) => mu[i] + (r - mu[i]) * scale);
    returns.push(row);
    regimes.push(scale > 1.5 ? 1 : 0);
    lastShock = (row[0] - mu[0]) / baseVols[0];
  }
  return { prices: toPrices(returns), returns, regimes, label: "Volatility clustering" };
}

// Expected returns change permanently partway through. Any estimator using a
// trailing window is fitting a mean that no longer exists.
function structuralBreak(n, T, seed, opts = {}) {
  const rng = makeRng(seed);
  const norm = makeNormal(rng);
  const breakAt = opts.breakAt || Math.floor(T / 2);

  const muBefore = opts.muBefore || Array.from({ length: n }, () => 0.0001 + rng() * 0.0005);
  // The ranking of assets reverses: whatever looked best becomes worst.
  const sorted = [...muBefore].sort((a, b) => a - b);
  const muAfter = opts.muAfter || muBefore.map((m) => {
    const rank = muBefore.filter((x) => x < m).length;
    return sorted[sorted.length - 1 - rank];
  });

  const vols = opts.vols || Array.from({ length: n }, () => 0.008 + rng() * 0.008);
  const L = cholesky(covFrom(vols, opts.corr || constantCorr(n, 0.3)));

  const returns = [];
  const regimes = [];
  for (let t = 0; t < T; t++) {
    const past = t < breakAt;
    regimes.push(past ? 0 : 1);
    returns.push(drawFrom(past ? muBefore : muAfter, L, norm));
  }
  return { prices: toPrices(returns), returns, regimes, breakAt, muBefore, muAfter, label: "Structural break" };
}

// Correlations rise toward 1 during drawdowns: diversification disappears at
// exactly the moment it is being relied upon.
function correlationBreakdown(n, T, seed, opts = {}) {
  const rng = makeRng(seed);
  const norm = makeNormal(rng);
  const mu = opts.mu || Array.from({ length: n }, () => 0.0002 + rng() * 0.0004);
  const vols = opts.vols || Array.from({ length: n }, () => 0.008 + rng() * 0.008);

  const calmL = cholesky(covFrom(vols, constantCorr(n, opts.calmCorr != null ? opts.calmCorr : 0.20)));
  const stressVols = vols.map((v) => v * (opts.stressVolMultiplier || 2.0));
  const stressL = cholesky(covFrom(stressVols, constantCorr(n, opts.stressCorr != null ? opts.stressCorr : 0.90)));

  const window = opts.window || 20;
  const threshold = opts.threshold != null ? opts.threshold : -0.03;

  const returns = [];
  const regimes = [];
  const recent = [];
  for (let t = 0; t < T; t++) {
    const trailing = recent.reduce((a, b) => a + b, 0);
    const stressed = recent.length >= window && trailing < threshold;
    regimes.push(stressed ? 1 : 0);
    const row = drawFrom(stressed ? mu.map((m) => -Math.abs(m)) : mu, stressed ? stressL : calmL, norm);
    returns.push(row);
    recent.push(row.reduce((a, b) => a + b, 0) / n);
    if (recent.length > window) recent.shift();
  }
  return { prices: toPrices(returns), returns, regimes, label: "Correlation breakdown" };
}

// A market where a genuine (but noisy) leading signal exists, with its
// informativeness as an explicit, controllable parameter. Built to answer one
// question honestly: does Black-Litterman + an external signal help, and does
// the benefit scale with how real the signal is — or does it just add churn?
//
// Each asset has a per-block "alpha" (excess expected return) that is redrawn
// every `blockLength` periods and held constant within the block. A "signal"
// is observable at the START of each block: signal = informativeness * alpha
// (standardized) + sqrt(1 - informativeness^2) * independent noise. At
// informativeness = 0 the signal is pure noise, uncorrelated with what
// actually happens; at 1 it is a perfect, if unrealistic, leading indicator.
// Real news/sentiment signals, per section 3i's signal-to-noise measurement,
// should be expected to sit far closer to 0 than to 1.
function withLeadingSignal(n, T, seed, opts = {}) {
  const rng = makeRng(seed);
  const norm = makeNormal(rng);
  const blockLength = opts.blockLength || 21;
  const informativeness = opts.informativeness != null ? opts.informativeness : 0.3;
  const alphaVol = opts.alphaVol != null ? opts.alphaVol : 0.0006; // per-period, roughly +/-15% annualized swings
  const baseMu = opts.baseMu || Array.from({ length: n }, () => 0.0001 + rng() * 0.0003);
  const vols = opts.vols || Array.from({ length: n }, () => 0.008 + rng() * 0.008);
  const corr = opts.corr || constantCorr(n, 0.25);
  const L = cholesky(covFrom(vols, corr));

  const returns = [];
  const signal = [];
  const alphaTrue = [];
  let blockAlpha = new Array(n).fill(0);
  let blockSignal = new Array(n).fill(0);

  for (let t = 0; t < T; t++) {
    if (t % blockLength === 0) {
      blockAlpha = Array.from({ length: n }, () => alphaVol * norm());
      blockSignal = blockAlpha.map((a) =>
        informativeness * (a / alphaVol) + Math.sqrt(Math.max(0, 1 - informativeness * informativeness)) * norm());
    }
    const mu = baseMu.map((m, i) => m + blockAlpha[i]);
    returns.push(drawFrom(mu, L, norm));
    signal.push(blockSignal.slice());
    alphaTrue.push(blockAlpha.slice());
  }
  return {
    prices: toPrices(returns), returns, signal, alphaTrue,
    regimes: new Array(T).fill(0), blockLength, informativeness,
    label: `Leading signal (informativeness=${informativeness})`,
  };
}

const SCENARIOS = {
  stationary, regimeSwitching, volatilityClustering, structuralBreak, correlationBreakdown,
  withLeadingSignal,
};

// ---------------------------------------------------------------- diagnostics

// Measures what a generated series actually did, so a scenario can be shown to
// have the property it claims rather than merely being named after it.
function describe(scenario) {
  const { returns, regimes } = scenario;
  const T = returns.length;
  const n = returns[0].length;

  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const sd = (a) => {
    if (a.length < 2) return 0;
    const m = mean(a);
    return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
  };

  const first = returns.map((r) => r[0]);
  const calmIdx = regimes.map((r, i) => (r === 0 ? i : -1)).filter((i) => i >= 0);
  const stressIdx = regimes.map((r, i) => (r === 1 ? i : -1)).filter((i) => i >= 0);

  // Autocorrelation of squared returns: the standard signature of clustering.
  const sq = first.map((r) => r * r);
  const msq = mean(sq);
  let num = 0, den = 0;
  for (let t = 1; t < T; t++) num += (sq[t] - msq) * (sq[t - 1] - msq);
  for (let t = 0; t < T; t++) den += (sq[t] - msq) ** 2;
  const volAutocorr = den > 0 ? num / den : 0;

  // Average pairwise correlation, computed separately by regime.
  const corrIn = (idx) => {
    if (idx.length < 10) return null;
    let total = 0, pairs = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = idx.map((t) => returns[t][i]);
        const b = idx.map((t) => returns[t][j]);
        const ma = mean(a), mb = mean(b);
        let cov = 0;
        for (let k = 0; k < a.length; k++) cov += (a[k] - ma) * (b[k] - mb);
        cov /= a.length - 1;
        const denom = sd(a) * sd(b);
        if (denom > 0) { total += cov / denom; pairs++; }
      }
    }
    return pairs > 0 ? total / pairs : null;
  };

  return {
    periods: T,
    stressFraction: stressIdx.length / T,
    volAutocorrelation: volAutocorr,
    calmVol: calmIdx.length > 1 ? sd(calmIdx.map((t) => returns[t][0])) : null,
    stressVol: stressIdx.length > 1 ? sd(stressIdx.map((t) => returns[t][0])) : null,
    calmCorrelation: corrIn(calmIdx),
    stressCorrelation: corrIn(stressIdx),
    meanFirstHalf: mean(first.slice(0, Math.floor(T / 2))),
    meanSecondHalf: mean(first.slice(Math.floor(T / 2))),
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    SCENARIOS, describe, makeRng, makeNormal, cholesky, covFrom, constantCorr, toPrices,
    stationary, regimeSwitching, volatilityClustering, structuralBreak, correlationBreakdown,
    withLeadingSignal,
  };
}
