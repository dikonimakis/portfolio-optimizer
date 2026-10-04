const D = require("./data-pipeline.js");
const C = require("./portfolio-core.js");

let pass = 0, fail = 0;
const failures = [];
function ok(name, cond, detail = "") {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; failures.push(name); console.log(`  FAIL  ${name}  ${detail}`); }
}
const close = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;
const section = (s) => console.log(`\n== ${s}`);

// ---------------------------------------------------------------- synthetic data
// Generates returns from a KNOWN mu and Sigma so estimators can be checked
// against ground truth rather than against each other.
function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) L[i][j] = Math.sqrt(Math.max(s, 1e-14));
      else L[i][j] = s / L[j][j];
    }
  }
  return L;
}
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
    let u, v, sq;
    do { u = 2 * rng() - 1; v = 2 * rng() - 1; sq = u * u + v * v; } while (sq === 0 || sq >= 1);
    const f = Math.sqrt(-2 * Math.log(sq) / sq);
    spare = v * f;
    return u * f;
  };
}
function generateReturns(mu, cov, T, seed) {
  const n = mu.length;
  const L = cholesky(cov);
  const norm = makeNormal(makeRng(seed));
  const out = [];
  for (let t = 0; t < T; t++) {
    const z = Array.from({ length: n }, () => norm());
    const row = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k <= i; k++) s += L[i][k] * z[k];
      row[i] = mu[i] + s;
    }
    out.push(row);
  }
  return out;
}
function pricesFromReturns(r0, returns) {
  const prices = [r0.slice()];
  for (const row of returns) {
    const prev = prices[prices.length - 1];
    prices.push(prev.map((p, i) => p * Math.exp(row[i])));
  }
  return prices;
}
function frobenius(A, B) {
  let s = 0;
  for (let i = 0; i < A.length; i++) for (let j = 0; j < A.length; j++) s += (A[i][j] - B[i][j]) ** 2;
  return Math.sqrt(s);
}

// ---------------------------------------------------------------- CSV parsing
section("CSV parsing");
{
  const csv = `Date,Alpha,Beta
2020-01-01,100,50
2020-01-02,101,50.5
2020-01-03,102,49.8`;
  const p = D.parsePriceCSV(csv);
  ok("reads asset names from the header", p.names.join(",") === "Alpha,Beta");
  ok("reads all price rows", p.prices.length === 3);
  ok("parses numbers correctly", p.prices[1][0] === 101 && p.prices[2][1] === 49.8);
}
{
  const csv = `Date,A\n2020-01-03,102\n2020-01-02,101\n2020-01-01,100`;
  const p = D.parsePriceCSV(csv);
  ok("newest-first files are flipped to chronological order",
    p.prices[0][0] === 100 && p.prices[2][0] === 102);
}
{
  const csv = `Date,"Fund, Global",B\n2020-01-01,"1,250.50",10\n2020-01-02,"$1,300.00",11`;
  const p = D.parsePriceCSV(csv);
  ok("handles quoted names containing commas", p.names[0] === "Fund, Global");
  ok("strips thousands separators and currency symbols", p.prices[0][0] === 1250.5 && p.prices[1][0] === 1300);
}
{
  const csv = `Date,A,B\n2020-01-01,100,50\n2020-01-02,,51\n2020-01-03,102,N/A\n2020-01-04,103,53`;
  const p = D.parsePriceCSV(csv);
  ok("missing values become NaN rather than zero",
    !Number.isFinite(p.prices[1][0]) && !Number.isFinite(p.prices[2][1]));
  ok("zero is preserved as a real value, not treated as missing",
    D.parseNumber("0") === 0 && !Number.isFinite(D.parseNumber("")));
}
{
  let threw = false;
  try { D.parsePriceCSV("Date,A"); } catch (e) { threw = true; }
  ok("a file with no data rows is rejected with a clear error", threw);
}

// ---------------------------------------------------------------- returns
section("Return computation");
{
  const prices = [[100], [110], [99]];
  const simple = D.pricesToReturns(prices, "simple").returns;
  ok("simple return is correct", close(simple[0][0], 0.10, 1e-12));
  ok("simple return handles a fall", close(simple[1][0], -0.10, 1e-12));

  const log = D.pricesToReturns(prices, "log").returns;
  ok("log return is correct", close(log[0][0], Math.log(1.1), 1e-12));
  ok("log returns are time-additive",
    close(log[0][0] + log[1][0], Math.log(99 / 100), 1e-12));
}
{
  const prices = [[100, 50], [101, NaN], [102, 52], [103, 53]];
  const res = D.pricesToReturns(prices, "log");
  ok("periods with a missing price are dropped, keeping series aligned", res.returns.length === 1);
  ok("dropped periods are reported", res.droppedRows.length === 2);
}
{
  const prices = [[100], [110], [0], [50], [55]];
  const res = D.pricesToReturns(prices, "log");
  ok("zero prices are excluded rather than producing infinities",
    res.returns.every((r) => r.every(Number.isFinite)));
  ok("periods either side of a zero price are both dropped", res.returns.length === 2);

  let threwAllBad = false;
  try { D.pricesToReturns([[100], [NaN], [0]], "log"); } catch (e) { threwAllBad = true; }
  ok("a series with no usable period at all fails with a clear message", threwAllBad);
}

// ---------------------------------------------------------------- frequency
section("Frequency inference");
{
  // A realistic weekday-only (stock market) calendar: skips Saturday/Sunday,
  // so roughly one gap in five is a 3-day weekend jump, not just 1-day gaps.
  const weekdaysOnly = [];
  for (let d = new Date(Date.UTC(2020, 0, 1)); weekdaysOnly.length < 260; d.setUTCDate(d.getUTCDate() + 1)) {
    const day = d.getUTCDay();
    if (day !== 0 && day !== 6) weekdaysOnly.push(d.toISOString().slice(0, 10));
  }
  ok("detects daily (weekday-market) data", D.inferFrequency(weekdaysOnly).periodsPerYear === 252);

  const monthly = [];
  for (let i = 0; i < 24; i++) monthly.push(new Date(Date.UTC(2020, i, 1)).toISOString().slice(0, 10));
  ok("detects monthly data", D.inferFrequency(monthly).periodsPerYear === 12);

  const weekly = [];
  for (let i = 0; i < 30; i++) weekly.push(new Date(Date.UTC(2020, 0, 1 + i * 7)).toISOString().slice(0, 10));
  ok("detects weekly data", D.inferFrequency(weekly).periodsPerYear === 52);

  // A genuine 24/7 calendar (crypto): every consecutive calendar day present,
  // including weekends. Median gap is also exactly 1 day, same as the
  // weekday case above -- the two are distinguished by what FRACTION of
  // gaps are 1 day, not by the median, which is why this needed its own test
  // rather than trusting the weekday-market test to cover it.
  const daily365 = [];
  for (let i = 0; i < 260; i++) daily365.push(new Date(Date.UTC(2020, 0, 1 + i)).toISOString().slice(0, 10));
  const freq365 = D.inferFrequency(daily365);
  ok("a 24/7 calendar (e.g. crypto) is detected as 365 periods/year, not 252",
    freq365.periodsPerYear === 365, `got ${freq365.periodsPerYear}`);
  ok("the 365-day classification is still marked confident", freq365.confident === true);

  // Guards against a false positive: a short weekday sample that simply
  // has not crossed a weekend yet must NOT be misread as a 24/7 market --
  // it falls back to the traditional (safe default) weekday assumption.
  const shortRun = [];
  for (let i = 0; i < 5; i++) shortRun.push(new Date(Date.UTC(2020, 0, 1 + i)).toISOString().slice(0, 10)); // Wed-Sun
  ok("a short run that hasn't crossed enough gaps yet defaults to the weekday assumption, not 365",
    D.inferFrequency(shortRun).periodsPerYear === 252);
}

// ---------------------------------------------------------------- estimator accuracy
section("Estimators recover known parameters");
{
  const trueMu = [0.0004, 0.0002, 0.0006];
  const trueCov = [
    [0.0001, 0.00002, 0.00004],
    [0.00002, 0.00005, 0.00001],
    [0.00004, 0.00001, 0.00015],
  ];
  const X = generateReturns(trueMu, trueCov, 20000, 777);

  // Correctness here must be judged against the standard error, not an
  // arbitrary tolerance: no estimator can beat sigma/sqrt(T).
  const m = D.colMeans(X);
  const stderrs = trueCov.map((r, i) => Math.sqrt(r[i] / X.length));
  const zScores = m.map((v, i) => (v - trueMu[i]) / stderrs[i]);
  ok("sample mean is within 4 standard errors of the true mean",
    zScores.every((z) => Math.abs(z) < 4),
    `z-scores ${zScores.map((z) => z.toFixed(2))}`);

  const S = D.sampleCovariance(X, true);
  const relErr = frobenius(S, trueCov) / frobenius(trueCov, trueCov.map((r) => r.map(() => 0)));
  ok("sample covariance converges to the true covariance", relErr < 0.05,
    `relative Frobenius error ${(relErr * 100).toFixed(2)}%`);

  ok("covariance is symmetric", S.every((r, i) => r.every((v, j) => close(v, S[j][i], 1e-15))));
  ok("covariance is positive semi-definite",
    C.symmetricEigen(S).values.every((v) => v > -1e-12));
}
{
  // Unbiasedness: 1/(T-1) vs 1/T should differ by exactly the factor T/(T-1)
  const X = generateReturns([0, 0], [[0.01, 0.002], [0.002, 0.02]], 50, 31337);
  const unb = D.sampleCovariance(X, true);
  const mle = D.sampleCovariance(X, false);
  ok("unbiased and MLE covariance differ by exactly T/(T-1)",
    close(unb[0][0] / mle[0][0], 50 / 49, 1e-12));
}

// ---------------------------------------------------------------- Ledoit-Wolf
section("Ledoit-Wolf shrinkage (the estimate that needs raw data)");
{
  const n = 8;
  const trueCov = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 0.0004 : 0.00008)));
  const trueMu = new Array(n).fill(0.0003);

  // Small sample: shrinkage should help a lot
  const Xsmall = generateReturns(trueMu, trueCov, 40, 2468);
  const lwS = D.ledoitWolfShrinkage(Xsmall);
  const sampleS = D.sampleCovariance(Xsmall, false);
  const errSample = frobenius(sampleS, trueCov);
  const errLW = frobenius(lwS.covariance, trueCov);
  ok("shrinkage intensity is a valid proportion", lwS.delta >= 0 && lwS.delta <= 1,
    `delta=${lwS.delta.toFixed(4)}`);
  ok("shrinkage beats the raw sample covariance on a short history",
    errLW < errSample,
    `LW error ${errLW.toExponential(3)} vs sample ${errSample.toExponential(3)} (${((1 - errLW / errSample) * 100).toFixed(1)}% better)`);
  console.log(`      T=40:   delta=${lwS.delta.toFixed(3)}, error reduced ${((1 - errLW / errSample) * 100).toFixed(1)}%`);

  // This true covariance has EXACTLY constant correlation, so the shrinkage
  // target is perfectly specified and delta should go to its ceiling.
  ok("when the target is exactly correct, shrinkage goes all the way to it",
    lwS.delta > 0.95, `delta=${lwS.delta.toFixed(4)}`);

  // Back-off test needs a covariance the constant-correlation target CANNOT
  // represent — heterogeneous variances and correlations, built from a two-
  // factor model so it is guaranteed positive definite.
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
  const hetMu = new Array(m2).fill(0.0003);

  const deltas = [40, 250, 1500, 10000].map((T) => ({
    T, delta: D.ledoitWolfShrinkage(generateReturns(hetMu, hetCov, T, 8642)).delta,
  }));
  deltas.forEach((d) => console.log(`      T=${String(d.T).padStart(5)}  delta=${d.delta.toFixed(4)}`));
  ok("with a misspecified target, shrinkage backs off as the sample grows",
    deltas[deltas.length - 1].delta < deltas[0].delta,
    deltas.map((d) => `T=${d.T}:${d.delta.toFixed(3)}`).join(" "));
  ok("shrinkage declines monotonically with sample size",
    deltas.every((d, i) => i === 0 || d.delta <= deltas[i - 1].delta + 0.02));

  ok("shrunk covariance stays positive semi-definite",
    C.symmetricEigen(lwS.covariance).values.every((v) => v > -1e-14));
  ok("shrunk covariance is symmetric",
    lwS.covariance.every((r, i) => r.every((v, j) => close(v, lwS.covariance[j][i], 1e-15))));
  ok("shrinkage preserves the sample variances on the diagonal",
    lwS.covariance.every((r, i) => close(r[i], sampleS[i][i], 1e-12)));
}
{
  // The headline case: sample covariance is singular when T < n, but the
  // optimizer needs to invert it. Shrinkage must fix that.
  const n = 12;
  const trueCov = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 0.0004 : 0.0001)));
  const X = generateReturns(new Array(n).fill(0.0002), trueCov, 9, 999);

  const sampleS = D.sampleCovariance(X, false);
  const minEigSample = Math.min(...C.symmetricEigen(sampleS).values);
  const lw = D.ledoitWolfShrinkage(X);
  const minEigLW = Math.min(...C.symmetricEigen(lw.covariance).values);

  ok("with fewer observations than assets the sample covariance is singular",
    minEigSample < 1e-9, `min eigenvalue ${minEigSample.toExponential(2)}`);
  ok("shrinkage makes it invertible again", minEigLW > 1e-9,
    `min eigenvalue ${minEigLW.toExponential(2)}, delta=${lw.delta.toFixed(3)}`);

  let inverted = true;
  try { C.invertMatrix(lw.covariance); } catch (e) { inverted = false; }
  ok("the shrunk matrix actually inverts, so the optimizer can run", inverted);
}

// ---------------------------------------------------------------- EWMA
section("Exponentially weighted covariance");
{
  const calm = generateReturns([0, 0], [[0.0001, 0], [0, 0.0001]], 300, 111);
  const turbulent = generateReturns([0, 0], [[0.0016, 0], [0, 0.0016]], 300, 222);
  const X = calm.concat(turbulent); // volatility rises in the second half

  const eq = D.sampleCovariance(X, true);
  const ew = D.ewmaCovariance(X, 0.94);
  ok("EWMA reacts to the recent volatility spike more than equal weighting",
    ew[0][0] > eq[0][0],
    `EWMA ${ew[0][0].toExponential(2)} vs equal-weight ${eq[0][0].toExponential(2)}`);
  ok("EWMA stays positive semi-definite",
    C.symmetricEigen(ew).values.every((v) => v > -1e-14));

  let threw = false;
  try { D.ewmaCovariance(X, 1.5); } catch (e) { threw = true; }
  ok("an invalid decay factor is rejected", threw);
}

// ---------------------------------------------------------------- annualization
section("Annualization");
{
  const mu = [0.0004];
  const cov = [[0.0001]];
  const a = D.annualize(mu, cov, 252, "log");
  ok("log mean scales linearly with periods per year", close(a.mu[0], 0.0004 * 252, 1e-12));
  ok("variance scales linearly, so volatility scales with the square root",
    close(a.vols[0], Math.sqrt(0.0001 * 252), 1e-12));

  const s = D.annualize([0.01], [[0.0001]], 12, "simple");
  ok("simple returns compound rather than scaling linearly",
    close(s.mu[0], Math.pow(1.01, 12) - 1, 1e-12));

  const two = D.annualize([0.001, 0.002], [[0.0001, 0.00005], [0.00005, 0.0004]], 252, "log");
  ok("correlation is preserved by annualization",
    close(two.corr[0][1], 0.00005 / Math.sqrt(0.0001 * 0.0004), 1e-12));
  ok("correlation diagonal is exactly 1", close(two.corr[0][0], 1, 1e-14));
}

// ---------------------------------------------------------------- data quality
section("Data quality checks");
{
  const names = ["A", "B"];
  const dates = Array.from({ length: 60 }, (_, i) => `2020-01-${i + 1}`);
  const prices = Array.from({ length: 60 }, (_, t) => [100 + t, 50]); // B never moves
  const rets = D.pricesToReturns(prices, "log").returns;
  const q = D.dataQualityReport(prices, names, dates, rets);
  ok("a never-changing price series is flagged as stale",
    q.warnings.some((w) => w.includes("B") && w.includes("unchanged")));
}
{
  const names = ["A"];
  const prices = [[100], [101], [102], [103], [104]];
  const rets = D.pricesToReturns(prices, "log").returns;
  const q = D.dataQualityReport(prices, names, ["a", "b", "c", "d", "e"], rets);
  ok("too few observations is flagged", q.warnings.some((w) => w.includes("observations")));
  ok("insufficiency is reported as a flag, not just prose", q.sufficient === false);
}
{
  const names = ["A"];
  const prices = [[100], [101], [50], [51]]; // looks like an unadjusted 2:1 split
  const rets = D.pricesToReturns(prices, "log").returns;
  const q = D.dataQualityReport(prices, names, ["a", "b", "c", "d"], rets);
  ok("a possible unadjusted split is flagged",
    q.warnings.some((w) => w.includes("split")));

  // The same -50% move on a 24/7 (365-period) market is not a split: crypto
  // has none, so telling the user to look for one would be misleading.
  const q365 = D.dataQualityReport(prices, names, ["a", "b", "c", "d"], rets, 365);
  const extreme365 = q365.warnings.filter((w) => w.includes("moves above 25%"));
  ok("on a 365-day market a large move is not blamed on stock splits",
    extreme365.length === 1 && !/split|dividend/i.test(extreme365[0]) && /crypto|24\/7/.test(extreme365[0]));
}

{
  // Short calendar history is the dominant source of bad expected returns,
  // and no amount of extra sampling frequency fixes it.
  const names = ["A", "B"];
  const T = 120; // roughly half a year of daily data
  const prices = Array.from({ length: T + 1 }, (_, t) => [100 * (1 + t * 0.001), 50 * (1 + t * 0.0005)]);
  const dates = Array.from({ length: T + 1 }, (_, t) =>
    new Date(Date.UTC(2020, 0, 1 + t)).toISOString().slice(0, 10));
  const rets = D.pricesToReturns(prices, "log").returns;
  const q = D.dataQualityReport(prices, names, dates, rets, 252);
  ok("a history shorter than three years is flagged",
    q.warnings.some((w) => w.includes("year") && w.includes("standard error")));
  ok("the short-history warning quantifies the uncertainty",
    q.warnings.some((w) => /\d+ percentage points/.test(w)));
  ok("short history marks the data as insufficient", q.sufficient === false);
  ok("the length of history is reported numerically", q.years > 0.4 && q.years < 0.6);
}
{
  const names = ["A"];
  const T = 252 * 6;
  const prices = Array.from({ length: T + 1 }, (_, t) => [100 * Math.exp(t * 0.0002)]);
  const dates = Array.from({ length: T + 1 }, (_, t) =>
    new Date(Date.UTC(2015, 0, 1 + t)).toISOString().slice(0, 10));
  const rets = D.pricesToReturns(prices, "log").returns;
  const q = D.dataQualityReport(prices, names, dates, rets, 252);
  ok("six years of history is not flagged as too short",
    !q.warnings.some((w) => w.includes("standard error")));
}

// ---------------------------------------------------------------- end to end
section("End to end: CSV in, optimizer-ready inputs out");
{
  const trueMu = [0.0005, 0.0002, 0.0004];
  const trueCov = [
    [0.00012, 0.00003, 0.00005],
    [0.00003, 0.00004, 0.00001],
    [0.00005, 0.00001, 0.00020],
  ];
  const rets = generateReturns(trueMu, trueCov, 1500, 5150);
  const prices = pricesFromReturns([100, 100, 100], rets);
  const start = new Date(2018, 0, 1);
  let csv = "Date,Equity,Bonds,Commodities\n";
  prices.forEach((row, t) => {
    const d = new Date(start.getTime() + t * 86400000).toISOString().slice(0, 10);
    csv += `${d},${row.map((v) => v.toFixed(6)).join(",")}\n`;
  });

  const est = D.estimateFromCSV(csv, { estimator: "ledoit-wolf" });
  ok("asset names survive the pipeline", est.names.join(",") === "Equity,Bonds,Commodities");
  ok("observation count is correct", est.observations === 1500);
  ok("annualized returns are in a plausible range",
    est.mu.every((m) => m > -1 && m < 2), `${est.mu.map((m) => (m * 100).toFixed(1) + "%")}`);
  ok("annualized volatilities are positive", est.vols.every((v) => v > 0));
  ok("correlation matrix is valid for the optimizer", C.validateCorrelation(est.corr).isPSD);
  ok("shrinkage intensity is reported to the user", est.shrinkage && est.shrinkage.delta >= 0);
  ok("frequency was inferred, not assumed", est.frequency.confident === true);

  // The real test: the output must actually drive the optimizer
  let optimized = null;
  try { optimized = C.maxSharpeLongOnly(est.mu, est.cov, 0.02); } catch (e) { /* leave null */ }
  ok("estimated inputs feed straight into the optimizer", optimized && optimized.weights);
  ok("resulting weights are a valid portfolio",
    optimized && close(optimized.weights.reduce((a, b) => a + b, 0), 1, 1e-8) &&
    optimized.weights.every((w) => w >= -1e-9));
  console.log(`      annualized: ${est.names.map((nm, i) => `${nm} ${(est.mu[i] * 100).toFixed(1)}%/${(est.vols[i] * 100).toFixed(1)}%`).join(", ")}`);
  console.log(`      Ledoit-Wolf delta chosen from data: ${est.shrinkage.delta.toFixed(4)}`);

  const sampleEst = D.estimateFromCSV(csv, { estimator: "sample" });
  ok("the sample estimator is selectable and reports no shrinkage",
    sampleEst.shrinkage === null);
  const ewmaEst = D.estimateFromCSV(csv, { estimator: "ewma", lambda: 0.97 });
  ok("the EWMA estimator is selectable", ewmaEst.cov.length === 3);
}

// ---------------------------------------------------------------- summary
console.log(`\n${"=".repeat(64)}`);
console.log(`${pass} passed, ${fail} failed`);
if (fail) { console.log("Failures:"); failures.forEach((f) => console.log(`  - ${f}`)); }
console.log("=".repeat(64));
process.exit(fail ? 1 : 0);
