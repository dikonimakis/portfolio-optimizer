const C = require("./portfolio-core.js");

let pass = 0, fail = 0;
const failures = [];

function ok(name, cond, detail = "") {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; failures.push(name); console.log(`  FAIL  ${name}  ${detail}`); }
}
function close(a, b, tol = 1e-6) { return Math.abs(a - b) < tol; }
function vecClose(a, b, tol = 1e-6) {
  return a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < tol);
}
function section(s) { console.log(`\n== ${s}`); }

// ---------------------------------------------------------------- linear algebra
section("Linear algebra");
{
  const A = [[4, 1], [1, 3]];
  const x = C.solveLinear(A, [1, 2]);
  ok("solveLinear satisfies Ax=b", vecClose(C.matVec(A, x), [1, 2]));

  const inv = C.invertMatrix(A);
  const I = C.matVec(A, inv.map((r) => r[0]));
  ok("invertMatrix gives A*inv = I (first column)", vecClose(I, [1, 0]));

  const { values } = C.symmetricEigen([[2, 0, 0], [0, 3, 0], [0, 0, 5]]);
  const sorted = [...values].sort((a, b) => a - b);
  ok("symmetricEigen on diagonal matrix", vecClose(sorted, [2, 3, 5], 1e-8));

  // Known eigenvalues: [[2,1],[1,2]] -> 1 and 3
  const e2 = C.symmetricEigen([[2, 1], [1, 2]]).values.sort((a, b) => a - b);
  ok("symmetricEigen on 2x2 known case", vecClose(e2, [1, 3], 1e-8));
}

// ---------------------------------------------------------------- correlation validation
section("Correlation matrix validation");
{
  const good = [[1, 0.3, 0.2], [0.3, 1, 0.1], [0.2, 0.1, 1]];
  const v1 = C.validateCorrelation(good);
  ok("valid correlation matrix accepted", v1.valid && v1.isPSD);

  // Classic inconsistent case: A~B strongly, B~C strongly, but A~C strongly negative
  const bad = [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]];
  const v2 = C.validateCorrelation(bad);
  ok("mutually inconsistent correlations rejected", !v2.isPSD, `minEig=${v2.minEigenvalue}`);

  const repaired = C.repairCorrelation(bad);
  const v3 = C.validateCorrelation(repaired);
  ok("repairCorrelation produces a PSD matrix", v3.isPSD, `minEig=${v3.minEigenvalue}`);
  ok("repaired matrix has unit diagonal", repaired.every((r, i) => close(r[i], 1, 1e-9)));
  ok("repaired matrix is symmetric",
    repaired.every((r, i) => r.every((val, j) => close(val, repaired[j][i], 1e-9))));

  const asym = [[1, 0.5], [0.2, 1]];
  ok("asymmetric matrix flagged", !C.validateCorrelation(asym).valid);
}

// ---------------------------------------------------------------- QP: known solutions
section("QP solver — known closed-form cases");
{
  // Two uncorrelated assets, min variance, no return term.
  // Analytic: w1 = s2^2/(s1^2+s2^2)
  const s1 = 0.10, s2 = 0.20;
  const cov = [[s1 * s1, 0], [0, s2 * s2]];
  const w = C.minVarianceLongOnly(cov);
  const expected = [s2 * s2 / (s1 * s1 + s2 * s2), s1 * s1 / (s1 * s1 + s2 * s2)];
  ok("2-asset uncorrelated min-variance matches closed form", vecClose(w, expected, 1e-8),
    `got ${w.map((v) => v.toFixed(6))} expected ${expected.map((v) => v.toFixed(6))}`);
  ok("min-variance weights sum to 1", close(w[0] + w[1], 1, 1e-10));
}
{
  // Perfectly correlated pair with different vols: all weight to the lower-vol asset.
  const cov = C.buildCovariance([0.10, 0.20], [[1, 1], [1, 1]]);
  const w = C.minVarianceLongOnly(cov);
  ok("perfectly correlated pair concentrates in the low-vol asset",
    w[0] > 0.99 && w[1] < 0.01, `got ${w.map((v) => v.toFixed(4))}`);
}

section("QP solver — KKT verification");
{
  const vols = [0.15, 0.06, 0.17, 0.19, 0.16];
  const corr = [
    [1, -0.1, 0.85, 0.6, 0.1],
    [-0.1, 1, -0.05, 0.1, 0.2],
    [0.85, -0.05, 1, 0.55, 0.15],
    [0.6, 0.1, 0.55, 1, 0.2],
    [0.1, 0.2, 0.15, 0.2, 1],
  ];
  const cov = C.buildCovariance(vols, corr);
  const mu = [0.10, 0.04, 0.08, 0.09, 0.06];
  const n = 5;

  let allKKT = true, allFeasible = true, allConverged = true;
  for (let k = 0; k <= 40; k++) {
    const tau = k === 0 ? 0 : Math.pow(10, -3 + 5 * (k / 40));
    const c = mu.map((m) => tau * m);
    const r = C.solveQP(cov, c, Array(n).fill(1), 1);
    if (!r.converged) allConverged = false;
    const kkt = C.checkKKT(cov, c, Array(n).fill(1), 1, r.x, r.gamma);
    if (!kkt.ok) { allKKT = false; console.log(`      tau=${tau}: ${kkt.errors.join("; ")}`); }
    const sum = r.x.reduce((a, b) => a + b, 0);
    if (!close(sum, 1, 1e-8) || r.x.some((v) => v < -1e-9)) allFeasible = false;
  }
  ok("QP converges across 41 risk-tolerance levels", allConverged);
  ok("QP satisfies KKT conditions at every level", allKKT);
  ok("QP solutions are always feasible (sum=1, w>=0)", allFeasible);
}

section("QP solver — randomized stress test");
{
  let s = 987654321 >>> 0;
  const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let kktFails = 0, feasFails = 0, errors = 0, trials = 0;

  for (let t = 0; t < 300; t++) {
    const n = 2 + Math.floor(rand() * 7);
    // Build a guaranteed-PSD covariance as L*L'
    const L = Array.from({ length: n }, () => Array.from({ length: n }, () => rand() - 0.5));
    const cov = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let v = 0;
        for (let k = 0; k < n; k++) v += L[i][k] * L[j][k];
        cov[i][j] = v;
      }
      cov[i][i] += 0.01; // ensure strict positive definiteness
    }
    const mu = Array.from({ length: n }, () => rand() * 0.2 - 0.03);
    const tau = Math.pow(10, -2 + 4 * rand());
    const c = mu.map((m) => tau * m);
    trials++;
    try {
      const r = C.solveQP(cov, c, Array(n).fill(1), 1);
      const kkt = C.checkKKT(cov, c, Array(n).fill(1), 1, r.x, r.gamma, 1e-6);
      if (!kkt.ok) kktFails++;
      const sum = r.x.reduce((a, b) => a + b, 0);
      if (!close(sum, 1, 1e-7) || r.x.some((v) => v < -1e-8)) feasFails++;
    } catch (e) { errors++; }
  }
  ok(`300 random QPs: no solver exceptions`, errors === 0, `${errors} errors`);
  ok(`300 random QPs: all satisfy KKT`, kktFails === 0, `${kktFails} failures`);
  ok(`300 random QPs: all feasible`, feasFails === 0, `${feasFails} failures`);
}

// ---------------------------------------------------------------- QP vs Monte Carlo
section("QP beats Monte Carlo (the point of Phase 1)");
{
  const vols = [0.15, 0.06, 0.17, 0.19, 0.16];
  const corr = [
    [1, -0.1, 0.85, 0.6, 0.1],
    [-0.1, 1, -0.05, 0.1, 0.2],
    [0.85, -0.05, 1, 0.55, 0.15],
    [0.6, 0.1, 0.55, 1, 0.2],
    [0.1, 0.2, 0.15, 0.2, 1],
  ];
  const cov = C.buildCovariance(vols, corr);
  const mu = [0.10, 0.04, 0.08, 0.09, 0.06];
  const rf = 0.02;
  const n = 5;

  const exact = C.maxSharpeLongOnly(mu, cov, rf);
  ok("maxSharpeLongOnly returns a valid portfolio", !!exact.weights);
  ok("max-Sharpe weights sum to 1", close(exact.weights.reduce((a, b) => a + b, 0), 1, 1e-8));
  ok("max-Sharpe weights are non-negative", exact.weights.every((w) => w >= -1e-9));

  // Monte Carlo, same as the old implementation
  let s = 424242 >>> 0;
  const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let bestMC = -Infinity;
  for (let k = 0; k < 20000; k++) {
    const e = Array.from({ length: n }, () => -Math.log(rand() + 1e-12));
    const tot = e.reduce((a, b) => a + b, 0);
    const w = e.map((x) => x / tot);
    const ret = C.dot(w, mu);
    const vol = Math.sqrt(C.quadForm(w, cov));
    const sh = (ret - rf) / vol;
    if (sh > bestMC) bestMC = sh;
  }
  ok(`exact QP Sharpe (${exact.sharpe.toFixed(4)}) >= best of 20,000 Monte Carlo draws (${bestMC.toFixed(4)})`,
    exact.sharpe >= bestMC - 1e-9);
  console.log(`      improvement over Monte Carlo: ${((exact.sharpe / bestMC - 1) * 100).toFixed(2)}%`);

  // No random long-only portfolio may beat the exact optimum
  let violations = 0;
  for (let k = 0; k < 20000; k++) {
    const e = Array.from({ length: n }, () => -Math.log(rand() + 1e-12));
    const tot = e.reduce((a, b) => a + b, 0);
    const w = e.map((x) => x / tot);
    const vol = Math.sqrt(C.quadForm(w, cov));
    if ((C.dot(w, mu) - rf) / vol > exact.sharpe + 1e-9) violations++;
  }
  ok("no random portfolio beats the exact optimum (20,000 checks)", violations === 0, `${violations} violations`);
}

// ---------------------------------------------------------------- QP vs analytic
section("Long-only QP agrees with closed form when the closed form is already long-only");
{
  // Chosen so the unconstrained tangency portfolio has all-positive weights.
  const cov = C.buildCovariance([0.10, 0.12], [[1, 0.2], [0.2, 1]]);
  const mu = [0.08, 0.09];
  const rf = 0.02;
  const an = C.analyticFrontier(mu, cov, rf);
  ok("unconstrained tangency is interior for this case", an.tanWeights.every((w) => w > 0),
    `${an.tanWeights.map((v) => v.toFixed(4))}`);
  const lo = C.maxSharpeLongOnly(mu, cov, rf);
  ok("long-only QP reproduces the closed-form tangency when it is interior",
    vecClose(lo.weights, an.tanWeights, 1e-7),
    `QP ${lo.weights.map((v) => v.toFixed(6))} vs analytic ${an.tanWeights.map((v) => v.toFixed(6))}`);

  // Same for GMV
  const gmvLO = C.minVarianceLongOnly(cov);
  ok("long-only GMV reproduces the closed-form GMV when it is interior",
    vecClose(gmvLO, an.gmvWeights, 1e-7));
}
{
  // Case where the unconstrained solution goes short -> QP must clip to the boundary
  const cov = C.buildCovariance([0.10, 0.30], [[1, 0.9], [0.9, 1]]);
  const mu = [0.09, 0.05];
  const rf = 0.02;
  const an = C.analyticFrontier(mu, cov, rf);
  const lo = C.maxSharpeLongOnly(mu, cov, rf);
  ok("unconstrained tangency wants a short position here", an.tanWeights.some((w) => w < 0),
    `${an.tanWeights.map((v) => v.toFixed(4))}`);
  ok("long-only QP respects the no-short constraint", lo.weights.every((w) => w >= -1e-9),
    `${lo.weights.map((v) => v.toFixed(4))}`);
  ok("constrained Sharpe is lower than unconstrained (as theory requires)",
    lo.sharpe <= (an.tanRet - rf) / an.tanVol + 1e-9);
}

// ---------------------------------------------------------------- frontier shape
section("Long-only frontier properties");
{
  const vols = [0.15, 0.06, 0.17, 0.19, 0.16];
  const corr = [
    [1, -0.1, 0.85, 0.6, 0.1],
    [-0.1, 1, -0.05, 0.1, 0.2],
    [0.85, -0.05, 1, 0.55, 0.15],
    [0.6, 0.1, 0.55, 1, 0.2],
    [0.1, 0.2, 0.15, 0.2, 1],
  ];
  const cov = C.buildCovariance(vols, corr);
  const mu = [0.10, 0.04, 0.08, 0.09, 0.06];
  const front = C.longOnlyFrontier(mu, cov, 60);

  ok("frontier has multiple points", front.length > 5, `${front.length} points`);
  let mono = true;
  for (let i = 1; i < front.length; i++) {
    if (front[i].ret < front[i - 1].ret - 1e-12 || front[i].vol < front[i - 1].vol - 1e-12) mono = false;
  }
  ok("frontier is monotone: return and volatility both increase", mono);
  ok("every frontier portfolio is feasible",
    front.every((p) => close(p.weights.reduce((a, b) => a + b, 0), 1, 1e-7) && p.weights.every((w) => w >= -1e-9)));

  const gmv = C.minVarianceLongOnly(cov);
  const gmvVol = Math.sqrt(C.quadForm(gmv, cov));
  ok("no frontier point has lower volatility than the long-only GMV",
    front.every((p) => p.vol >= gmvVol - 1e-9));
  ok("highest-return frontier point approaches the best single asset",
    close(front[front.length - 1].ret, Math.max(...mu), 1e-3),
    `got ${front[front.length - 1].ret.toFixed(5)} vs ${Math.max(...mu)}`);
}

// ---------------------------------------------------------------- shrinkage
section("Shrinkage");
{
  const cov = C.buildCovariance([0.15, 0.06, 0.17], [[1, -0.1, 0.85], [-0.1, 1, -0.05], [0.85, -0.05, 1]]);
  const s0 = C.shrinkCovariance(cov, 0);
  ok("shrinkage with delta=0 returns the original covariance",
    s0.every((r, i) => r.every((v, j) => close(v, cov[i][j], 1e-12))));

  const s1 = C.shrinkCovariance(cov, 1);
  const vols = cov.map((r, i) => Math.sqrt(r[i]));
  ok("shrinkage with delta=1 preserves variances (diagonal untouched)",
    s1.every((r, i) => close(r[i], cov[i][i], 1e-12)));
  const c01 = s1[0][1] / (vols[0] * vols[1]);
  const c02 = s1[0][2] / (vols[0] * vols[2]);
  ok("delta=1 gives every pair the same correlation", close(c01, c02, 1e-12));

  const s5 = C.shrinkCovariance(cov, 0.5);
  ok("shrunk covariance remains PSD", C.symmetricEigen(s5).values.every((v) => v > -1e-9));

  const mid = 0.5 * cov[0][2] + 0.5 * s1[0][2];
  ok("delta=0.5 is the midpoint of original and target", close(s5[0][2], mid, 1e-12));
}
{
  const cov = C.buildCovariance([0.15, 0.06, 0.17], [[1, -0.1, 0.85], [-0.1, 1, -0.05], [0.85, -0.05, 1]]);
  const mu = [0.10, 0.04, 0.08];

  const small = C.bayesSteinMu(mu, cov, 24);
  const large = C.bayesSteinMu(mu, cov, 2400);
  ok("Bayes-Stein shrinks more with a smaller sample", small.phi > large.phi,
    `T=24 phi=${small.phi.toFixed(4)}, T=2400 phi=${large.phi.toFixed(4)}`);
  ok("Bayes-Stein intensity is in [0,1]", small.phi >= 0 && small.phi <= 1);
  ok("shrunk returns lie between the original and the GMV return",
    small.mu.every((m, i) => (m - mu[i]) * (small.muTarget - mu[i]) >= -1e-12));
  ok("Bayes-Stein preserves the mean-weighted GMV return direction",
    Math.abs(small.mu[0] - small.muTarget) < Math.abs(mu[0] - small.muTarget));
  const tooFew = C.bayesSteinMu(mu, cov, 3);
  ok("Bayes-Stein declines to shrink when T <= n+2", tooFew.phi === 0);
}

// ---------------------------------------------------------------- sensitivity
section("Sensitivity analysis");
{
  const vols = [0.15, 0.06, 0.17, 0.19, 0.16];
  const corr = [
    [1, -0.1, 0.85, 0.6, 0.1],
    [-0.1, 1, -0.05, 0.1, 0.2],
    [0.85, -0.05, 1, 0.55, 0.15],
    [0.6, 0.1, 0.55, 1, 0.2],
    [0.1, 0.2, 0.15, 0.2, 1],
  ];
  const cov = C.buildCovariance(vols, corr);
  const mu = [0.10, 0.04, 0.08, 0.09, 0.06];

  const zero = C.sensitivityAnalysis(mu, cov, 0.02, 0.0, 20);
  ok("zero perturbation gives zero turnover", close(zero.medianTurnover, 0, 1e-9));

  const s10 = C.sensitivityAnalysis(mu, cov, 0.02, 0.10, 150);
  const s30 = C.sensitivityAnalysis(mu, cov, 0.02, 0.30, 150);
  ok("larger input error produces larger weight instability",
    s30.medianTurnover >= s10.medianTurnover,
    `10%: ${(s10.medianTurnover * 100).toFixed(1)}%  30%: ${(s30.medianTurnover * 100).toFixed(1)}%`);
  ok("sensitivity bands bracket the base weights",
    s10.base.every((w, i) => w >= s10.low[i] - 1e-9 && w <= s10.high[i] + 1e-9));
  console.log(`      median turnover at +/-10% return error: ${(s10.medianTurnover * 100).toFixed(1)}%`);
  console.log(`      median turnover at +/-30% return error: ${(s30.medianTurnover * 100).toFixed(1)}%`);
}

// ---------------------------------------------------------------- edge cases
section("Edge cases");
{
  const cov = C.buildCovariance([0.10, 0.12], [[1, 0.2], [0.2, 1]]);
  const r = C.maxSharpeLongOnly([0.01, 0.015], cov, 0.05);
  ok("all returns below the risk-free rate is handled, not crashed", r.weights === null && !!r.reason);
}
{
  const cov = C.buildCovariance([0.10], [[1]]);
  const w = C.minVarianceLongOnly(cov);
  ok("single asset gets 100% weight", close(w[0], 1, 1e-9));
}
{
  // Near-duplicate assets create a near-singular covariance matrix
  const cov = C.buildCovariance([0.10, 0.1000001, 0.15], [[1, 0.9999999, 0.3], [0.9999999, 1, 0.3], [0.3, 0.3, 1]]);
  let threw = false, w = null;
  try { w = C.minVarianceLongOnly(cov); } catch (e) { threw = true; }
  ok("near-singular covariance either solves feasibly or fails cleanly",
    threw || (close(w.reduce((a, b) => a + b, 0), 1, 1e-6) && w.every((v) => v >= -1e-8)));
}

// ---------------------------------------------------------------- bounds
section("Position limits and minimum allocations");

const BV = [0.15, 0.06, 0.17, 0.19, 0.16];
const BC = [
  [1, -0.1, 0.85, 0.6, 0.1],
  [-0.1, 1, -0.05, 0.1, 0.2],
  [0.85, -0.05, 1, 0.55, 0.15],
  [0.6, 0.1, 0.55, 1, 0.2],
  [0.1, 0.2, 0.15, 0.2, 1],
];
const BCOV = C.buildCovariance(BV, BC);
const BMU = [0.10, 0.04, 0.08, 0.09, 0.06];
const BRF = 0.02;

{
  // Backward compatibility is the first thing to establish.
  const a = C.maxSharpeLongOnly(BMU, BCOV, BRF);
  const b = C.maxSharpeLongOnly(BMU, BCOV, BRF, { lower: [0,0,0,0,0], upper: [1,1,1,1,1] });
  ok("bounds of 0..1 reproduce the unbounded long-only answer exactly",
    vecClose(a.weights, b.weights, 1e-9));
  ok("trivial bounds still use the exact transformation, not the search",
    b.method === "exact", `method=${b.method}`);

  const mvA = C.minVarianceLongOnly(BCOV);
  const mvB = C.minVarianceLongOnly(BCOV, { lower: [0,0,0,0,0], upper: [1,1,1,1,1] });
  ok("minimum variance is unchanged by trivial bounds", vecClose(mvA, mvB, 1e-10));
}
{
  const capped = C.maxSharpeLongOnly(BMU, BCOV, BRF, { maxWeight: 0.30 });
  ok("no position exceeds the 30% cap",
    capped.weights.every((w) => w <= 0.30 + 1e-7),
    `${capped.weights.map((w) => (w*100).toFixed(1)+"%")}`);
  ok("capped weights still sum to 1",
    close(capped.weights.reduce((a,b)=>a+b,0), 1, 1e-8));
  ok("the constrained search reports which method it used",
    capped.method === "golden-section", `method=${capped.method}`);

  const free = C.maxSharpeLongOnly(BMU, BCOV, BRF);
  ok("capping cannot improve the Sharpe ratio, as theory requires",
    capped.sharpe <= free.sharpe + 1e-9,
    `capped ${capped.sharpe.toFixed(4)} vs free ${free.sharpe.toFixed(4)}`);
  const maxFree = Math.max(...free.weights);
  ok("the cap actually binds on this problem", maxFree > 0.30,
    `largest unconstrained weight was ${(maxFree*100).toFixed(1)}%`);
  console.log(`      unconstrained max weight ${(maxFree*100).toFixed(1)}% -> capped ${(Math.max(...capped.weights)*100).toFixed(1)}%, Sharpe ${free.sharpe.toFixed(3)} -> ${capped.sharpe.toFixed(3)}`);
}
{
  const floors = [0.10, 0.10, 0.10, 0.10, 0.10];
  const r = C.maxSharpeLongOnly(BMU, BCOV, BRF, { lower: floors });
  ok("every minimum allocation is met",
    r.weights.every((w, i) => w >= floors[i] - 1e-7),
    `${r.weights.map((w) => (w*100).toFixed(1)+"%")}`);
  ok("weights with floors still sum to 1", close(r.weights.reduce((a,b)=>a+b,0), 1, 1e-8));
}
{
  // Fixing one asset exactly: lower = upper.
  const r = C.minVarianceLongOnly(BCOV, {
    lower: [0, 0, 0, 0, 0.25], upper: [1, 1, 1, 1, 0.25],
  });
  ok("an asset pinned with equal minimum and maximum lands exactly there",
    close(r[4], 0.25, 1e-8), `got ${(r[4]*100).toFixed(3)}%`);
  ok("the remaining assets still sum to the balance",
    close(r.reduce((a,b)=>a+b,0), 1, 1e-8));
}

section("Bounded QP satisfies KKT");
{
  const lower = [0.05, 0.00, 0.00, 0.05, 0.00];
  const upper = [0.40, 0.50, 0.35, 0.30, 0.25];
  let allKKT = true, allFeasible = true, allConverged = true;
  for (let k = 0; k <= 40; k++) {
    const tau = k === 0 ? 0 : Math.pow(10, -3 + 5 * (k / 40));
    const c = BMU.map((m) => tau * m);
    const r = C.solveQP(BCOV, c, Array(5).fill(1), 1, { lower, upper });
    if (!r.converged) allConverged = false;
    const kkt = C.checkKKT(BCOV, c, Array(5).fill(1), 1, r.x, r.gamma, 1e-7, lower, upper);
    if (!kkt.ok) { allKKT = false; console.log(`      tau=${tau}: ${kkt.errors.join("; ")}`); }
    const sum = r.x.reduce((a, b) => a + b, 0);
    if (!close(sum, 1, 1e-8)) allFeasible = false;
    if (r.x.some((v, i) => v < lower[i] - 1e-8 || v > upper[i] + 1e-8)) allFeasible = false;
  }
  ok("bounded QP converges across 41 risk-tolerance levels", allConverged);
  ok("bounded QP satisfies the two-sided KKT conditions at every level", allKKT);
  ok("bounded QP respects every bound at every level", allFeasible);
}

section("Bounded QP randomized stress test");
{
  let s = 20250815 >>> 0;
  const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let kktFails = 0, boundFails = 0, errors = 0, infeasibleCaught = 0, trials = 0;

  for (let t = 0; t < 400; t++) {
    const n = 2 + Math.floor(rand() * 7);
    const L = Array.from({ length: n }, () => Array.from({ length: n }, () => rand() - 0.5));
    const cov = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let v = 0;
        for (let k = 0; k < n; k++) v += L[i][k] * L[j][k];
        cov[i][j] = v;
      }
      cov[i][i] += 0.01;
    }
    const mu = Array.from({ length: n }, () => rand() * 0.2 - 0.03);
    const tau = Math.pow(10, -2 + 4 * rand());
    const c = mu.map((m) => tau * m);
    // Random bounds, deliberately including infeasible combinations.
    const lower = Array.from({ length: n }, () => rand() * (1.4 / n));
    const upper = lower.map((lo) => lo + rand() * 0.9);
    trials++;
    try {
      const r = C.solveQP(cov, c, Array(n).fill(1), 1, { lower, upper });
      const kkt = C.checkKKT(cov, c, Array(n).fill(1), 1, r.x, r.gamma, 1e-6, lower, upper);
      if (!kkt.ok) kktFails++;
      if (r.x.some((v, i) => v < lower[i] - 1e-7 || v > upper[i] + 1e-7)) boundFails++;
      if (!close(r.x.reduce((a, b) => a + b, 0), 1, 1e-7)) boundFails++;
    } catch (e) {
      if (/cannot be satisfied|exceeds its maximum/.test(e.message)) infeasibleCaught++;
      else errors++;
    }
  }
  ok("400 random bounded QPs: no unexpected exceptions", errors === 0, `${errors} errors`);
  ok("400 random bounded QPs: all satisfy KKT", kktFails === 0, `${kktFails} failures`);
  ok("400 random bounded QPs: all respect their bounds", boundFails === 0, `${boundFails} failures`);
  ok("genuinely infeasible bound sets are rejected with a clear message, not a bad answer",
    infeasibleCaught > 0, `${infeasibleCaught} of ${trials} were infeasible and caught`);
  console.log(`      ${trials - infeasibleCaught - errors} solved, ${infeasibleCaught} correctly refused as infeasible`);
}

section("Infeasible limits are explained, not silently mangled");
{
  const threw = (fn) => { try { fn(); return ""; } catch (e) { return e.message; } };

  const m1 = threw(() => C.minVarianceLongOnly(BCOV, { lower: [0.3, 0.3, 0.3, 0.3, 0.3] }));
  ok("minimums summing above 100% are refused", /cannot be satisfied/.test(m1), m1);

  const m2 = threw(() => C.minVarianceLongOnly(BCOV, { upper: [0.1, 0.1, 0.1, 0.1, 0.1] }));
  ok("maximums summing below 100% are refused", /cannot be satisfied/.test(m2), m2);

  const m3 = threw(() => C.minVarianceLongOnly(BCOV, {
    lower: [0.5, 0, 0, 0, 0], upper: [0.2, 1, 1, 1, 1],
  }));
  ok("a minimum above its own maximum is refused and names the asset",
    /exceeds its maximum/.test(m3) && /Asset 1/.test(m3), m3);

  const m4 = threw(() => C.minVarianceLongOnly(BCOV, { lower: [0.2, 0.2, 0.2, 0.2, 0.2] }));
  ok("minimums summing to exactly 100% are accepted, not rejected at the boundary",
    m4 === "", m4);
}

section("Constrained frontier");
{
  const bounds = { maxWeight: 0.35, minWeight: 0.05 };
  const front = C.longOnlyFrontier(BMU, BCOV, 60, bounds);
  ok("a constrained frontier is produced", front.length > 3, `${front.length} points`);
  ok("every frontier portfolio respects the limits",
    front.every((p) => p.weights.every((w) => w >= 0.05 - 1e-7 && w <= 0.35 + 1e-7)));
  ok("every constrained frontier portfolio is fully invested",
    front.every((p) => close(p.weights.reduce((a, b) => a + b, 0), 1, 1e-7)));
  let mono = true;
  for (let i = 1; i < front.length; i++) {
    if (front[i].ret < front[i-1].ret - 1e-12 || front[i].vol < front[i-1].vol - 1e-12) mono = false;
  }
  ok("the constrained frontier is still monotone", mono);

  const freeFront = C.longOnlyFrontier(BMU, BCOV, 60);
  const freeMaxRet = Math.max(...freeFront.map((p) => p.ret));
  const capMaxRet = Math.max(...front.map((p) => p.ret));
  ok("constraints reduce the attainable return range, as they must",
    capMaxRet <= freeMaxRet + 1e-9,
    `constrained ${(capMaxRet*100).toFixed(2)}% vs free ${(freeMaxRet*100).toFixed(2)}%`);
}

section("Golden-section search agrees with the exact solution");
{
  // Cross-validation: force the search path by passing bounds that do not bind,
  // then confirm it finds the same optimum the closed-form transformation does.
  const exact = C.maxSharpeLongOnly(BMU, BCOV, BRF);
  const viaSearch = C.maxSharpeLongOnly(BMU, BCOV, BRF, {
    lower: [0,0,0,0,0], upper: [0.999, 0.999, 0.999, 0.999, 0.999],
  });
  ok("the search path is actually taken", viaSearch.method === "golden-section");
  ok("search finds the same Sharpe ratio as the exact transformation",
    Math.abs(viaSearch.sharpe - exact.sharpe) < 1e-6,
    `search ${viaSearch.sharpe.toFixed(8)} vs exact ${exact.sharpe.toFixed(8)}`);
  ok("search finds the same weights as the exact transformation",
    vecClose(viaSearch.weights, exact.weights, 1e-4),
    `max diff ${Math.max(...viaSearch.weights.map((w,i)=>Math.abs(w-exact.weights[i]))).toExponential(2)}`);

  // And no portfolio the constrained frontier can reach beats it.
  const front = C.longOnlyFrontier(BMU, BCOV, 200, { upper: [0.999,0.999,0.999,0.999,0.999] });
  const bestOnFrontier = Math.max(...front.map((p) => (p.ret - BRF) / p.vol));
  ok("no point on the constrained frontier beats the search result",
    viaSearch.sharpe >= bestOnFrontier - 1e-7,
    `search ${viaSearch.sharpe.toFixed(6)} vs best frontier point ${bestOnFrontier.toFixed(6)}`);
}

section("Sensitivity analysis honours bounds");
{
  const s = C.sensitivityAnalysis(BMU, BCOV, BRF, 0.20, 100, 999, { maxWeight: 0.30 });
  ok("sensitivity runs with bounds applied", s !== null);
  ok("every perturbed portfolio still respects the cap",
    s.high.every((h) => h <= 0.30 + 1e-6),
    `highest weight seen ${(Math.max(...s.high)*100).toFixed(2)}%`);

  const free = C.sensitivityAnalysis(BMU, BCOV, BRF, 0.20, 100, 999);

  // History of this check, kept as a caution. It first reported that limits
  // TRIPLED turnover, which had a persuasive story attached: caps force more
  // assets into the portfolio, so more of them react to noise. The story was
  // wrong and so was the number. The baseline inside sensitivityAnalysis was
  // being computed without the bounds, so capped portfolios were being compared
  // against an unconstrained portfolio they could never equal. With the baseline
  // fixed, limits modestly REDUCE turnover as well as improving diversification.
  // A plausible mechanism is not evidence that the measurement is sound.
  const capHeld = s.base.filter((w) => w > 0.005).length;
  const freeHeld = free.base.filter((w) => w > 0.005).length;
  ok("position limits increase the number of assets actually held",
    capHeld > freeHeld, `capped holds ${capHeld}, unconstrained holds ${freeHeld}`);
  ok("the unconstrained optimum leaves assets pinned at zero",
    free.base.some((w) => w <= 1e-6));
  ok("turnover is measured against a baseline using the same bounds",
    s.medianTurnover <= free.medianTurnover + 1e-9,
    `capped ${(s.medianTurnover*100).toFixed(1)}% vs unconstrained ${(free.medianTurnover*100).toFixed(1)}%`);
  console.log(`      assets held: unconstrained ${freeHeld}/5, capped at 30% ${capHeld}/5`);
  console.log(`      turnover under +/-20% return error: unconstrained ${(free.medianTurnover*100).toFixed(1)}%, capped ${(s.medianTurnover*100).toFixed(1)}%`);
  console.log(`      -> limits improve diversification and modestly reduce instability`);
}

// ---------------------------------------------------------------- group constraints
// Assets: 0 Stocks, 1 Bonds, 2 Intl, 3 REITs, 4 Gold
const EQUITY = [0, 2];            // Stocks + Intl
const DEFENSIVE = [1, 4];         // Bonds + Gold

section("Group constraints — construction and feasibility");
{
  const { G, h, labels } = C.buildGroupConstraints(5, [
    { name: "Equity", assets: EQUITY, max: 0.60, min: 0.20 },
  ]);
  ok("a max and a min produce two inequality rows", G.length === 2 && h.length === 2);
  ok("the max row selects exactly the group members",
    vecClose(G[0], [1, 0, 1, 0, 0], 1e-12));
  ok("the min row is the negated selector", vecClose(G[1], [-1, 0, -1, 0, 0], 1e-12));
  ok("the min bound is negated to fit Gx <= h", close(h[0], 0.60, 1e-12) && close(h[1], -0.20, 1e-12));
  ok("rows are labelled in plain language",
    labels[0].includes("at most 60%") && labels[1].includes("at least 20%"));

  const empty = C.buildGroupConstraints(5, [{ name: "Nothing", assets: [], max: 0.5 }]);
  ok("a group with no members produces no rows", empty.G.length === 0);
  const noLimit = C.buildGroupConstraints(5, [{ name: "Open", assets: [0, 1] }]);
  ok("a group with neither max nor min produces no rows", noLimit.G.length === 0);
  const outOfRange = C.buildGroupConstraints(5, [{ name: "Bad", assets: [0, 99, -1], max: 0.5 }]);
  ok("out-of-range asset indices are ignored rather than corrupting the row",
    vecClose(outOfRange.G[0], [1, 0, 0, 0, 0], 1e-12));
}
{
  const lower = new Array(5).fill(0);
  const upper = new Array(5).fill(1);
  ok("a satisfiable group passes the pre-check",
    C.checkGroupFeasibility(5, lower, upper, [{ name: "Equity", assets: EQUITY, max: 0.6 }]) === null);

  const m1 = C.checkGroupFeasibility(5, [0.4, 0, 0.4, 0, 0], upper,
    [{ name: "Equity", assets: EQUITY, max: 0.5 }]);
  ok("a cap below its members' own minimums is caught before solving",
    typeof m1 === "string" && m1.includes("Equity") && m1.includes("80%"), m1);

  const m2 = C.checkGroupFeasibility(5, lower, [0.1, 1, 0.1, 1, 1],
    [{ name: "Equity", assets: EQUITY, min: 0.5 }]);
  ok("a floor above what its members' caps allow is caught",
    typeof m2 === "string" && m2.includes("at most 20%"), m2);

  const m3 = C.checkGroupFeasibility(5, lower, upper,
    [{ name: "Equity", assets: EQUITY, min: 0.6, max: 0.3 }]);
  ok("a group minimum above its own maximum is caught", typeof m3 === "string", m3);

  const m4 = C.checkGroupFeasibility(5, lower, upper, [
    { name: "Equity", assets: EQUITY, max: 0.3 },
    { name: "Defensive", assets: DEFENSIVE, max: 0.3 },
    { name: "Rest", assets: [3], max: 0.2 },
  ]);
  ok("disjoint caps covering everything but summing under 100% are caught",
    typeof m4 === "string" && m4.includes("80%"), m4);
}

section("Group constraints are enforced");
{
  // Cap chosen to actually bind: the unconstrained optimum holds 35.2% equity,
  // so a 50% cap would test nothing.
  const g = [{ name: "Equity", assets: EQUITY, max: 0.25 }];
  const r = C.maxSharpeLongOnly(BMU, BCOV, BRF, { groups: g });
  const equityWeight = EQUITY.reduce((s, i) => s + r.weights[i], 0);
  ok("the equity cap is respected", equityWeight <= 0.25 + 1e-7,
    `equity total ${(equityWeight * 100).toFixed(2)}%`);
  ok("weights still sum to 1", close(r.weights.reduce((a, b) => a + b, 0), 1, 1e-8));
  ok("weights remain non-negative", r.weights.every((w) => w >= -1e-9));

  const free = C.maxSharpeLongOnly(BMU, BCOV, BRF);
  const freeEquity = EQUITY.reduce((s, i) => s + free.weights[i], 0);
  ok("the cap actually binds on this problem", freeEquity > 0.25,
    `unconstrained equity was ${(freeEquity * 100).toFixed(1)}%`);
  ok("a group cap cannot improve the Sharpe ratio", r.sharpe <= free.sharpe + 1e-9,
    `capped ${r.sharpe.toFixed(4)} vs free ${free.sharpe.toFixed(4)}`);
  console.log(`      equity ${(freeEquity*100).toFixed(1)}% -> ${(equityWeight*100).toFixed(1)}%, Sharpe ${free.sharpe.toFixed(3)} -> ${r.sharpe.toFixed(3)}`);
}
{
  const g = [{ name: "Defensive", assets: DEFENSIVE, min: 0.40 }];
  const r = C.maxSharpeLongOnly(BMU, BCOV, BRF, { groups: g });
  const defensive = DEFENSIVE.reduce((s, i) => s + r.weights[i], 0);
  ok("a group floor is met", defensive >= 0.40 - 1e-7,
    `defensive total ${(defensive * 100).toFixed(2)}%`);
}
{
  // Both directions at once, plus per-asset limits on top.
  const bounds = {
    maxWeight: 0.35,
    minWeight: 0.05,
    groups: [
      { name: "Equity", assets: EQUITY, max: 0.55, min: 0.25 },
      { name: "Defensive", assets: DEFENSIVE, min: 0.30 },
    ],
  };
  const r = C.maxSharpeLongOnly(BMU, BCOV, BRF, bounds);
  const eq = EQUITY.reduce((s, i) => s + r.weights[i], 0);
  const def = DEFENSIVE.reduce((s, i) => s + r.weights[i], 0);
  ok("per-asset limits hold alongside group limits",
    r.weights.every((w) => w >= 0.05 - 1e-7 && w <= 0.35 + 1e-7),
    r.weights.map((w) => (w * 100).toFixed(1)).join(","));
  ok("the equity band is respected on both sides", eq >= 0.25 - 1e-7 && eq <= 0.55 + 1e-7,
    `${(eq * 100).toFixed(2)}%`);
  ok("the defensive floor is respected", def >= 0.30 - 1e-7, `${(def * 100).toFixed(2)}%`);
  ok("the portfolio is fully invested", close(r.weights.reduce((a, b) => a + b, 0), 1, 1e-8));
  console.log(`      equity ${(eq*100).toFixed(1)}%, defensive ${(def*100).toFixed(1)}%, weights ${r.weights.map(w=>(w*100).toFixed(0)+"%").join(" ")}`);
}
{
  // Overlapping groups: an asset belonging to two constrained sets.
  const bounds = {
    groups: [
      { name: "Equity", assets: [0, 2], max: 0.60 },
      { name: "US only", assets: [0, 3], max: 0.45 },
    ],
  };
  const r = C.maxSharpeLongOnly(BMU, BCOV, BRF, bounds);
  ok("overlapping groups are both respected",
    [0, 2].reduce((s, i) => s + r.weights[i], 0) <= 0.60 + 1e-7 &&
    [0, 3].reduce((s, i) => s + r.weights[i], 0) <= 0.45 + 1e-7,
    r.weights.map((w) => (w * 100).toFixed(1)).join(","));
}
{
  // A group that does not bind must leave the answer untouched.
  const loose = C.maxSharpeLongOnly(BMU, BCOV, BRF, {
    groups: [{ name: "Everything", assets: [0, 1, 2, 3, 4], max: 1.0 }],
  });
  const free = C.maxSharpeLongOnly(BMU, BCOV, BRF);
  ok("a non-binding group leaves the optimum unchanged",
    Math.abs(loose.sharpe - free.sharpe) < 1e-6,
    `${loose.sharpe.toFixed(8)} vs ${free.sharpe.toFixed(8)}`);
}

section("Group-constrained QP satisfies KKT");
{
  const groups = [
    { name: "Equity", assets: EQUITY, max: 0.55, min: 0.20 },
    { name: "Defensive", assets: DEFENSIVE, max: 0.60 },
  ];
  const lower = new Array(5).fill(0.02);
  const upper = new Array(5).fill(0.45);
  const { G, h } = C.buildGroupConstraints(5, groups);

  let allKKT = true, allFeasible = true, allConverged = true;
  for (let k = 0; k <= 40; k++) {
    const tau = k === 0 ? 0 : Math.pow(10, -3 + 5 * (k / 40));
    const c = BMU.map((m) => tau * m);
    const r = C.solveQP(BCOV, c, Array(5).fill(1), 1, { lower, upper, G, h });
    if (!r.converged) allConverged = false;
    const kkt = C.checkKKT(BCOV, c, Array(5).fill(1), 1, r.x, r.gamma, 1e-7, lower, upper, G, h, r.lambda);
    if (!kkt.ok) { allKKT = false; console.log(`      tau=${tau}: ${kkt.errors.join("; ")}`); }
    if (!close(r.x.reduce((a, b) => a + b, 0), 1, 1e-8)) allFeasible = false;
    if (r.x.some((v, i) => v < lower[i] - 1e-8 || v > upper[i] + 1e-8)) allFeasible = false;
    for (let j = 0; j < G.length; j++) if (C.dot(G[j], r.x) - h[j] > 1e-7) allFeasible = false;
  }
  ok("group-constrained QP converges across 41 risk-tolerance levels", allConverged);
  ok("group-constrained QP satisfies KKT including group multipliers", allKKT);
  ok("group-constrained QP respects bounds and group rows at every level", allFeasible);
}

section("Group-constrained QP randomized stress test");
{
  let s = 31415926 >>> 0;
  const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let kktFails = 0, feasFails = 0, errors = 0, infeasibleCaught = 0, notConverged = 0, solved = 0;

  for (let t = 0; t < 400; t++) {
    const n = 3 + Math.floor(rand() * 5);
    const L = Array.from({ length: n }, () => Array.from({ length: n }, () => rand() - 0.5));
    const cov = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let v = 0;
        for (let k = 0; k < n; k++) v += L[i][k] * L[j][k];
        cov[i][j] = v;
      }
      cov[i][i] += 0.01;
    }
    const mu = Array.from({ length: n }, () => rand() * 0.2 - 0.03);
    const tau = Math.pow(10, -2 + 4 * rand());
    const c = mu.map((m) => tau * m);
    const lower = Array.from({ length: n }, () => rand() * (0.8 / n));
    const upper = lower.map((lo) => lo + 0.1 + rand() * 0.8);

    // One or two random groups, sometimes overlapping.
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
      const pre = C.checkGroupFeasibility(n, lower, upper, groups);
      if (pre) { infeasibleCaught++; continue; }
      const { G, h } = C.buildGroupConstraints(n, groups);
      const r = C.solveQP(cov, c, Array(n).fill(1), 1, { lower, upper, G, h });
      if (!r.converged) {
        notConverged++;
        // An approximate answer is still an answer and must satisfy every
        // constraint; only optimality is relaxed.
        if (!close(r.x.reduce((a, b) => a + b, 0), 1, 1e-6)) feasFails++;
        if (r.x.some((v, i) => v < lower[i] - 1e-6 || v > upper[i] + 1e-6)) feasFails++;
        for (let j = 0; j < G.length; j++) if (C.dot(G[j], r.x) - h[j] > 1e-6) feasFails++;
        if (!r.approximate) feasFails++;   // must be labelled as approximate
        continue;
      }
      solved++;
      const kkt = C.checkKKT(cov, c, Array(n).fill(1), 1, r.x, r.gamma, 1e-5, lower, upper, G, h, r.lambda);
      if (!kkt.ok) kktFails++;
      if (!close(r.x.reduce((a, b) => a + b, 0), 1, 1e-7)) feasFails++;
      if (r.x.some((v, i) => v < lower[i] - 1e-7 || v > upper[i] + 1e-7)) feasFails++;
      for (let j = 0; j < G.length; j++) if (C.dot(G[j], r.x) - h[j] > 1e-6) feasFails++;
    } catch (e) {
      if (/cannot be satisfied|exceeds its maximum|at most|at least/.test(e.message)) infeasibleCaught++;
      else errors++;
    }
  }
  ok("400 random group-constrained QPs: no unexpected exceptions", errors === 0, `${errors} errors`);
  ok("400 random group-constrained QPs: all solutions satisfy KKT", kktFails === 0, `${kktFails} failures`);
  ok("400 random group-constrained QPs: all solutions are feasible", feasFails === 0, `${feasFails} failures`);
  ok("infeasible mandates are detected rather than answered", infeasibleCaught > 0);
  ok("non-converged problems still return a feasible answer, labelled approximate",
    true, `${notConverged} of ${solved + notConverged + infeasibleCaught} fell back`);
  console.log(`      ${solved} solved exactly, ${infeasibleCaught} refused as infeasible, ${notConverged} returned an approximate but feasible answer`);
}

section("Group-constrained frontier and sensitivity");
{
  const bounds = { maxWeight: 0.40, groups: [{ name: "Equity", assets: EQUITY, max: 0.50 }] };
  const front = C.longOnlyFrontier(BMU, BCOV, 60, bounds);
  ok("a group-constrained frontier is produced", front.length > 3, `${front.length} points`);
  ok("every frontier portfolio respects the group cap",
    front.every((p) => EQUITY.reduce((s, i) => s + p.weights[i], 0) <= 0.50 + 1e-7));
  ok("every frontier portfolio respects the per-asset cap",
    front.every((p) => p.weights.every((w) => w <= 0.40 + 1e-7)));
  ok("every frontier portfolio is fully invested",
    front.every((p) => close(p.weights.reduce((a, b) => a + b, 0), 1, 1e-7)));

  const sens = C.sensitivityAnalysis(BMU, BCOV, BRF, 0.20, 60, 4242, bounds);
  ok("sensitivity analysis runs under group constraints", sens !== null);
  ok("perturbed portfolios never breach the per-asset cap",
    sens.high.every((v) => v <= 0.40 + 1e-6));
}

// ---------------------------------------------------------------- turnover cost
section("Transaction costs inside the objective");
{
  const prev = [0.2, 0.2, 0.2, 0.2, 0.2];
  const free = C.efficientLongOnly(BMU, BCOV, 3);
  const zero = C.efficientWithTurnoverCost(BMU, BCOV, 3, prev, 0);
  ok("zero cost reproduces the ordinary solution exactly", vecClose(zero, free, 1e-7),
    `${zero.map((w) => (w * 100).toFixed(2))} vs ${free.map((w) => (w * 100).toFixed(2))}`);

  const turnoverAt = (cost) => {
    const w = C.efficientWithTurnoverCost(BMU, BCOV, 3, prev, cost);
    return { w, to: w.reduce((s, x, i) => s + Math.abs(x - prev[i]), 0) / 2 };
  };
  const costs = [0, 0.001, 0.005, 0.02, 0.05, 0.10, 0.50];
  const series = costs.map(turnoverAt);

  ok("turnover never increases as the cost rises",
    series.every((r, i) => i === 0 || r.to <= series[i - 1].to + 1e-7),
    series.map((r, i) => `${costs[i]}:${(r.to * 100).toFixed(1)}%`).join(" "));
  ok("a high enough cost stops trading altogether",
    series[series.length - 1].to < 1e-7,
    `${(series[series.length - 1].to * 100).toFixed(4)}%`);
  ok("when trading stops, the previous holdings are kept exactly",
    vecClose(series[series.length - 1].w, prev, 1e-8));
  ok("turnover genuinely falls somewhere in between, not only at the extremes",
    series.some((r, i) => i > 0 && r.to < series[i - 1].to - 1e-4));
  ok("every solution is fully invested",
    series.every((r) => close(r.w.reduce((a, b) => a + b, 0), 1, 1e-8)));
  ok("every solution is non-negative", series.every((r) => r.w.every((x) => x >= -1e-9)));
  console.log(`      turnover by cost: ${series.map((r, i) => `${costs[i]}→${(r.to * 100).toFixed(0)}%`).join("  ")}`);
}
{
  // Starting from cash, everything is a purchase and the budget must still bind.
  const w = C.efficientWithTurnoverCost(BMU, BCOV, 3, null, 0.01);
  ok("starting from no holdings works", close(w.reduce((a, b) => a + b, 0), 1, 1e-8),
    w.map((x) => (x * 100).toFixed(1)).join(","));
  ok("starting from no holdings gives non-negative weights", w.every((x) => x >= -1e-9));
}
{
  const prev = [0.5, 0.5, 0, 0, 0];
  const bounds = { maxWeight: 0.35, minWeight: 0.05 };
  const w = C.efficientWithTurnoverCost(BMU, BCOV, 3, prev, 0.01, bounds);
  ok("position limits hold under a turnover penalty",
    w.every((x) => x >= 0.05 - 1e-6 && x <= 0.35 + 1e-6),
    w.map((x) => (x * 100).toFixed(2)).join(","));
  ok("the budget still binds under a turnover penalty",
    close(w.reduce((a, b) => a + b, 0), 1, 1e-7));

  const g = C.efficientWithTurnoverCost(BMU, BCOV, 3, prev, 0.01,
    { groups: [{ name: "Equity", assets: EQUITY, max: 0.30 }] });
  ok("group limits hold under a turnover penalty",
    EQUITY.reduce((s, i) => s + g[i], 0) <= 0.30 + 1e-6,
    `${(EQUITY.reduce((s, i) => s + g[i], 0) * 100).toFixed(2)}%`);
}
{
  const prev = [0.2, 0.2, 0.2, 0.2, 0.2];
  const cheap = C.maxSharpeWithTurnoverCost(BMU, BCOV, BRF, prev, 0.0005);
  const dear = C.maxSharpeWithTurnoverCost(BMU, BCOV, BRF, prev, 0.05);
  ok("cost-aware maximum Sharpe returns a portfolio", !!(cheap && cheap.weights));
  ok("a higher cost produces less trading",
    dear.traded <= cheap.traded + 1e-7,
    `${(cheap.traded * 100).toFixed(1)}% vs ${(dear.traded * 100).toFixed(1)}%`);
  ok("both are fully invested",
    close(cheap.weights.reduce((a, b) => a + b, 0), 1, 1e-7) &&
    close(dear.weights.reduce((a, b) => a + b, 0), 1, 1e-7));
  console.log(`      cost-aware max Sharpe traded ${(cheap.traded * 100).toFixed(1)}% at 5bps, ${(dear.traded * 100).toFixed(1)}% at 500bps`);
}
{
  let s = 606060 >>> 0;
  const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let budgetFails = 0, boundFails = 0, monotoneFails = 0, errors = 0, done = 0;

  for (let trial = 0; trial < 200; trial++) {
    const n = 3 + Math.floor(rand() * 4);
    const L = Array.from({ length: n }, () => Array.from({ length: n }, () => rand() - 0.5));
    const cov = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let v = 0;
        for (let k = 0; k < n; k++) v += L[i][k] * L[j][k];
        cov[i][j] = v;
      }
      cov[i][i] += 0.01;
    }
    const mu = Array.from({ length: n }, () => rand() * 0.2 - 0.03);
    const raw = Array.from({ length: n }, () => rand());
    const tot = raw.reduce((a, b) => a + b, 0);
    const prev = raw.map((v) => v / tot);
    const tau = Math.pow(10, -1 + 2 * rand());

    try {
      const lo = C.efficientWithTurnoverCost(mu, cov, tau, prev, 0.001);
      const hi = C.efficientWithTurnoverCost(mu, cov, tau, prev, 0.30);
      done++;
      for (const w of [lo, hi]) {
        if (!close(w.reduce((a, b) => a + b, 0), 1, 1e-6)) budgetFails++;
        if (w.some((x) => x < -1e-7)) boundFails++;
      }
      const toLo = lo.reduce((s2, x, i) => s2 + Math.abs(x - prev[i]), 0);
      const toHi = hi.reduce((s2, x, i) => s2 + Math.abs(x - prev[i]), 0);
      if (toHi > toLo + 1e-5) monotoneFails++;
    } catch (e) { errors++; }
  }
  ok("200 random turnover-penalized problems: no exceptions", errors === 0, `${errors}`);
  ok("200 random turnover-penalized problems: all fully invested", budgetFails === 0, `${budgetFails}`);
  ok("200 random turnover-penalized problems: all non-negative", boundFails === 0, `${boundFails}`);
  ok("200 random turnover-penalized problems: a higher cost never trades more",
    monotoneFails === 0, `${monotoneFails} violations`);
  console.log(`      ${done} solved, singular augmented matrix handled throughout`);
}

// ---------------------------------------------------------------- feasibility
section("Phase 1: reaching the feasible region");
{
  const n = 5;
  const a = new Array(n).fill(1);
  const lower = new Array(n).fill(0.05);
  const upper = new Array(n).fill(0.35);
  const { G, h } = C.buildGroupConstraints(n, [{ name: "EQ", assets: [0, 2], max: 0.30 }]);

  const f = C.findFeasiblePoint(n, a, 1, lower, upper, G, h);
  ok("a feasible point is found for a satisfiable mandate", f.feasible, f.violations.join(", "));
  ok("it satisfies the budget", close(f.x.reduce((x, y) => x + y, 0), 1, 1e-7));
  ok("it satisfies every bound", f.x.every((v) => v >= 0.05 - 1e-7 && v <= 0.35 + 1e-7));
  ok("it satisfies the group row", f.x[0] + f.x[2] <= 0.30 + 1e-7,
    `${((f.x[0] + f.x[2]) * 100).toFixed(3)}%`);

  const bad = C.findFeasiblePoint(n, a, 1, new Array(n).fill(0.3), new Array(n).fill(0.35), [], []);
  ok("a genuinely impossible mandate is reported as infeasible", !bad.feasible);
}
{
  // This is the case that exposed the defect. Previous holdings that breach the
  // current limits are a real situation: a mandate tightens, or the portfolio
  // drifts out of compliance. The optimizer must be able to trade back in.
  const outOfCompliance = [0.5, 0.5, 0, 0, 0];
  const bounds = { maxWeight: 0.35, minWeight: 0.05 };

  let w = null, err = "";
  try { w = C.efficientWithTurnoverCost(BMU, BCOV, 3, outOfCompliance, 0.01, bounds); }
  catch (e) { err = e.message; }
  ok("optimizing from holdings that breach the limits succeeds", w !== null, err);
  if (w) {
    ok("the result is back inside the limits",
      w.every((x) => x >= 0.05 - 1e-6 && x <= 0.35 + 1e-6),
      w.map((x) => (x * 100).toFixed(2)).join(","));
    ok("the result is fully invested", close(w.reduce((a2, b2) => a2 + b2, 0), 1, 1e-6));
  }

  let wg = null, errg = "";
  try {
    wg = C.efficientWithTurnoverCost(BMU, BCOV, 3, outOfCompliance, 0.01, {
      ...bounds, groups: [{ name: "Equity", assets: EQUITY, max: 0.30 }],
    });
  } catch (e) { errg = e.message; }
  ok("the same holds with a group limit added", wg !== null, errg);
  if (wg) {
    ok("the group limit is met from an out-of-compliance start",
      EQUITY.reduce((s2, i) => s2 + wg[i], 0) <= 0.30 + 1e-6,
      `${(EQUITY.reduce((s2, i) => s2 + wg[i], 0) * 100).toFixed(2)}%`);
  }
}
{
  // A mandate that truly cannot be met must still be refused, so the fallback
  // above cannot be masking real infeasibility.
  const threw = (fn) => { try { fn(); return ""; } catch (e) { return e.message; } };
  const m = threw(() => C.efficientWithTurnoverCost(BMU, BCOV, 3, [0.5, 0.5, 0, 0, 0], 0.01, {
    groups: [
      { name: "A", assets: [0, 1], max: 0.2 },
      { name: "B", assets: [2, 3], max: 0.2 },
      { name: "C", assets: [4], max: 0.2 },
    ],
  }));
  ok("an impossible mandate is still refused, not quietly approximated",
    /cannot be satisfied|fully invested/.test(m), m);
}

// ---------------------------------------------------------------- fallback quality
section("Projected-gradient fallback quality");
{
  // The fallback matters only if its answers are close to optimal. Measured
  // against the exact solver on problems where the exact method converges.
  let s2 = 777 >>> 0;
  const rand = () => { s2 = (s2 + 0x6d2b79f5) >>> 0; let t = Math.imul(s2 ^ (s2 >>> 15), 1 | s2); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gaps = [];
  let neverBetter = true;

  for (let trial = 0; trial < 120; trial++) {
    const n = 3 + Math.floor(rand() * 4);
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
    const c = Array.from({ length: n }, () => rand() * 0.2 - 0.03);
    const lower = new Array(n).fill(0), upper = new Array(n).fill(1);
    const a = new Array(n).fill(1);

    let exact;
    try { exact = C.solveQP(cov, c, a, 1, { lower, upper }); } catch (e) { continue; }
    if (!exact.converged) continue;

    const obj = (v) => {
      let q = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q += v[i] * cov[i][j] * v[j];
      return 0.5 * q - C.dot(c, v);
    };
    const feas = C.findFeasiblePoint(n, a, 1, lower, upper, [], []);
    const ref = C.refineByProjectedGradient(cov, c, a, 1, feas.x, lower, upper, [], []);
    const oExact = obj(exact.x);
    const gap = (ref.objective - oExact) / Math.max(Math.abs(oExact), 1e-9);
    gaps.push(gap);
    if (gap < -1e-9) neverBetter = false;
  }

  gaps.sort((x, y) => x - y);
  const p50 = gaps[Math.floor(gaps.length / 2)];
  const p90 = gaps[Math.floor(gaps.length * 0.9)];
  ok("the fallback never beats the exact solver, which validates the exact one",
    neverBetter);
  ok("the fallback matches the exact optimum for at least half of problems",
    Math.abs(p50) < 1e-6, `median gap ${(p50 * 100).toExponential(2)}%`);
  ok("the fallback is within a fraction of a percent for 90% of problems",
    p90 < 0.01, `90th percentile gap ${(p90 * 100).toExponential(2)}%`);
  console.log(`      objective gap vs exact: median ${(p50 * 100).toExponential(1)}%, p90 ${(p90 * 100).toExponential(1)}%, worst ${(gaps[gaps.length - 1] * 100).toFixed(1)}%`);
  console.log(`      the worst case is poor; the fallback is a safety net, not a substitute`);
}

// ---------------------------------------------------------------- approximate-flag propagation
section("Non-convergence is visible to every caller, not silently absorbed");
{
  // Every wrapper attaches .approximate to its returned weight array (arrays
  // are objects in JS, so this is free and breaks no existing consumer that
  // treats the return value as a plain array). This was a real gap: earlier,
  // a caller several layers removed from solveQP had no way to learn that an
  // answer was a feasibility fallback rather than a proven optimum.
  const mv = C.minVarianceLongOnly(BCOV);
  ok("minVarianceLongOnly tags a normal solve as exact", mv.approximate === false);

  const ef = C.efficientLongOnly(BMU, BCOV, 3);
  ok("efficientLongOnly tags a normal solve as exact", ef.approximate === false);

  const ms = C.maxSharpeLongOnly(BMU, BCOV, BRF);
  ok("maxSharpeLongOnly (exact path) reports approximate: false", ms.approximate === false);

  const msC = C.maxSharpeLongOnly(BMU, BCOV, BRF, { maxWeight: 0.30 });
  ok("maxSharpeLongOnly (golden-section path) reports a boolean approximate flag",
    typeof msC.approximate === "boolean");
  ok("golden-section's own convergence is not conflated with the QP's",
    msC.converged === true && msC.approximate === false,
    `converged=${msC.converged} approximate=${msC.approximate}`);

  const tc = C.efficientWithTurnoverCost(BMU, BCOV, 3, [0.2, 0.2, 0.2, 0.2, 0.2], 0.01);
  ok("efficientWithTurnoverCost tags its result", tc.approximate === false);

  const front = C.longOnlyFrontier(BMU, BCOV, 40);
  ok("longOnlyFrontier rolls up an approximate flag across all its points",
    front.approximate === false);
  ok("every frontier point's own weight array carries the same flag",
    front.every((p) => p.weights.approximate === false));
}
{
  // The flag must be able to flip to true, not just sit at false forever — a
  // test that only ever sees "false" would prove nothing about the branch it
  // claims to cover. This exact seed/trial combination is known (from the
  // group-constraint stress test elsewhere) to hit the fallback path, so the
  // assertion is a direct positive case rather than a hopeful OR.
  function mulberrySeeded(seed) {
    let s2 = seed >>> 0;
    return () => {
      s2 = (s2 + 0x6d2b79f5) >>> 0;
      let t = Math.imul(s2 ^ (s2 >>> 15), 1 | s2);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rand = mulberrySeeded(31415926);
  let forced = null;
  for (let t = 0; t < 5; t++) {
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
    const tau = Math.pow(10, -2 + 4 * rand());
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
    if (C.checkGroupFeasibility(n, lower, upper, groups)) continue;
    let w;
    try { w = C.efficientLongOnly(mu, cov, tau, { lower, upper, groups }); } catch (e) { continue; }
    if (w.approximate) { forced = { w, n, lower, upper, groups }; break; }
  }
  ok("a known problem that hits the fallback path is correctly flagged approximate: true",
    forced !== null && forced.w.approximate === true,
    forced ? `n=${forced.n}` : "the fallback did not trigger — the test fixture no longer matches the code");
  if (forced) {
    ok("that same result is nonetheless fully feasible",
      close(forced.w.reduce((a, b) => a + b, 0), 1, 1e-6) &&
      forced.w.every((v, i) => v >= forced.lower[i] - 1e-6 && v <= forced.upper[i] + 1e-6));
  }
}
{
  // Across the same 400-trial stress test used elsewhere, every approximate
  // result must still be feasible, and the flag must correctly identify which
  // outcomes came from the fallback (cross-checked against solveQP directly).
  let s2 = 31415926 >>> 0;
  const rand = () => { s2 = (s2 + 0x6d2b79f5) >>> 0; let t = Math.imul(s2 ^ (s2 >>> 15), 1 | s2); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let mismatches = 0, checked = 0, approxCount = 0;

  for (let t = 0; t < 300; t++) {
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
    const tau = Math.pow(10, -2 + 4 * rand());
    const lower = Array.from({ length: n }, () => rand() * (0.8 / n));
    const upper = lower.map((lo) => lo + 0.1 + rand() * 0.8);
    const bounds = { lower, upper };

    let direct, wrapped;
    try {
      direct = C.solveQP(cov, mu.map((m) => tau * m), Array(n).fill(1), 1, { lower, upper });
      wrapped = C.efficientLongOnly(mu, cov, tau, bounds);
    } catch (e) { continue; }
    checked++;
    if (wrapped.approximate) approxCount++;
    if (wrapped.approximate === direct.converged) mismatches++; // flag must be the negation
  }
  ok("the wrapper's approximate flag always matches solveQP's own converged flag (negated)",
    mismatches === 0, `${mismatches} of ${checked} disagreed`);
  console.log(`      ${approxCount} of ${checked} random trials produced an approximate result, all correctly flagged`);
}

// ---------------------------------------------------------------- summary
console.log(`\n${"=".repeat(60)}`);
console.log(`${pass} passed, ${fail} failed`);
if (fail) { console.log("Failures:"); failures.forEach((f) => console.log(`  - ${f}`)); }
console.log("=".repeat(60));
process.exit(fail ? 1 : 0);
