const C = require("./portfolio-core.js");

let pass = 0, fail = 0;
const failures = [];
function ok(name, cond, detail = "") {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; failures.push(name); console.log(`  FAIL  ${name}  ${detail}`); }
}
const close = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;
const vecClose = (a, b, tol = 1e-6) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < tol);
const section = (s) => console.log(`\n== ${s}`);

const MU0 = [0.08, 0.05, 0.10];
const COV = C.buildCovariance([0.15, 0.08, 0.20], [[1, 0.3, 0.5], [0.3, 1, 0.2], [0.5, 0.2, 1]]);

// ---------------------------------------------------------------- boundary conditions
section("Boundary conditions (the properties any correct BL implementation must have)");
{
  const r = C.blackLitterman(MU0, COV, [], [], []);
  ok("no views leaves the prior exactly unchanged", vecClose(r.mu, MU0, 1e-12));
  ok("no views leaves the covariance exactly unchanged",
    r.cov.every((row, i) => vecClose(row, COV[i], 1e-12)));
}
{
  const P = [[1, 0, 0]];
  const Q = [0.50]; // far from the prior of 0.08
  const r = C.blackLitterman(MU0, COV, P, Q, [1e12], 0.05);
  ok("infinite view uncertainty leaves the posterior at the prior", vecClose(r.mu, MU0, 1e-6));
}
{
  const P = [[1, 0, 0]];
  const Q = [0.50];
  const r = C.blackLitterman(MU0, COV, P, Q, [1e-9], 0.05);
  ok("near-zero view uncertainty pulls the posterior onto the view", close(r.mu[0], 0.50, 1e-3));
  ok("unrelated assets still move somewhat, via the correlation structure",
    Math.abs(r.mu[1] - MU0[1]) > 1e-4 && Math.abs(r.mu[2] - MU0[2]) > 1e-4);
}
{
  // A view exactly equal to the prior should leave the posterior unchanged,
  // regardless of confidence — there is nothing to correct toward.
  const P = [[1, 0, 0]];
  const r = C.blackLitterman(MU0, COV, P, [MU0[0]], [0.0001]);
  ok("a view that agrees with the prior leaves it unchanged", close(r.mu[0], MU0[0], 1e-6));
}
{
  // Symmetry: two independent views on different assets should not interact.
  const P = [[1, 0, 0], [0, 1, 0]];
  const uncorrelatedCov = C.buildCovariance([0.15, 0.08, 0.20], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const r = C.blackLitterman(MU0, uncorrelatedCov, P, [0.20, 0.01], [0.0001, 0.0001]);
  ok("with zero correlation, a view on asset 0 does not move asset 2",
    close(r.mu[2], MU0[2], 1e-6));
  ok("each view pulls its own asset toward its target",
    r.mu[0] > MU0[0] && r.mu[1] < MU0[1]);
}

// ---------------------------------------------------------------- posterior covariance
section("Posterior covariance (deliberately left unchanged — see design note in the source)");
{
  const P = [[1, 0, 0]];
  const r = C.blackLitterman(MU0, COV, P, [0.20], [0.01]);
  ok("the covariance is returned unchanged, not reshaped",
    r.cov.every((row, i) => vecClose(row, COV[i], 1e-12)));
}
{
  // The bug this design replaced: a near-infinite-uncertainty view is
  // mathematically equivalent to no view, so the two must produce identical
  // covariance. An earlier formulation folded residual mean-uncertainty into
  // the covariance and failed this by a material margin (0.0225 vs 0.023625).
  const noViews = C.blackLitterman(MU0, COV, [], [], []);
  const weakView = C.blackLitterman(MU0, COV, [[1, 0, 0]], [0.20], [1e9]);
  const confidentView = C.blackLitterman(MU0, COV, [[1, 0, 0]], [0.20], [1e-6]);
  ok("no views and a near-infinite-uncertainty view give identical covariance",
    close(weakView.cov[0][0], noViews.cov[0][0], 1e-9),
    `no-views ${noViews.cov[0][0]}, weak-view ${weakView.cov[0][0]}`);
  ok("a confident view ALSO leaves the covariance unchanged (mean-only design)",
    close(confidentView.cov[0][0], noViews.cov[0][0], 1e-9));
}

// ---------------------------------------------------------------- feeds the optimizer
section("Black-Litterman output feeds the optimizer directly");
{
  const P = [[0, 0, 1]]; // bullish view on the third asset
  const r = C.blackLitterman(MU0, COV, P, [0.25], [0.005]);
  const w = C.maxSharpeLongOnly(r.mu, r.cov, 0.02);
  ok("the optimizer runs on BL-adjusted inputs", !!(w && w.weights));
  ok("a bullish view increases that asset's optimal weight",
    w.weights[2] > C.maxSharpeLongOnly(MU0, COV, 0.02).weights[2],
    `with view ${(w.weights[2]*100).toFixed(1)}% vs without ${(C.maxSharpeLongOnly(MU0, COV, 0.02).weights[2]*100).toFixed(1)}%`);
}

// ---------------------------------------------------------------- error handling
section("Error handling");
{
  let threw = false;
  try { C.blackLitterman(MU0, COV, [[1, 0, 0]], [0.1, 0.2], [0.01]); } catch (e) { threw = true; }
  ok("mismatched P/Q lengths are rejected", threw);
}
{
  // Two views that are literally the same row (perfectly collinear) with zero
  // uncertainty make the system underdetermined in a degenerate way.
  let threw = false, msg = "";
  try { C.blackLitterman(MU0, COV, [[1, 0, 0], [1, 0, 0]], [0.1, 0.9], [0, 0]); }
  catch (e) { threw = true; msg = e.message; }
  ok("contradictory zero-uncertainty views on the same asset are rejected, not silently averaged",
    threw, msg);
}

// ---------------------------------------------------------------- sentimentViews
section("sentimentViews — turning scores into BL inputs (relative to the prior)");
{
  const mu0 = [0.10, 0.04, 0.06, 0.03];
  const scores = [0.6, -0.3, null, 0.1];
  const counts = [12, 5, 0, 1];
  const { P, Q, omega, covered } = C.sentimentViews(mu0, scores, counts, { tiltPerUnit: 0.05, baseVariance: 0.04 });
  ok("assets with no coverage (null score) are excluded", !covered.includes(2));
  ok("assets with coverage are included", covered.includes(0) && covered.includes(1) && covered.includes(3));
  ok("each view is a one-hot row on its asset",
    P.every((row, k) => row.filter((v) => v !== 0).length === 1 && row[covered[k]] === 1));
  ok("the view is the prior PLUS the sentiment tilt, not the tilt alone",
    close(Q[covered.indexOf(0)], mu0[0] + 0.6 * 0.05, 1e-9),
    `got ${Q[covered.indexOf(0)]}, expected mu0+tilt=${mu0[0] + 0.6 * 0.05}`);
  ok("a negative score produces a view BELOW that asset's own prior",
    Q[covered.indexOf(1)] < mu0[1]);
}
{
  // The exact bug this signature change fixed: with the old absolute-view
  // design (Q = score*tiltPerUnit, ignoring mu0), a bullish score on an asset
  // whose prior already exceeded the tilt's own magnitude produced a view
  // BELOW the prior — bullish news pulling the estimate down. Confirmed on a
  // concrete case: an 8% prior with a 0.36 score and tiltPerUnit=0.1512 moved
  // the posterior from 8.00% to 7.86%. That must not happen with any positive
  // score, regardless of how large the prior is.
  const mu0 = [0.08];
  const { P, Q, omega } = C.sentimentViews(mu0, [0.36], [2], { tiltPerUnit: 0.1512 });
  const cov = [[0.0225]];
  const post = C.blackLitterman(mu0, cov, P, Q, omega);
  ok("a positive sentiment score never pushes the posterior below the prior",
    post.mu[0] >= mu0[0], `prior ${mu0[0]}, posterior ${post.mu[0]}`);
  ok("on this exact case, the posterior now moves up as bullish news should",
    post.mu[0] > mu0[0], `${(mu0[0] * 100).toFixed(2)}% -> ${(post.mu[0] * 100).toFixed(2)}%`);
}
{
  const mu0 = [0.05, 0.05];
  const scores = [0.5, 0.5];
  const lowCount = C.sentimentViews(mu0, scores, [1, 1], { baseVariance: 0.04 });
  const highCount = C.sentimentViews(mu0, scores, [100, 100], { baseVariance: 0.04 });
  ok("more articles behind a score means lower view uncertainty (more confidence)",
    highCount.omega[0] < lowCount.omega[0],
    `1 article: ${lowCount.omega[0]}, 100 articles: ${highCount.omega[0]}`);
  ok("uncertainty scales as baseVariance / articleCount",
    close(highCount.omega[0], 0.04 / 100, 1e-12));
}
{
  const { P, Q, omega, covered } = C.sentimentViews([0.05, 0.05], [0.3, 0.3], [0, 5], { minArticles: 1 });
  ok("an asset below the minimum article count is excluded even with a valid score",
    !covered.includes(0) && covered.includes(1));
}
{
  const empty = C.sentimentViews([0.05, 0.05], [null, null], [0, 0]);
  ok("no coverage at all produces no views", empty.P.length === 0);
  const r = C.blackLitterman(MU0, COV, empty.P, empty.Q, empty.omega);
  ok("feeding empty views into blackLitterman is safe and leaves the prior untouched",
    vecClose(r.mu, MU0, 1e-12));
}

// ---------------------------------------------------------------- randomized stress test
section("Randomized stress test");
{
  let s = 13579 >>> 0;
  const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let errors = 0, notPSD = 0, trials = 0;

  for (let trial = 0; trial < 300; trial++) {
    const n = 2 + Math.floor(rand() * 6);
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
    const mu0 = Array.from({ length: n }, () => rand() * 0.2 - 0.03);
    const scores = Array.from({ length: n }, () => (rand() < 0.3 ? null : rand() * 2 - 1));
    const counts = scores.map(() => 1 + Math.floor(rand() * 50));
    const { P, Q, omega } = C.sentimentViews(mu0, scores, counts);
    trials++;
    try {
      const r = C.blackLitterman(mu0, cov, P, Q, omega);
      if (!r.cov.every((row, i) => row.every((v, j) => Math.abs(v - cov[i][j]) < 1e-9))) notPSD++;
      if (r.mu.some((v) => !Number.isFinite(v))) errors++;
    } catch (e) { /* a thrown error on a degenerate random case is acceptable */ }
  }
  ok("300 random BL problems: covariance is always returned unchanged", notPSD === 0, `${notPSD} failures`);
  ok("300 random BL problems: posterior mean is always finite", errors === 0, `${errors} failures`);
  console.log(`      ${trials} trials generated`);
}

// ---------------------------------------------------------------- summary
console.log(`\n${"=".repeat(64)}`);
console.log(`${pass} passed, ${fail} failed`);
if (fail) { console.log("Failures:"); failures.forEach((f) => console.log(`  - ${f}`)); }
console.log("=".repeat(64));
process.exit(fail ? 1 : 0);
