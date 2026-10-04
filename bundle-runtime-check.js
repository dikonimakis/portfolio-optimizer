// ============================================================================
// portfolio-core.js
// Mean-variance optimization core. No external dependencies.
// Exact long-only optimization via active-set QP (not Monte Carlo).
// ============================================================================

const EPS = 1e-10;

// ---------------------------------------------------------------- linear algebra

function zeros(n) { return Array(n).fill(0); }

function matVec(A, x) {
  const n = A.length;
  const out = zeros(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < x.length; j++) s += A[i][j] * x[j];
    out[i] = s;
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Gaussian elimination with partial pivoting. Throws if singular.
function solveLinear(Ain, bin) {
  const n = Ain.length;
  const A = Ain.map((r, i) => [...r, bin[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    }
    if (Math.abs(A[piv][col]) < 1e-14) throw new Error("Singular matrix");
    if (piv !== col) { const t = A[piv]; A[piv] = A[col]; A[col] = t; }
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / A[col][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) A[r][c] -= f * A[col][c];
    }
  }
  const x = zeros(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = A[r][n];
    for (let c = r + 1; c < n; c++) s -= A[r][c] * x[c];
    x[r] = s / A[r][r];
  }
  return x;
}

function invertMatrix(A) {
  const n = A.length;
  const inv = [];
  for (let i = 0; i < n; i++) {
    const e = zeros(n); e[i] = 1;
    inv.push(solveLinear(A, e));
  }
  // solveLinear returns columns; transpose to get the inverse
  const out = Array.from({ length: n }, () => zeros(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[j][i] = inv[i][j];
  return out;
}

// Jacobi eigenvalue algorithm for symmetric matrices.
// Returns { values, vectors } where vectors[k] is the k-th eigenvector.
function symmetricEigen(Ain, maxSweeps = 100) {
  const n = Ain.length;
  const A = Ain.map((r) => [...r]);
  let V = Array.from({ length: n }, (_, i) => {
    const r = zeros(n); r[i] = 1; return r;
  });
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-22) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-18) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = A.map((r, i) => r[i]);
  // eigenvector k is column k of V
  const vectors = [];
  for (let k = 0; k < n; k++) vectors.push(V.map((r) => r[k]));
  return { values, vectors };
}

// ---------------------------------------------------------------- matrix validation

function buildCovariance(vols, corr) {
  const n = vols.length;
  const cov = Array.from({ length: n }, () => zeros(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) cov[i][j] = vols[i] * vols[j] * corr[i][j];
  }
  return cov;
}

// Checks a correlation matrix is symmetric, unit-diagonal, and PSD.
function validateCorrelation(corr, tol = -1e-8) {
  const n = corr.length;
  const issues = [];
  for (let i = 0; i < n; i++) {
    if (Math.abs(corr[i][i] - 1) > 1e-9) issues.push(`Diagonal entry [${i}] is not 1`);
    for (let j = 0; j < n; j++) {
      if (Math.abs(corr[i][j] - corr[j][i]) > 1e-9) issues.push(`Not symmetric at [${i},${j}]`);
      if (corr[i][j] < -1 - 1e-9 || corr[i][j] > 1 + 1e-9) issues.push(`Entry [${i},${j}] outside [-1,1]`);
    }
  }
  const { values } = symmetricEigen(corr);
  const minEig = Math.min(...values);
  const isPSD = minEig >= tol;
  if (!isPSD) {
    issues.push(`Not positive semi-definite (minimum eigenvalue ${minEig.toFixed(4)}). These correlations are mutually inconsistent — no set of assets can have all of them simultaneously.`);
  }
  return { valid: issues.length === 0, isPSD, minEigenvalue: minEig, issues };
}

// Nearest valid correlation matrix by eigenvalue clipping + rescaling to unit diagonal.
function repairCorrelation(corr, floor = 1e-8) {
  const n = corr.length;
  const { values, vectors } = symmetricEigen(corr);
  const rebuilt = Array.from({ length: n }, () => zeros(n));
  for (let k = 0; k < n; k++) {
    const lam = Math.max(values[k], floor);
    const v = vectors[k];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) rebuilt[i][j] += lam * v[i] * v[j];
  }
  const out = Array.from({ length: n }, () => zeros(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i][j] = rebuilt[i][j] / Math.sqrt(rebuilt[i][i] * rebuilt[j][j]);
    }
  }
  for (let i = 0; i < n; i++) out[i][i] = 1;
  return out;
}

// ---------------------------------------------------------------- shrinkage

// Shrink covariance toward a constant-correlation target (Ledoit-Wolf target F).
// NOTE: the data-optimal intensity requires the raw return series; with only a
// summary covariance available, delta is supplied by the caller.
function shrinkCovariance(cov, delta) {
  const n = cov.length;
  const vols = cov.map((r, i) => Math.sqrt(r[i]));
  let sum = 0, count = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) { sum += cov[i][j] / (vols[i] * vols[j]); count++; }
  }
  const rbar = count > 0 ? sum / count : 0;
  const out = Array.from({ length: n }, () => zeros(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const target = i === j ? cov[i][i] : rbar * vols[i] * vols[j];
      out[i][j] = delta * target + (1 - delta) * cov[i][j];
    }
  }
  return out;
}

// Jorion (1986) Bayes-Stein shrinkage of expected returns toward the GMV return.
// T = number of observations behind the estimates.
function bayesSteinMu(mu, cov, T) {
  const n = mu.length;
  if (!(T > n + 2)) return { mu: [...mu], phi: 0, muTarget: null };
  const invCov = invertMatrix(cov);
  const ones = Array(n).fill(1);
  const A = dot(ones, matVec(invCov, ones));
  const B = dot(ones, matVec(invCov, mu));
  const muGmv = B / A;
  const d = mu.map((m) => m - muGmv);
  const quad = dot(d, matVec(invCov, d));
  const phi = (n + 2) / ((n + 2) + T * quad);
  const clamped = Math.max(0, Math.min(1, phi));
  return {
    mu: mu.map((m) => (1 - clamped) * m + clamped * muGmv),
    phi: clamped,
    muTarget: muGmv,
  };
}

// ---------------------------------------------------------------- analytic frontier

// Closed-form (short-selling allowed) frontier scalars and key portfolios.
function analyticFrontier(mu, cov, rf) {
  const n = mu.length;
  const invCov = invertMatrix(cov);
  const ones = Array(n).fill(1);
  const invOnes = matVec(invCov, ones);
  const invMu = matVec(invCov, mu);
  const A = dot(ones, invOnes);
  const B = dot(ones, invMu);
  const C = dot(mu, invMu);
  const D = A * C - B * B;

  const gmvWeights = invOnes.map((x) => x / A);
  const gmvRet = B / A;
  const gmvVol = Math.sqrt(1 / A);

  const excess = mu.map((m) => m - rf);
  const invExcess = matVec(invCov, excess);
  const denom = dot(ones, invExcess);
  let tanWeights = null, tanRet = null, tanVol = null;
  if (Math.abs(denom) > 1e-9) {
    tanWeights = invExcess.map((x) => x / denom);
    tanRet = dot(tanWeights, mu);
    tanVol = Math.sqrt(quadForm(tanWeights, cov));
  }
  return { A, B, C, D, gmvWeights, gmvRet, gmvVol, tanWeights, tanRet, tanVol,
    varianceAt: (r) => (A * r * r - 2 * B * r + C) / D };
}

function quadForm(w, cov) {
  const n = w.length;
  let v = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) v += w[i] * w[j] * cov[i][j];
  return Math.max(v, 0);
}

// ---------------------------------------------------------------- active-set QP

// Turns a human description of group limits into inequality rows G x <= h.
//
//   groups: [{ name: "Equities", assets: [0, 2, 3], max: 0.60, min: 0.10 }]
//
// A maximum becomes  sum(x_i for i in group) <= max.
// A minimum becomes -sum(x_i for i in group) <= -min.
function buildGroupConstraints(n, groups) {
  const G = [];
  const h = [];
  const labels = [];
  if (!groups || groups.length === 0) return { G, h, labels };

  for (const g of groups) {
    const members = (g.assets || []).filter((i) => Number.isInteger(i) && i >= 0 && i < n);
    if (members.length === 0) continue;
    const name = g.name || `group of ${members.length}`;

    if (Number.isFinite(g.max)) {
      const row = zeros(n);
      for (const i of members) row[i] = 1;
      G.push(row); h.push(g.max);
      labels.push(`${name}: at most ${(g.max * 100).toFixed(0)}%`);
    }
    if (Number.isFinite(g.min)) {
      const row = zeros(n);
      for (const i of members) row[i] = -1;
      G.push(row); h.push(-g.min);
      labels.push(`${name}: at least ${(g.min * 100).toFixed(0)}%`);
    }
  }
  return { G, h, labels };
}

// Checks the group limits against the box bounds for contradictions that can be
// found without solving. Cheap, and catches the mistakes people actually make.
function checkGroupFeasibility(n, lower, upper, groups) {
  if (!groups || groups.length === 0) return null;
  for (const g of groups) {
    const members = (g.assets || []).filter((i) => Number.isInteger(i) && i >= 0 && i < n);
    if (members.length === 0) continue;
    const name = g.name || "group";
    let loSum = 0, hiSum = 0;
    for (const i of members) {
      loSum += lower[i];
      hiSum += Number.isFinite(upper[i]) ? upper[i] : 1;
    }
    if (Number.isFinite(g.max) && loSum > g.max + 1e-9) {
      return `"${name}" is capped at ${(g.max * 100).toFixed(0)}%, but its members' minimum allocations already require ${(loSum * 100).toFixed(0)}%.`;
    }
    if (Number.isFinite(g.min) && hiSum < g.min - 1e-9) {
      return `"${name}" needs at least ${(g.min * 100).toFixed(0)}%, but its members' position limits allow at most ${(hiSum * 100).toFixed(0)}%.`;
    }
    if (Number.isFinite(g.max) && Number.isFinite(g.min) && g.min > g.max + 1e-12) {
      return `"${name}" has a minimum above its maximum.`;
    }
  }
  // Disjoint group caps that together cannot reach a fully invested portfolio.
  const covered = new Set();
  let capSum = 0, allDisjoint = true;
  for (const g of groups) {
    if (!Number.isFinite(g.max)) continue;
    for (const i of g.assets || []) {
      if (covered.has(i)) { allDisjoint = false; break; }
      covered.add(i);
    }
    if (!allDisjoint) break;
    capSum += g.max;
  }
  if (allDisjoint && covered.size === n && capSum < 1 - 1e-9) {
    return `The group caps cover every asset and sum to ${(capSum * 100).toFixed(0)}%, so the portfolio cannot be fully invested.`;
  }
  return null;
}

// ---------------------------------------------------------------- feasibility

// Finds a point satisfying  a'x = b,  l <= x <= u,  Gx <= h,  by cyclic
// projection onto each constraint set in turn (POCS). Every set here is convex
// and closed, so the iteration converges to a point in their intersection
// whenever one exists.
//
// This exists because the active-set method in solveQP begins from a point that
// satisfies the bounds and the budget but not necessarily the general rows of G.
// When those are violated at the outset — which happens whenever a portfolio is
// re-optimized against limits its current holdings already breach — the method
// has no route back to feasibility and reports the problem as impossible. That
// was a real defect: the problems it rejected were solvable.
function findFeasiblePoint(n, a, b, lower, upper, G = [], h = [], opts = {}) {
  const maxIter = opts.maxIter || 2000;
  const tol = opts.tol || 1e-11;

  // Start from the budget-satisfying point closest to the middle of the box.
  let x = new Array(n);
  for (let i = 0; i < n; i++) {
    const lo = lower[i];
    const hi = Number.isFinite(upper[i]) ? upper[i] : lo + 1;
    x[i] = opts.start && Number.isFinite(opts.start[i])
      ? Math.min(Math.max(opts.start[i], lo), hi)
      : (lo + hi) / 2;
  }

  const aa = dot(a, a);
  const gg = G.map((row) => dot(row, row));

  for (let iter = 0; iter < maxIter; iter++) {
    let moved = 0;

    // bounds
    for (let i = 0; i < n; i++) {
      const clamped = Math.min(Math.max(x[i], lower[i]), Number.isFinite(upper[i]) ? upper[i] : Infinity);
      moved += Math.abs(clamped - x[i]);
      x[i] = clamped;
    }

    // budget
    if (aa > 1e-14) {
      const excess = (dot(a, x) - b) / aa;
      if (Math.abs(excess) > 0) {
        for (let i = 0; i < n; i++) x[i] -= excess * a[i];
        moved += Math.abs(excess) * Math.sqrt(aa);
      }
    }

    // general rows
    for (let j = 0; j < G.length; j++) {
      if (gg[j] <= 1e-14) continue;
      const viol = dot(G[j], x) - h[j];
      if (viol > 0) {
        const step = viol / gg[j];
        for (let i = 0; i < n; i++) x[i] -= step * G[j][i];
        moved += viol / Math.sqrt(gg[j]);
      }
    }

    if (moved < tol) break;
  }

  // Verify rather than assume: POCS converges only if the intersection is non-empty.
  const errors = [];
  if (Math.abs(dot(a, x) - b) > 1e-7) errors.push("budget");
  for (let i = 0; i < n; i++) {
    if (x[i] < lower[i] - 1e-7 || x[i] > upper[i] + 1e-7) errors.push(`bound ${i}`);
  }
  for (let j = 0; j < G.length; j++) {
    if (dot(G[j], x) - h[j] > 1e-7) errors.push(`row ${j}`);
  }
  return { x, feasible: errors.length === 0, violations: errors };
}

// Refines a feasible point toward the optimum by projected gradient descent.
//
// The active-set method above is exact but can fail to converge on degenerate
// problems, where redundant active constraints repeatedly produce singular KKT
// systems. Returning the bare feasibility point in that case is safe but weak:
// it satisfies every constraint and optimizes nothing. This walks downhill from
// there, projecting back onto the feasible set after each step, which converges
// to the constrained minimum for a convex objective.
//
// Slower than the active-set method and accurate only to tolerance, so it is a
// fallback rather than the primary path, and the result is marked as such.
function refineByProjectedGradient(S, c, a, b, x0, lower, upper, G, h, opts = {}) {
  const n = S.length;
  const maxSteps = opts.maxSteps || 400;
  const project = (v) => findFeasiblePoint(n, a, b, lower, upper, G, h, { start: v, maxIter: 300 });

  const objective = (v) => {
    let q = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) q += v[i] * S[i][j] * v[j];
    }
    return 0.5 * q - dot(c, v);
  };

  let x = x0.slice();
  let best = x.slice();
  let bestObj = objective(x);

  // Step size from the largest diagonal entry, a cheap proxy for the spectral
  // norm that avoids an eigenvalue decomposition on every call.
  let scale = 0;
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(S[i][i]));
  let step = scale > 0 ? 1 / (n * scale) : 1e-3;

  for (let k = 0; k < maxSteps; k++) {
    const Sx = matVec(S, x);
    const grad = new Array(n);
    for (let i = 0; i < n; i++) grad[i] = Sx[i] - c[i];

    const trial = x.map((v, i) => v - step * grad[i]);
    const proj = project(trial);
    if (!proj.feasible) { step *= 0.5; if (step < 1e-16) break; continue; }

    const objTrial = objective(proj.x);
    if (objTrial < bestObj - 1e-15) {
      bestObj = objTrial;
      best = proj.x.slice();
      x = proj.x;
      step *= 1.1;                    // making progress: lengthen the stride
    } else {
      step *= 0.5;                    // overshot: shorten it
      if (step < 1e-16) break;
    }
  }
  return { x: best, objective: bestObj };
}

// Solves:   min (1/2) x'Sx - c'x   s.t.  a'x = b,  l <= x <= u,  Gx <= h
// Convex QP (S positive semi-definite) -> any KKT point is a global optimum.
//
// Active-set method over two kinds of inequality: the per-variable bounds and
// the general rows of G. A working set W holds the rows of G currently treated
// as equalities. With multiplier gamma on the budget constraint and lambda >= 0
// on the active rows of G, the gradient residual is
//     r = Sx - c - gamma*a + G_W' lambda
// and optimality requires r = 0 on free variables, r >= 0 at a lower bound,
// r <= 0 at an upper bound, and lambda >= 0 on every row in W.
function solveQP(S, c, a, b, opts = {}) {
  const n = S.length;
  const maxIter = opts.maxIter || 500;
  const tol = opts.tol || 1e-9;
  const lower = opts.lower || zeros(n);
  const upper = opts.upper || new Array(n).fill(Infinity);
  const G = opts.G || [];
  const h = opts.h || [];
  const m = G.length;

  for (let i = 0; i < n; i++) {
    if (lower[i] > upper[i] + 1e-12) {
      throw new Error(`Asset ${i + 1}: the minimum allocation exceeds its maximum.`);
    }
  }

  // Budget feasibility given the bounds alone.
  {
    let lo = 0, hi = 0;
    for (let i = 0; i < n; i++) {
      const t1 = a[i] * lower[i];
      const t2 = a[i] * (Number.isFinite(upper[i]) ? upper[i] : 1e12);
      lo += Math.min(t1, t2);
      hi += Math.max(t1, t2);
    }
    if (b < lo - 1e-9 || b > hi + 1e-9) {
      throw new Error("These limits cannot be satisfied: the minimum allocations and maximum allocations do not allow a fully invested portfolio.");
    }
  }

  const atLower = new Set();
  const atUpper = new Set();
  let free = new Set();
  for (let i = 0; i < n; i++) {
    if (Math.abs(a[i]) < 1e-14) { atLower.add(i); continue; }
    free.add(i);
  }
  if (free.size === 0) throw new Error("No variable can satisfy the budget constraint.");

  let working = new Set();      // indices of rows of G held as equalities
  const blocked = new Set();    // rows abandoned after repeated singular systems
  const dropCount = new Map();  // how often each row has been released

  // The iteration below begins from a no-trade point, which satisfies the bounds
  // and budget but may violate the general rows. When it does, seed the working
  // set from an explicitly constructed feasible point so the method starts inside
  // the feasible region rather than trying to find its way in.
  if (m > 0) {
    const start = zeros(n);
    for (let i = 0; i < n; i++) start[i] = Math.min(Math.max(0, lower[i]), Number.isFinite(upper[i]) ? upper[i] : Infinity);
    let anyViolated = false;
    for (let j = 0; j < m; j++) {
      if (dot(G[j], start) - h[j] > 1e-9) { anyViolated = true; break; }
    }
    if (anyViolated) {
      const feas = findFeasiblePoint(n, a, b, lower, upper, G, h, { start: opts.x0 });
      if (!feas.feasible) {
        throw new Error("These limits cannot be satisfied simultaneously.");
      }
      for (let j = 0; j < m; j++) {
        if (Math.abs(dot(G[j], feas.x) - h[j]) < 1e-7) working.add(j);
      }
    }
  }

  let x = zeros(n);
  let gamma = 0;
  let lambda = new Map();

  for (let iter = 0; iter < maxIter; iter++) {
    const F = [...free].sort((p, q) => p - q);
    const k = F.length;
    const W = [...working].sort((p, q) => p - q);

    const xFixed = zeros(n);
    for (const i of atLower) xFixed[i] = lower[i];
    for (const i of atUpper) xFixed[i] = upper[i];

    let bAdj = b;
    for (const i of [...atLower, ...atUpper]) bAdj -= a[i] * xFixed[i];

    if (k === 0) {
      if (Math.abs(bAdj) < 1e-7) {
        return { x: xFixed.slice(), gamma, converged: true, iterations: iter + 1 };
      }
      throw new Error("These limits cannot be satisfied simultaneously.");
    }

    const SxFixed = matVec(S, xFixed);

    // KKT system:
    //   S_FF x_F - gamma a_F + G_WF' lambda = c_F - (S x_fixed)_F
    //   a_F' x_F                            = b - a_fixed' x_fixed
    //   G_WF x_F                            = h_W - G_W,fixed x_fixed
    const dim = k + 1 + W.length;
    const M = Array.from({ length: dim }, () => zeros(dim));
    const rhs = zeros(dim);

    for (let r = 0; r < k; r++) {
      for (let cc = 0; cc < k; cc++) M[r][cc] = S[F[r]][F[cc]];
      M[r][k] = -a[F[r]];
      for (let q = 0; q < W.length; q++) M[r][k + 1 + q] = G[W[q]][F[r]];
      rhs[r] = c[F[r]] - SxFixed[F[r]];
    }
    for (let cc = 0; cc < k; cc++) M[k][cc] = a[F[cc]];
    rhs[k] = bAdj;
    for (let q = 0; q < W.length; q++) {
      let hAdj = h[W[q]];
      for (const i of [...atLower, ...atUpper]) hAdj -= G[W[q]][i] * xFixed[i];
      for (let cc = 0; cc < k; cc++) M[k + 1 + q][cc] = G[W[q]][F[cc]];
      rhs[k + 1 + q] = hAdj;
    }

    let sol;
    try {
      sol = solveLinear(M, rhs);
    } catch (e) {
      // Redundant or conflicting active constraints make the system singular.
      // Drop the most recently added working row first; failing that, pin a
      // variable and retry with a smaller free set.
      if (W.length > 0) {
        // A singular system means the active rows are locally redundant, not
        // that the constraint is wrong. Blocking permanently was a defect: rows
        // dropped here were never reconsidered, and the method then terminated
        // at a point violating them while reporting the problem impossible.
        // Rows are now released for retry, with a bounded attempt count so a
        // pathological problem still terminates rather than cycling.
        const drop = W[W.length - 1];
        working.delete(drop);
        const tries = (dropCount.get(drop) || 0) + 1;
        dropCount.set(drop, tries);
        if (tries >= 3) blocked.add(drop);
        continue;
      }
      if (k === 1) throw new Error("Covariance matrix is degenerate — cannot solve.");
      const drop = F[F.length - 1];
      free.delete(drop);
      atLower.add(drop);
      continue;
    }

    const xNew = xFixed.slice();
    for (let r = 0; r < k; r++) xNew[F[r]] = sol[r];
    gamma = sol[k];
    lambda = new Map();
    for (let q = 0; q < W.length; q++) lambda.set(W[q], sol[k + 1 + q]);

    // Primal feasibility, bounds first.
    let worst = -1, worstViol = tol, worstIsUpper = false;
    for (let r = 0; r < k; r++) {
      const i = F[r];
      const belowBy = lower[i] - xNew[i];
      const aboveBy = xNew[i] - upper[i];
      if (belowBy > worstViol) { worstViol = belowBy; worst = i; worstIsUpper = false; }
      if (aboveBy > worstViol) { worstViol = aboveBy; worst = i; worstIsUpper = true; }
    }
    if (worst >= 0) {
      free.delete(worst);
      if (worstIsUpper) atUpper.add(worst); else atLower.add(worst);
      continue;
    }

    // Primal feasibility, group rows.
    let addRow = -1, addViol = 1e-8;
    for (let j = 0; j < m; j++) {
      if (working.has(j) || blocked.has(j)) continue;
      const viol = dot(G[j], xNew) - h[j];
      if (viol > addViol) { addViol = viol; addRow = j; }
    }
    if (addRow >= 0) {
      working.add(addRow);
      continue;
    }

    x = xNew.map((v, i) => Math.min(Math.max(v, lower[i]), upper[i]));

    // Dual feasibility on the group rows: a negative multiplier means the
    // constraint is pushing the wrong way and should be released.
    let dropRow = -1, dropViol = -1e-9;
    for (const [j, lam] of lambda) {
      if (lam < dropViol) { dropViol = lam; dropRow = j; }
    }
    if (dropRow >= 0) {
      working.delete(dropRow);
      continue;
    }

    // Dual feasibility on the bounds, including the group contribution.
    const Sx = matVec(S, x);
    const gLambda = zeros(n);
    for (const [j, lam] of lambda) {
      for (let i = 0; i < n; i++) gLambda[i] += G[j][i] * lam;
    }

    let bestAdd = -1, bestViol = tol;
    for (const i of atLower) {
      if (Math.abs(a[i]) < 1e-14) continue;
      if (lower[i] >= upper[i] - 1e-12) continue;
      const r = Sx[i] - c[i] - gamma * a[i] + gLambda[i];
      if (-r > bestViol) { bestViol = -r; bestAdd = i; }
    }
    for (const i of atUpper) {
      if (Math.abs(a[i]) < 1e-14) continue;
      if (lower[i] >= upper[i] - 1e-12) continue;
      const r = Sx[i] - c[i] - gamma * a[i] + gLambda[i];
      if (r > bestViol) { bestViol = r; bestAdd = i; }
    }
    if (bestAdd >= 0) {
      atLower.delete(bestAdd);
      atUpper.delete(bestAdd);
      free.add(bestAdd);
      continue;
    }

    // Anything abandoned for singularity must still hold, or the answer is wrong.
    // Rather than declaring an impossible problem, check whether a feasible point
    // exists at all: if one does, the failure was the method's, not the mandate's.
    let abandonedViolated = false;
    for (const j of blocked) {
      if (dot(G[j], x) - h[j] > 1e-6) { abandonedViolated = true; break; }
    }
    if (abandonedViolated) {
      const feas = findFeasiblePoint(n, a, b, lower, upper, G, h, { start: x });
      if (!feas.feasible) throw new Error("These limits cannot be satisfied simultaneously.");
      const refined = refineByProjectedGradient(S, c, a, b, feas.x, lower, upper, G, h);
      return { x: refined.x, gamma, lambda, converged: false, approximate: true, iterations: iter + 1 };
    }

    return { x, gamma, lambda, converged: true, iterations: iter + 1 };
  }
  // Exhausting the iteration budget means the active set never settled. Rather
  // than return whatever the last iterate happened to be, fall back to a verified
  // feasible point refined toward the optimum, and say the answer is approximate.
  const feas = findFeasiblePoint(n, a, b, lower, upper, G, h, { start: x });
  if (!feas.feasible) throw new Error("These limits cannot be satisfied simultaneously.");
  const refined = refineByProjectedGradient(S, c, a, b, feas.x, lower, upper, G, h);
  return { x: refined.x, gamma, lambda, converged: false, approximate: true, iterations: maxIter };
}

// Verifies the full KKT conditions at x. Used by the test suite: the solver
// does not grade its own work.
function checkKKT(S, c, a, b, x, gamma, tol = 1e-7, lower = null, upper = null, G = [], h = [], lambda = null) {
  const n = S.length;
  const lo = lower || zeros(n);
  const up = upper || new Array(n).fill(Infinity);
  const Sx = matVec(S, x);
  const errors = [];

  if (Math.abs(dot(a, x) - b) > tol) errors.push("Equality constraint violated");

  const gLambda = zeros(n);
  if (lambda) {
    for (const [j, lam] of lambda) {
      if (lam < -tol) errors.push(`Group multiplier ${j} is negative (${lam.toExponential(2)})`);
      const slack = h[j] - dot(G[j], x);
      if (Math.abs(lam * slack) > tol) errors.push(`Complementary slackness violated on group row ${j}`);
      for (let i = 0; i < n; i++) gLambda[i] += G[j][i] * lam;
    }
  }
  for (let j = 0; j < G.length; j++) {
    if (dot(G[j], x) - h[j] > tol) errors.push(`Group constraint ${j} violated`);
  }

  for (let i = 0; i < n; i++) {
    if (x[i] < lo[i] - tol) errors.push(`x[${i}] below its lower bound`);
    if (x[i] > up[i] + tol) errors.push(`x[${i}] above its upper bound`);

    const r = Sx[i] - c[i] - gamma * a[i] + gLambda[i];
    const atLo = Math.abs(x[i] - lo[i]) <= tol;
    const atUp = Number.isFinite(up[i]) && Math.abs(x[i] - up[i]) <= tol;

    if (atLo && atUp) continue;
    if (atLo) {
      if (r < -tol) errors.push(`At lower bound but gradient residual ${r.toExponential(2)} < 0 at ${i}`);
    } else if (atUp) {
      if (r > tol) errors.push(`At upper bound but gradient residual ${r.toExponential(2)} > 0 at ${i}`);
    } else {
      if (Math.abs(r) > tol) errors.push(`Interior but gradient residual ${r.toExponential(2)} != 0 at ${i}`);
    }
  }
  return { ok: errors.length === 0, errors };
}

// ---------------------------------------------------------------- Black-Litterman

// Black-Litterman (1992): blends a prior estimate of expected returns with
// subjective "views" (e.g. "I believe asset 3 will outperform by 2%, and I'm
// fairly confident"), weighted by how much each is trusted.
//
//   posterior_mu = mu0 + tau*Cov*P' * inv(P*tau*Cov*P' + Omega) * (Q - P*mu0)
//
// - mu0     prior expected returns (n-vector) — typically the shrunk historical
//           mean, or a market-equilibrium estimate; NOT the raw sample mean,
//           since section 3i established that raw means barely carry signal
// - cov     covariance matrix (n x n)
// - P       view matrix (k x n): row i says which assets view i concerns
// - Q       view vector (k): the expressed views ("asset 3 returns 12%")
// - omega   view uncertainty, either a k x k covariance matrix or a k-vector
//           of variances (diagonal case, the overwhelmingly common one)
// - tau     scales how much weight the prior carries versus the views; smaller
//           tau means more confidence in the prior. 0.05 is a conventional
//           default (Idzorek, 2005).
//
// This implementation deliberately adjusts ONLY the mean, returning the input
// covariance unchanged. The textbook formula also produces a posterior
// covariance by adding back a term representing residual uncertainty in the
// mean estimate — but that term must vanish smoothly as view confidence goes
// to zero, and an earlier version here did not: it returned a materially
// different (larger) covariance for "no views at all" than for "a view with
// near-infinite uncertainty," which are supposed to be the same case. Beyond
// that specific bug, folding parameter uncertainty into the covariance would
// partially undo the data-driven Ledoit-Wolf shrinkage this project already
// performs (section 3e), replacing a fitted shrinkage intensity with a far
// less rigorous heuristic (baseVariance/articleCount in `sentimentViews`).
// Keeping the two concerns separate — Ledoit-Wolf estimates the covariance,
// Black-Litterman only reshapes the mean — avoids both problems.
//
// A view with infinite variance (no confidence) leaves the mean at the prior;
// a view with variance near zero pulls the mean almost exactly onto that view.
// Both are asserted in the test suite, along with continuity as omega grows.
function blackLitterman(mu0, cov, P, Q, omega, tau = 0.05) {
  const n = mu0.length;
  const k = P.length;
  if (k === 0) return { mu: mu0.slice(), cov: cov.map((r) => r.slice()) };
  if (Q.length !== k) throw new Error("Q must have one entry per row of P.");

  const tauCov = cov.map((row) => row.map((v) => v * tau));
  const Omega = Array.isArray(omega[0])
    ? omega
    : omega.map((v, i) => omega.map((_, j) => (i === j ? v : 0)));

  // M = P * tauCov * P' + Omega   (k x k)
  const tauCovPt = Array.from({ length: n }, (_, i) =>
    P.map((row) => row.reduce((s, p, j) => s + p * tauCov[i][j], 0)));
  const M = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => {
      let s = Omega[i][j];
      for (let c = 0; c < n; c++) s += P[i][c] * tauCovPt[c][j];
      return s;
    }));

  let Minv;
  try { Minv = invertMatrix(M); }
  catch (e) { throw new Error("The views are not linearly independent, or Omega is singular — cannot combine them."); }

  const residual = Q.map((q, i) => q - dot(P[i], mu0));
  const weighted = Minv.map((row) => row.reduce((s, v, j) => s + v * residual[j], 0));

  // adjustment_i = sum_k tauCovPt[i][k] * weighted[k]
  const adjustment = tauCovPt.map((row) => row.reduce((s, v, k2) => s + v * weighted[k2], 0));
  const muPost = mu0.map((m, i) => m + adjustment[i]);

  return { mu: muPost, cov: cov.map((r) => r.slice()) };
}

// Builds Black-Litterman inputs from a per-asset sentiment score, one absolute
// view per asset that has coverage. This is a heuristic bridge, not a derived
// result: there is no theorem saying a sentiment score of 0.4 means a 2%
// return tilt. The mapping is a modelling choice, exposed via `tiltPerUnit`
// so it is visible and adjustable rather than buried.
//
//   mu0         the prior mean for every asset (n-vector) — the view is built
//               RELATIVE to this, not as an absolute replacement. An earlier
//               version computed the view as `score * tiltPerUnit` alone,
//               discarding mu0 entirely: a maximally bullish score (1.0) with
//               tiltPerUnit=0.05 would then assert an absolute view of "5%
//               return," which is a DOWNGRADE for any asset whose historical
//               mean already exceeded 5% — bullish news could pull the
//               estimate down. Confirmed empirically: a real (positive)
//               sentiment score on an asset with an 8% prior moved its
//               posterior mean from 8.00% to 7.86%. Fixed by making the view
//               additive to the prior, so a positive score always pushes the
//               estimate up from wherever it already was, and a negative
//               score always pushes it down — which is what "tilt" means.
//   scores      per-asset sentiment in roughly [-1, 1]; NaN/null = no coverage
//   articleCount per-asset article volume behind each score (more = more
//               confidence, following the usual 1/sqrt(n) shrinkage of a
//               noisy estimate)
//   tiltPerUnit the return tilt implied by a sentiment score of 1.0 (annualized)
//   baseVariance the view-uncertainty assigned to a single-article score;
//               variance for a score is baseVariance / articleCount
function sentimentViews(mu0, scores, articleCounts, opts = {}) {
  const tiltPerUnit = opts.tiltPerUnit != null ? opts.tiltPerUnit : 0.05;
  const baseVariance = opts.baseVariance != null ? opts.baseVariance : 0.04;
  const minArticles = opts.minArticles || 1;

  const P = [];
  const Q = [];
  const omega = [];
  const covered = [];
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    const n2 = articleCounts ? articleCounts[i] : 1;
    if (!Number.isFinite(s) || !Number.isFinite(n2) || n2 < minArticles) continue;
    const row = zeros(scores.length);
    row[i] = 1;
    P.push(row);
    Q.push(mu0[i] + s * tiltPerUnit);
    omega.push(baseVariance / n2);
    covered.push(i);
  }
  return { P, Q, omega, covered };
}

const ONES = (n) => Array(n).fill(1);

// Normalizes a bounds specification into { lower, upper } arrays.
// Accepts { lower: [], upper: [] } or scalar shorthands { maxWeight, minWeight }.
function normalizeBounds(n, bounds) {
  if (!bounds) {
    return { lower: zeros(n), upper: new Array(n).fill(Infinity), G: [], h: [], labels: [], trivial: true };
  }
  const lower = bounds.lower
    ? bounds.lower.slice()
    : new Array(n).fill(Number.isFinite(bounds.minWeight) ? bounds.minWeight : 0);
  const upper = bounds.upper
    ? bounds.upper.slice()
    : new Array(n).fill(Number.isFinite(bounds.maxWeight) ? bounds.maxWeight : Infinity);

  const groups = bounds.groups || [];
  const infeasible = checkGroupFeasibility(n, lower, upper, groups);
  if (infeasible) throw new Error(infeasible);
  const { G, h, labels } = buildGroupConstraints(n, groups);

  const trivial = lower.every((v) => v <= 0) && upper.every((v) => v >= 1) && G.length === 0;
  return { lower, upper, G, h, labels, trivial };
}

// Global minimum variance, subject to bounds.
//
// Returns the weight array with `.approximate` and `.iterations` attached
// (arrays are objects in JS, so this costs nothing and breaks no existing
// caller that treats the return value as a plain array).
function minVarianceLongOnly(cov, bounds) {
  const n = cov.length;
  const { lower, upper, G, h } = normalizeBounds(n, bounds);
  const r = solveQP(cov, zeros(n), ONES(n), 1, { lower, upper, G, h });
  r.x.approximate = !r.converged;
  r.x.iterations = r.iterations;
  return r.x;
}

// Efficient portfolio at risk tolerance tau >= 0, subject to bounds.
// tau = 0 -> minimum variance; tau -> infinity -> maximum return.
function efficientLongOnly(mu, cov, tau, bounds) {
  const n = mu.length;
  const { lower, upper, G, h } = normalizeBounds(n, bounds);
  const r = solveQP(cov, mu.map((m) => tau * m), ONES(n), 1, { lower, upper, G, h });
  r.x.approximate = !r.converged;
  r.x.iterations = r.iterations;
  return r.x;
}

function evaluate(w, mu, cov, rf) {
  const ret = dot(w, mu);
  const vol = Math.sqrt(quadForm(w, cov));
  return { weights: w, ret, vol, sharpe: vol > 0 ? (ret - rf) / vol : 0 };
}

// Maximum-Sharpe portfolio.
//
// With plain long-only bounds this is solved EXACTLY by the Cornuejols-Tutuncu
// transformation: the Sharpe ratio is scale-invariant in w, so
//     min y'Sy  s.t. (mu - rf)'y = 1, y >= 0,   then  w = y / sum(y)
//
// That transformation does NOT survive general position limits: a constraint
// w_i <= u_i becomes y_i <= u_i * sum(y), which is a general linear inequality
// rather than a bound on y. So when bounds are non-trivial the optimum is found
// by golden-section search over risk tolerance along the constrained frontier.
// The efficient frontier is concave, so the Sharpe ratio — the slope of the line
// from (0, rf) to a frontier point — is unimodal along it, which is exactly the
// condition golden-section search requires. It converges to tolerance rather
// than closing in one step, but it is not a heuristic.
function maxSharpeLongOnly(mu, cov, rf, bounds) {
  const n = mu.length;
  const { lower, upper, G, h, trivial } = normalizeBounds(n, bounds);
  const excess = mu.map((m) => m - rf);

  if (trivial) {
    if (Math.max(...excess) <= 0) {
      return { weights: null, reason: "No asset has an expected return above the risk-free rate, so no risky portfolio beats cash." };
    }
    const r = solveQP(cov, zeros(n), excess, 1, { lower: zeros(n), upper: new Array(n).fill(Infinity) });
    const sum = r.x.reduce((s, v) => s + v, 0);
    if (sum <= EPS) return { weights: null, reason: "Degenerate solution." };
    return { ...evaluate(r.x.map((v) => v / sum), mu, cov, rf), converged: r.converged, approximate: !r.converged, method: "exact" };
  }

  // Constrained case: search over risk tolerance.
  const scale = Math.max(...cov.map((r, i) => r[i])) / Math.max(1e-9, Math.max(...excess.map(Math.abs)));
  const sharpeAt = (logTau) => {
    try {
      const w = efficientLongOnly(mu, cov, Math.exp(logTau) * scale, { lower, upper, groups: bounds && bounds.groups });
      return { ...evaluate(w, mu, cov, rf), approximate: !!w.approximate };
    } catch (e) { return null; }
  };

  // Coarse bracket first: golden-section needs a unimodal interval, and a
  // sweep also guards against the optimum sitting at an endpoint.
  let bestLog = null, best = null;
  for (let i = 0; i <= 48; i++) {
    const lg = -12 + (24 * i) / 48;
    const r = sharpeAt(lg);
    if (r && (!best || r.sharpe > best.sharpe)) { best = r; bestLog = lg; }
  }
  if (!best) {
    const w = minVarianceLongOnly(cov, { lower, upper, groups: bounds && bounds.groups });
    return { ...evaluate(w, mu, cov, rf), approximate: !!w.approximate, method: "fallback-min-variance" };
  }

  // Golden-section refinement within one step either side of the coarse best.
  let lo = bestLog - 0.5, hi = bestLog + 0.5;
  const phi = (Math.sqrt(5) - 1) / 2;
  let c1 = hi - phi * (hi - lo), c2 = lo + phi * (hi - lo);
  let f1 = sharpeAt(c1), f2 = sharpeAt(c2);
  for (let it = 0; it < 60 && hi - lo > 1e-10; it++) {
    const s1 = f1 ? f1.sharpe : -Infinity;
    const s2 = f2 ? f2.sharpe : -Infinity;
    if (s1 > s2) { hi = c2; c2 = c1; f2 = f1; c1 = hi - phi * (hi - lo); f1 = sharpeAt(c1); }
    else { lo = c1; c1 = c2; f1 = f2; c2 = lo + phi * (hi - lo); f2 = sharpeAt(c2); }
  }
  for (const cand of [f1, f2]) if (cand && cand.sharpe > best.sharpe) best = cand;
  // "converged: true" here means the golden-section search itself converged
  // (it always does, in a fixed number of steps) — a different claim from
  // "approximate", which reports whether the QP solve AT the winning point
  // was exact. The two must not be conflated: an earlier version hardcoded
  // converged:true and silently dropped whichever candidate's approximate
  // flag it carried.
  return { ...best, converged: true, approximate: !!best.approximate, method: "golden-section" };
}

// ---------------------------------------------------------------- turnover cost

// Efficient portfolio with transaction costs INSIDE the objective:
//
//   min (1/2) w'Sw - tau*mu'w + cost * sum|w - wPrev|
//
// Charging costs after optimizing lets the optimizer produce a portfolio it
// cannot afford to reach. Backtests here show maximum Sharpe turning over 18.7%
// per rebalance against 2.1% for equal weighting; at any realistic cost that
// difference is paid out of returns. Making the optimizer aware of the cost lets
// it decline trades whose expected benefit does not cover them.
//
// The absolute value is not smooth, so the standard reformulation splits the
// trade into non-negative buy and sell legs:
//
//   w = wPrev + b - s,   b >= 0,  s >= 0,   sum|w - wPrev| = 1'b + 1's
//
// which is a quadratic program in x = [b; s] with 2n variables. Writing
// A = [I, -I] so that w - wPrev = Ax:
//
//   quadratic term  (1/2) x' (A'SA) x
//   linear term     [A'(S wPrev - tau*mu) + cost*1]' x
//
// A'SA is singular by construction (rank n in 2n dimensions), which is harmless:
// at the optimum no asset is both bought and sold, so the active set pins one leg
// of every pair at zero and the reduced system has full rank. A small ridge is
// added anyway so intermediate iterates cannot hit an exactly singular system.
function efficientWithTurnoverCost(mu, cov, tau, prevWeights, cost, bounds) {
  const n = mu.length;
  if (!(cost >= 0)) throw new Error("Transaction cost must be zero or positive.");
  const wPrev = prevWeights && prevWeights.length === n ? prevWeights.slice() : zeros(n);
  const { lower, upper, G, h } = normalizeBounds(n, bounds);

  if (cost === 0) {
    const r0 = solveQP(cov, mu.map((m) => tau * m), ONES(n), 1, { lower, upper, G, h });
    r0.x.approximate = !r0.converged;
    return r0.x;
  }

  const dim = 2 * n;
  const ridge = 1e-9 * Math.max(...cov.map((r, i) => Math.abs(r[i])), 1e-12);

  // S_aug = [[S, -S], [-S, S]] + ridge*I
  const S = Array.from({ length: dim }, () => zeros(dim));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      S[i][j] = cov[i][j];
      S[i][j + n] = -cov[i][j];
      S[i + n][j] = -cov[i][j];
      S[i + n][j + n] = cov[i][j];
    }
  }
  for (let i = 0; i < dim; i++) S[i][i] += ridge;

  // linear: solveQP minimizes (1/2)x'Sx - c'x, so c = -(A'(S wPrev - tau mu) + cost*1)
  const SwPrev = matVec(cov, wPrev);
  const g = SwPrev.map((v, i) => v - tau * mu[i]);
  const c = zeros(dim);
  for (let i = 0; i < n; i++) {
    c[i] = -(g[i] + cost);
    c[i + n] = -(-g[i] + cost);
  }

  // budget: 1'(wPrev + b - s) = 1  ->  1'b - 1's = 1 - 1'wPrev
  const a = zeros(dim);
  for (let i = 0; i < n; i++) { a[i] = 1; a[i + n] = -1; }
  const bRhs = 1 - wPrev.reduce((x, y) => x + y, 0);

  // Per-asset bounds on w become inequality rows in (b, s):
  //    b_i - s_i <=  upper_i - wPrev_i
  //   -b_i + s_i <=  wPrev_i - lower_i
  const Gaug = [];
  const haug = [];
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(upper[i])) {
      const row = zeros(dim); row[i] = 1; row[i + n] = -1;
      Gaug.push(row); haug.push(upper[i] - wPrev[i]);
    }
    const row2 = zeros(dim); row2[i] = -1; row2[i + n] = 1;
    Gaug.push(row2); haug.push(wPrev[i] - lower[i]);
  }
  // Group rows carry over the same way: G w <= h  ->  G(b - s) <= h - G wPrev
  for (let j = 0; j < G.length; j++) {
    const row = zeros(dim);
    for (let i = 0; i < n; i++) { row[i] = G[j][i]; row[i + n] = -G[j][i]; }
    Gaug.push(row);
    haug.push(h[j] - dot(G[j], wPrev));
  }

  const res = solveQP(S, c, a, bRhs, {
    lower: zeros(dim),
    upper: new Array(dim).fill(Infinity),
    G: Gaug, h: haug, maxIter: 800,
  });

  const w = new Array(n);
  for (let i = 0; i < n; i++) w[i] = wPrev[i] + res.x[i] - res.x[i + n];
  // Clamp away arithmetic drift, then restore the budget exactly.
  for (let i = 0; i < n; i++) w[i] = Math.min(Math.max(w[i], lower[i]), Number.isFinite(upper[i]) ? upper[i] : 1);
  const total = w.reduce((x, y) => x + y, 0);
  if (Math.abs(total - 1) > 1e-12 && total > 1e-9) {
    const slack = 1 - total;
    // Spread the residual over assets that have room, so no bound is breached.
    let room = 0;
    for (let i = 0; i < n; i++) {
      room += slack > 0 ? (Number.isFinite(upper[i]) ? upper[i] : 1) - w[i] : w[i] - lower[i];
    }
    if (room > 1e-12) {
      for (let i = 0; i < n; i++) {
        const cap = slack > 0 ? (Number.isFinite(upper[i]) ? upper[i] : 1) - w[i] : w[i] - lower[i];
        w[i] += slack * (cap / room);
      }
    }
  }
  w.approximate = !res.converged;
  return w;
}

// Maximum-Sharpe portfolio that accounts for the cost of getting there.
// Searches risk tolerance exactly as the constrained case does, but scores each
// candidate NET of the cost of trading into it from the current holdings.
function maxSharpeWithTurnoverCost(mu, cov, rf, prevWeights, cost, bounds) {
  const n = mu.length;
  const wPrev = prevWeights && prevWeights.length === n ? prevWeights.slice() : zeros(n);
  const scale = Math.max(...cov.map((r, i) => r[i])) /
                Math.max(1e-9, Math.max(...mu.map((m) => Math.abs(m - rf))));

  const score = (logTau) => {
    let w;
    try { w = efficientWithTurnoverCost(mu, cov, Math.exp(logTau) * scale, wPrev, cost, bounds); }
    catch (e) { return null; }
    const ret = dot(w, mu);
    const vol = Math.sqrt(quadForm(w, cov));
    if (!(vol > 0)) return null;
    let traded = 0;
    for (let i = 0; i < n; i++) traded += Math.abs(w[i] - wPrev[i]);
    // The cost is a one-off drag; comparing it against an annual return means
    // treating it as paid over the coming year, which is what a holding period
    // of roughly a year implies.
    return { weights: w, ret, vol, traded, sharpe: (ret - rf - cost * traded) / vol, approximate: !!w.approximate };
  };

  let best = null;
  for (let i = 0; i <= 48; i++) {
    const r = score(-12 + (24 * i) / 48);
    if (r && (!best || r.sharpe > best.sharpe)) best = r;
  }
  if (!best) return { weights: null, reason: "No portfolio could be formed under these limits." };
  return { ...best, method: "turnover-aware" };
}

// Traces the efficient frontier by sweeping risk tolerance, subject to bounds.
function longOnlyFrontier(mu, cov, nPoints = 60, bounds) {
  const { lower, upper } = normalizeBounds(mu.length, bounds);
  const groups = bounds && bounds.groups;
  const scale = Math.max(...cov.map((r, i) => r[i])) / Math.max(1e-9, Math.max(...mu.map(Math.abs)));
  const pts = [];
  const seen = new Set();
  for (let i = 0; i < nPoints; i++) {
    // Geometric sweep: dense near minimum variance, spread out toward max return.
    const tau = i === 0 ? 0 : scale * Math.pow(10, -3 + (6 * i) / (nPoints - 1));
    let w;
    try { w = efficientLongOnly(mu, cov, tau, { lower, upper, groups }); } catch (e) { continue; }
    const ret = dot(w, mu);
    const vol = Math.sqrt(quadForm(w, cov));
    const key = `${ret.toFixed(6)}|${vol.toFixed(6)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pts.push({ ret, vol, weights: w });
  }
  pts.sort((p, q) => p.vol - q.vol);
  // Keep only genuinely efficient points (return must increase with volatility).
  const eff = [];
  let best = -Infinity;
  for (const p of pts) {
    if (p.ret > best + 1e-12) { eff.push(p); best = p.ret; }
  }
  eff.approximate = eff.some((p) => !!p.weights.approximate);
  return eff;
}

// ---------------------------------------------------------------- sensitivity

// Perturbs expected returns and re-optimizes, to expose estimation risk.
// This is the single biggest weakness of mean-variance optimization
// (Michaud 1989, "the error maximizer"), so it is measured rather than hidden.
function sensitivityAnalysis(mu, cov, rf, relPerturbation = 0.10, nTrials = 200, seed = 12345, bounds = null) {
  const n = mu.length;
  // mulberry32. A plain LCG is unusable here: multiplying a 31-bit state by a
  // ~2^30 constant overflows JavaScript's 2^53 integer precision and silently
  // discards low-order bits, producing badly non-uniform draws. Math.imul does
  // correct 32-bit multiplication, so this generator is sound.
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // The baseline must use the same bounds as the perturbed runs, or turnover
  // is measured against a portfolio the constraints would never have produced.
  const base = maxSharpeLongOnly(mu, cov, rf, bounds);
  if (!base.weights) return null;

  const lo = Array(n).fill(Infinity);
  const hi = Array(n).fill(-Infinity);
  const sum = zeros(n);
  const turnovers = [];
  let ok = 0;

  for (let t = 0; t < nTrials; t++) {
    const muP = mu.map((m) => m * (1 + relPerturbation * (2 * rand() - 1)));
    let res;
    try { res = maxSharpeLongOnly(muP, cov, rf, bounds); } catch (e) { continue; }
    if (!res.weights) continue;
    ok++;
    let turnover = 0;
    for (let i = 0; i < n; i++) {
      const w = res.weights[i];
      if (w < lo[i]) lo[i] = w;
      if (w > hi[i]) hi[i] = w;
      sum[i] += w;
      turnover += Math.abs(w - base.weights[i]);
    }
    turnovers.push(turnover / 2);
  }
  if (ok === 0) return null;
  turnovers.sort((a, b) => a - b);
  return {
    base: base.weights,
    low: lo, high: hi,
    mean: sum.map((v) => v / ok),
    medianTurnover: turnovers[Math.floor(turnovers.length / 2)],
    maxTurnover: turnovers[turnovers.length - 1],
    trials: ok,
  };
}

// ============================================================================
// data-pipeline.js
// Turns real historical price series into the mu / Sigma inputs the optimizer
// needs. Zero dependencies. Runs in Node and the browser.
//
// The key gain over manual assumptions: with an actual return series available,
// the Ledoit-Wolf shrinkage intensity can be DERIVED FROM THE DATA rather than
// guessed by the user.
// ============================================================================

// ---------------------------------------------------------------- CSV parsing

// Minimal RFC-4180-ish splitter: handles quoted fields containing commas.
function splitCSVLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++; }
        else inQuotes = false;
      } else cur += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

function parseNumber(s) {
  if (s == null) return NaN;
  const cleaned = String(s).replace(/[$€£\s,]/g, "").replace(/[()]/g, "");
  if (cleaned === "" || cleaned === "-" || /^(n\/?a|null|nan|#n\/a)$/i.test(cleaned)) return NaN;
  const v = parseFloat(cleaned);
  return Number.isFinite(v) ? v : NaN;
}

// Parses a wide-format price CSV:  Date, AssetA, AssetB, ...
// Returns { dates, names, prices } where prices[t][i] may be NaN.
function parsePriceCSV(text) {
  const lines = String(text).split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length < 2) throw new Error("The file needs a header row and at least one row of prices.");

  const header = splitCSVLine(lines[0]);
  if (header.length < 2) throw new Error("The file needs a date column and at least one price column.");

  const names = header.slice(1).map((h, i) => h || `Column ${i + 2}`);
  const dates = [];
  const prices = [];

  for (let r = 1; r < lines.length; r++) {
    const cells = splitCSVLine(lines[r]);
    if (cells.length < 2) continue;
    const row = names.map((_, i) => parseNumber(cells[i + 1]));
    if (row.every((v) => !Number.isFinite(v))) continue; // wholly blank row
    dates.push(cells[0]);
    prices.push(row);
  }
  if (prices.length < 2) throw new Error("Found fewer than two usable rows of prices.");

  // Chronological order. Many exports are newest-first; detect and flip.
  const t0 = Date.parse(dates[0]);
  const tN = Date.parse(dates[dates.length - 1]);
  if (Number.isFinite(t0) && Number.isFinite(tN) && t0 > tN) {
    dates.reverse();
    prices.reverse();
  }
  return { dates, names, prices };
}

// ---------------------------------------------------------------- returns

// Converts a price matrix to periodic returns.
// method: "log" (default, time-additive) or "simple".
// Rows where any asset is missing are dropped so all series stay aligned —
// covariance requires contemporaneous observations.
function pricesToReturns(prices, method = "log") {
  const T = prices.length;
  const n = prices[0].length;
  const rows = [];
  const droppedRows = [];

  for (let t = 1; t < T; t++) {
    const prev = prices[t - 1];
    const cur = prices[t];
    let usable = true;
    const row = new Array(n);
    for (let i = 0; i < n; i++) {
      const p0 = prev[i];
      const p1 = cur[i];
      if (!Number.isFinite(p0) || !Number.isFinite(p1) || p0 <= 0 || p1 <= 0) { usable = false; break; }
      row[i] = method === "simple" ? p1 / p0 - 1 : Math.log(p1 / p0);
    }
    if (usable) rows.push(row);
    else droppedRows.push(t);
  }
  // Each estimator enforces its own minimum sample size; this step only needs
  // at least one usable observation to have produced anything at all.
  if (rows.length < 1) {
    throw new Error("No period had a usable price for every asset, so no returns could be computed.");
  }
  return { returns: rows, droppedRows, method };
}

const PERIODS = { daily: 252, weekly: 52, monthly: 12, quarterly: 4, annual: 1 };

// Infers observation frequency from the median gap between dates.
function inferFrequency(dates) {
  const ts = dates.map((d) => Date.parse(d)).filter(Number.isFinite);
  if (ts.length < 3) return { frequency: "daily", periodsPerYear: 252, confident: false };
  const gaps = [];
  for (let i = 1; i < ts.length; i++) {
    const g = (ts[i] - ts[i - 1]) / 86400000;
    if (g > 0) gaps.push(g);
  }
  if (gaps.length === 0) return { frequency: "daily", periodsPerYear: 252, confident: false };
  const sorted = [...gaps].sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  let frequency;
  if (med <= 4) frequency = "daily";
  else if (med <= 10) frequency = "weekly";
  else if (med <= 45) frequency = "monthly";
  else if (med <= 120) frequency = "quarterly";
  else frequency = "annual";

  if (frequency === "daily" && gaps.length >= 20) {
    const oneDayFraction = gaps.filter((g) => g <= 1.5).length / gaps.length;
    if (oneDayFraction >= 0.9) {
      return { frequency: "daily-365", periodsPerYear: 365, confident: true, medianGapDays: med };
    }
  }
  return { frequency, periodsPerYear: PERIODS[frequency], confident: true, medianGapDays: med };
}

// ---------------------------------------------------------------- estimators

function colMeans(X) {
  const T = X.length;
  const n = X[0].length;
  const m = new Array(n).fill(0);
  for (let t = 0; t < T; t++) for (let i = 0; i < n; i++) m[i] += X[t][i];
  return m.map((v) => v / T);
}

// Sample covariance. unbiased=true uses 1/(T-1); false uses 1/T (the MLE,
// which is what the Ledoit-Wolf derivation assumes).
function sampleCovariance(X, unbiased = true) {
  const T = X.length;
  const n = X[0].length;
  const m = colMeans(X);
  const denom = unbiased ? T - 1 : T;
  if (denom <= 0) throw new Error("Not enough observations for a covariance estimate.");
  const S = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < n; i++) {
      const di = X[t][i] - m[i];
      for (let j = i; j < n; j++) {
        S[i][j] += di * (X[t][j] - m[j]);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      S[i][j] /= denom;
      S[j][i] = S[i][j];
    }
  }
  return S;
}

// Exponentially weighted covariance (RiskMetrics). Recent observations count
// for more, which matters when volatility regimes shift.
function ewmaCovariance(X, lambda = 0.94) {
  const T = X.length;
  const n = X[0].length;
  if (!(lambda > 0 && lambda < 1)) throw new Error("Decay factor must be between 0 and 1.");
  const m = colMeans(X);
  const S = Array.from({ length: n }, () => new Array(n).fill(0));
  let wSum = 0;
  for (let t = 0; t < T; t++) {
    const w = Math.pow(lambda, T - 1 - t);
    wSum += w;
    for (let i = 0; i < n; i++) {
      const di = X[t][i] - m[i];
      for (let j = i; j < n; j++) S[i][j] += w * di * (X[t][j] - m[j]);
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) { S[i][j] /= wSum; S[j][i] = S[i][j]; }
  }
  return S;
}

// ---------------------------------------------------------------- Ledoit-Wolf

// Ledoit & Wolf (2003), "Improved estimation of the covariance matrix of stock
// returns", with the constant-correlation target.
//
// Shrinks the noisy sample covariance toward a structured target, with the
// intensity chosen to minimise expected squared error:
//     delta* = (pi - rho) / gamma / T
// This is the estimate that could NOT be computed from summary statistics
// alone — it needs the full return series.
function ledoitWolfShrinkage(X) {
  const T = X.length;
  const n = X[0].length;
  if (T < 3) throw new Error("Ledoit-Wolf needs at least three observations.");

  const m = colMeans(X);
  const D = X.map((row) => row.map((v, i) => v - m[i])); // demeaned
  const S = sampleCovariance(X, false);                  // MLE, as the paper assumes
  const sd = S.map((r, i) => Math.sqrt(Math.max(r[i], 0)));

  // Average pairwise correlation -> constant-correlation target F
  let rSum = 0, rCount = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const den = sd[i] * sd[j];
      if (den > 0) { rSum += S[i][j] / den; rCount++; }
    }
  }
  const rbar = rCount > 0 ? rSum / rCount : 0;

  const F = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    F[i][i] = S[i][i];
    for (let j = i + 1; j < n; j++) {
      F[i][j] = F[j][i] = rbar * sd[i] * sd[j];
    }
  }

  // pi: summed asymptotic variances of the sample covariance entries
  const piMat = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let acc = 0;
      for (let t = 0; t < T; t++) {
        const d = D[t][i] * D[t][j] - S[i][j];
        acc += d * d;
      }
      piMat[i][j] = piMat[j][i] = acc / T;
    }
  }
  let pi = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) pi += piMat[i][j];

  // rho: covariance between the target's estimation error and the sample's
  // theta[k][i][j] = (1/T) sum_t (D_tk^2 - S_kk)(D_ti D_tj - S_ij)
  const theta = (k, i, j) => {
    let acc = 0;
    for (let t = 0; t < T; t++) {
      acc += (D[t][k] * D[t][k] - S[k][k]) * (D[t][i] * D[t][j] - S[i][j]);
    }
    return acc / T;
  };
  let rho = 0;
  for (let i = 0; i < n; i++) rho += piMat[i][i];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (sd[i] <= 0 || sd[j] <= 0) continue;
      rho += (rbar / 2) * ((sd[j] / sd[i]) * theta(i, i, j) + (sd[i] / sd[j]) * theta(j, i, j));
    }
  }

  // gamma: squared Frobenius distance between target and sample
  let gamma = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const d = F[i][j] - S[i][j];
      gamma += d * d;
    }
  }

  let delta = gamma > 0 ? ((pi - rho) / gamma) / T : 0;
  delta = Math.max(0, Math.min(1, delta));

  const shrunk = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) shrunk[i][j] = delta * F[i][j] + (1 - delta) * S[i][j];
  }
  return { covariance: shrunk, delta, sample: S, target: F, rbar, pi, rho, gamma };
}

// ---------------------------------------------------------------- annualization

// Scales periodic estimates to annual figures.
// Log returns are time-additive, so the mean scales by k directly. Simple
// returns are compounded: (1+r)^k - 1. Covariance scales by k under the
// standard i.i.d. assumption in both cases.
function annualize(muPeriodic, covPeriodic, periodsPerYear, method = "log") {
  const k = periodsPerYear;
  const mu = method === "simple"
    ? muPeriodic.map((r) => Math.pow(1 + r, k) - 1)
    : muPeriodic.map((r) => r * k);
  const cov = covPeriodic.map((row) => row.map((v) => v * k));
  const vols = cov.map((r, i) => Math.sqrt(Math.max(r[i], 0)));
  const corr = cov.map((row, i) =>
    row.map((v, j) => (vols[i] > 0 && vols[j] > 0 ? v / (vols[i] * vols[j]) : i === j ? 1 : 0))
  );
  return { mu, cov, vols, corr };
}

// ---------------------------------------------------------------- data quality

// Flags the problems that quietly corrupt estimates: too few observations,
// gaps, stale prices, and extreme moves that may be unadjusted splits.
function dataQualityReport(prices, names, dates, returns, periodsPerYear = null) {
  const T = prices.length;
  const n = names.length;
  const warnings = [];
  const perAsset = [];

  for (let i = 0; i < n; i++) {
    const col = prices.map((r) => r[i]);
    const missing = col.filter((v) => !Number.isFinite(v)).length;
    const valid = col.filter(Number.isFinite);
    let stale = 0;
    for (let t = 1; t < col.length; t++) {
      if (Number.isFinite(col[t]) && Number.isFinite(col[t - 1]) && col[t] === col[t - 1]) stale++;
    }
    let extreme = 0;
    if (returns) {
      for (const row of returns) if (Math.abs(row[i]) > 0.25) extreme++;
    }
    const info = {
      name: names[i],
      missing,
      missingPct: T > 0 ? missing / T : 0,
      stalePct: col.length > 1 ? stale / (col.length - 1) : 0,
      extremeMoves: extreme,
      nonPositive: valid.filter((v) => v <= 0).length,
    };
    perAsset.push(info);

    if (info.missingPct > 0.1) {
      warnings.push(`${names[i]}: ${Math.round(info.missingPct * 100)}% of prices are missing.`);
    }
    if (info.stalePct > 0.3) {
      warnings.push(`${names[i]}: ${Math.round(info.stalePct * 100)}% of prices are unchanged from the previous period, which understates its volatility.`);
    }
    if (info.nonPositive > 0) {
      warnings.push(`${names[i]}: contains ${info.nonPositive} zero or negative prices, which cannot be turned into returns.`);
    }
    if (extreme > 0) {
      // A 365-period calendar means a 24/7 market (crypto): there are no
      // splits or dividends to adjust for, so large moves are usually real.
      const tail = periodsPerYear === 365
        ? "On a 24/7 market such as crypto this is often genuine volatility rather than a data error; check the source only if it looks wrong."
        : "Check for unadjusted stock splits or dividends.";
      warnings.push(`${names[i]}: ${extreme} single-period moves above 25%. ${tail}`);
    }
  }

  const obs = returns ? returns.length : 0;
  if (obs < 30) {
    warnings.push(`Only ${obs} observations. Estimates from this little history are very unreliable.`);
  } else if (obs < n * 10) {
    warnings.push(`${obs} observations for ${n} assets. Covariance estimates need roughly 10 observations per asset to be stable; shrinkage is strongly recommended here.`);
  }

  // Expected returns are far harder to estimate than covariance, and the
  // binding constraint is calendar time, not observation count. Sampling a
  // year more finely does not improve the mean estimate at all.
  let years = null;
  if (periodsPerYear && obs > 0) {
    years = obs / periodsPerYear;
    if (years < 3) {
      const example = (0.18 / Math.sqrt(years) * 100).toFixed(0);
      warnings.push(
        `This history covers about ${years.toFixed(1)} year${years >= 1.95 ? "s" : ""}. Expected returns estimated over so short a window carry enormous uncertainty — for an asset with 18% volatility the standard error of the annual return is roughly ${example} percentage points. Treat any resulting Sharpe ratio above about 1.5 as an artefact of estimation error, not a real opportunity.`
      );
    }
  }

  return {
    perAsset, warnings, observations: obs, assets: n, years,
    sufficient: obs >= Math.max(30, n * 2) && (years === null || years >= 3),
  };
}

// ---------------------------------------------------------------- end-to-end

// Full pipeline: CSV text in, optimizer-ready annualized inputs out.
function estimateFromCSV(csvText, opts = {}) {
  const { dates, names, prices } = parsePriceCSV(csvText);
  const method = opts.method || "log";
  const { returns, droppedRows } = pricesToReturns(prices, method);

  const freq = opts.periodsPerYear
    ? { frequency: opts.frequency || "custom", periodsPerYear: opts.periodsPerYear, confident: true }
    : inferFrequency(dates);

  const muPeriodic = colMeans(returns);

  let covPeriodic, shrinkage = null;
  const estimator = opts.estimator || "ledoit-wolf";
  if (estimator === "sample") {
    covPeriodic = sampleCovariance(returns, true);
  } else if (estimator === "ewma") {
    covPeriodic = ewmaCovariance(returns, opts.lambda || 0.94);
  } else {
    const lw = ledoitWolfShrinkage(returns);
    covPeriodic = lw.covariance;
    shrinkage = { delta: lw.delta, rbar: lw.rbar };
  }

  const annual = annualize(muPeriodic, covPeriodic, freq.periodsPerYear, method);
  const quality = dataQualityReport(prices, names, dates, returns, freq.periodsPerYear);
  if (droppedRows.length > 0) {
    quality.warnings.push(`${droppedRows.length} period(s) dropped because at least one asset had no usable price. Covariance requires all assets observed on the same dates.`);
  }

  return {
    names,
    mu: annual.mu,
    cov: annual.cov,
    vols: annual.vols,
    corr: annual.corr,
    frequency: freq,
    observations: returns.length,
    periodStart: dates[0],
    periodEnd: dates[dates.length - 1],
    estimator,
    shrinkage,
    method,
    quality,
  };
}

// ============================================================================
// backtest.js
// Walk-forward out-of-sample testing.
//
// Everything the optimizer reports elsewhere is in-sample: it is scored on the
// same data it was fitted to, which flatters it. This module re-estimates and
// re-optimizes at each rebalance using ONLY data available at that moment, then
// measures what the resulting portfolio actually earned afterwards.
//
// The dominant failure mode in backtesting is look-ahead bias — letting future
// information leak into a past decision. The estimation window is therefore
// strictly [t - lookback, t - 1] and the return earned is at index t. There is
// an explicit test for this: injecting a spike at time T must leave all
// performance before T untouched.
// ============================================================================

// In the browser build every dependency is already in scope.
const _core = () => ({ minVarianceLongOnly, maxSharpeLongOnly, buildCovariance, blackLitterman, sentimentViews });
const _pipe = () => ({ sampleCovariance, ledoitWolfShrinkage, ewmaCovariance, colMeans, pricesToReturns });

// ---------------------------------------------------------------- helpers

function simpleReturnsFrom(prices) {
  const out = [];
  const usable = [];
  for (let t = 1; t < prices.length; t++) {
    const prev = prices[t - 1], cur = prices[t];
    let ok = true;
    const row = new Array(cur.length);
    for (let i = 0; i < cur.length; i++) {
      if (!Number.isFinite(prev[i]) || !Number.isFinite(cur[i]) || prev[i] <= 0 || cur[i] <= 0) { ok = false; break; }
      row[i] = cur[i] / prev[i] - 1;
    }
    if (ok) { out.push(row); usable.push(t); }
  }
  return { returns: out, priceIndex: usable };
}

// ---------------------------------------------------------------- strategies

// Each strategy receives ONLY the trailing window and returns target weights.
const STRATEGIES = {
  equal: {
    label: "Equal weight (1/N)",
    weights: (window, n) => new Array(n).fill(1 / n),
  },
  minvar: {
    label: "Minimum variance",
    weights: (window, n, ctx) => {
      const cov = estimateCovariance(window, ctx);
      return _core().minVarianceLongOnly(cov, ctx.bounds);
    },
  },
  maxsharpe: {
    label: "Maximum Sharpe",
    weights: (window, n, ctx) => {
      const cov = estimateCovariance(window, ctx);
      const mu = _pipe().colMeans(window).map((m) => m * ctx.periodsPerYear);
      const covAnnual = cov.map((row) => row.map((v) => v * ctx.periodsPerYear));
      const r = _core().maxSharpeLongOnly(mu, covAnnual, ctx.rf, ctx.bounds);
      if (!r || !r.weights) return new Array(n).fill(1 / n);
      return r.weights;
    },
  },
  hold: {
    label: "Buy and hold",
    weights: (window, n) => new Array(n).fill(1 / n),
    rebalance: false,
  },
  // Cost-aware variants: the optimizer is told what trading costs and can
  // decline trades whose benefit does not cover them.
  minvarCost: {
    label: "Minimum variance, cost-aware",
    weights: (window, n, ctx) => {
      const cov = estimateCovariance(window, ctx);
      const covAnnual = cov.map((row) => row.map((v) => v * ctx.periodsPerYear));
      const zeroMu = new Array(n).fill(0);
      return efficientWithTurnoverCost(zeroMu, covAnnual, 0, ctx.prevWeights, ctx.tradeCost, ctx.bounds);
    },
  },
  maxsharpeCost: {
    label: "Maximum Sharpe, cost-aware",
    weights: (window, n, ctx) => {
      const cov = estimateCovariance(window, ctx);
      const mu = colMeansOf(window).map((m) => m * ctx.periodsPerYear);
      const covAnnual = cov.map((row) => row.map((v) => v * ctx.periodsPerYear));
      const r = maxSharpeWithTurnoverCost(mu, covAnnual, ctx.rf, ctx.prevWeights, ctx.tradeCost, ctx.bounds);
      if (!r || !r.weights) return new Array(n).fill(1 / n);
      return r.weights;
    },
  },
  // Sentiment-aware: identical estimation to maxsharpe, except the historical
  // mean is used as a Black-Litterman PRIOR rather than the final answer, and
  // an external signal (ctx.currentSignal, e.g. sentiment) supplies a view
  // that nudges it. If no signal is available at a given rebalance, this is
  // mathematically identical to plain maxsharpe (blackLitterman with zero
  // views returns the prior unchanged — see portfolio-core.js).
  maxsharpeSentiment: {
    label: "Maximum Sharpe + sentiment (Black-Litterman)",
    weights: (window, n, ctx) => {
      const cov = estimateCovariance(window, ctx);
      const covAnnual = cov.map((row) => row.map((v) => v * ctx.periodsPerYear));
      const mu0 = colMeansOf(window).map((m) => m * ctx.periodsPerYear);

      let mu = mu0;
      if (ctx.currentSignal) {
        const counts = ctx.currentSignal.map(() => ctx.sentimentArticles || 10);
        const { P, Q, omega } = _core().sentimentViews(mu0, ctx.currentSignal, counts, {
          tiltPerUnit: ctx.sentimentTilt != null ? ctx.sentimentTilt : 0.05,
          baseVariance: ctx.sentimentBaseVariance != null ? ctx.sentimentBaseVariance : 0.04,
        });
        if (P.length > 0) {
          mu = _core().blackLitterman(mu0, covAnnual, P, Q, omega, ctx.sentimentTau || 0.05).mu;
        }
      }
      const r = _core().maxSharpeLongOnly(mu, covAnnual, ctx.rf, ctx.bounds);
      if (!r || !r.weights) return new Array(n).fill(1 / n);
      return r.weights;
    },
  },
};

const colMeansOf = (w) => _pipe().colMeans(w);

function estimateCovariance(window, ctx) {
  const P = _pipe();
  if (ctx.estimator === "sample") return P.sampleCovariance(window, true);
  if (ctx.estimator === "ewma") return P.ewmaCovariance(window, ctx.lambda || 0.94);
  return P.ledoitWolfShrinkage(window).covariance;
}

// ---------------------------------------------------------------- metrics

function maxDrawdown(periodReturns) {
  let wealth = 1, peak = 1, worst = 0, peakIdx = 0, troughIdx = 0, curPeak = 0;
  for (let t = 0; t < periodReturns.length; t++) {
    wealth *= 1 + periodReturns[t];
    if (wealth > peak) { peak = wealth; curPeak = t; }
    const dd = wealth / peak - 1;
    if (dd < worst) { worst = dd; peakIdx = curPeak; troughIdx = t; }
  }
  return { maxDrawdown: worst, peakIndex: peakIdx, troughIndex: troughIdx };
}

function computeMetrics(periodReturns, periodsPerYear, rf = 0) {
  const T = periodReturns.length;
  if (T === 0) return null;

  let wealth = 1;
  for (const r of periodReturns) wealth *= 1 + r;
  const years = T / periodsPerYear;

  // Geometric, not arithmetic: the arithmetic mean overstates what an investor
  // actually compounds, and the gap widens with volatility.
  const annualReturn = years > 0 ? Math.pow(wealth, 1 / years) - 1 : 0;

  const mean = periodReturns.reduce((a, b) => a + b, 0) / T;
  let varSum = 0;
  for (const r of periodReturns) varSum += (r - mean) ** 2;
  const annualVol = Math.sqrt((varSum / Math.max(1, T - 1)) * periodsPerYear);

  const downside = periodReturns.filter((r) => r < 0);
  let dsSum = 0;
  for (const r of downside) dsSum += r * r;
  const downsideVol = downside.length > 0
    ? Math.sqrt((dsSum / T) * periodsPerYear) : 0;

  const dd = maxDrawdown(periodReturns);

  return {
    periods: T,
    years,
    totalReturn: wealth - 1,
    annualReturn,
    annualVol,
    sharpe: annualVol > 0 ? (annualReturn - rf) / annualVol : 0,
    sortino: downsideVol > 0 ? (annualReturn - rf) / downsideVol : 0,
    maxDrawdown: dd.maxDrawdown,
    winRate: periodReturns.filter((r) => r > 0).length / T,
    finalWealth: wealth,
  };
}

// ---------------------------------------------------------------- engine

// prices: array of rows, one per date, one column per asset
// opts.lookback         estimation window length, in periods
// opts.rebalanceEvery   periods between rebalances
// opts.strategy         key of STRATEGIES
// opts.estimator        'ledoit-wolf' | 'sample' | 'ewma'
// opts.bounds           position/group limits passed straight to the optimizer
// opts.costBps          cost in basis points per unit of weight traded
// opts.rf               annual risk-free rate
function walkForward(prices, dates, opts = {}) {
  const lookback = opts.lookback || 252;
  const rebalanceEvery = opts.rebalanceEvery || 21;
  const strategyKey = opts.strategy || "equal";
  const strategy = STRATEGIES[strategyKey];
  if (!strategy) throw new Error(`Unknown strategy "${strategyKey}".`);

  const periodsPerYear = opts.periodsPerYear || 252;
  const costRate = (opts.costBps || 0) / 10000;
  const ctx = {
    estimator: opts.estimator || "ledoit-wolf",
    lambda: opts.lambda,
    bounds: opts.bounds || null,
    rf: opts.rf || 0,
    periodsPerYear,
    tradeCost: costRate,
    prevWeights: null,
  };

  const { returns, priceIndex } = simpleReturnsFrom(prices);
  const n = prices[0].length;
  if (returns.length <= lookback) {
    throw new Error(`Need more than ${lookback} observations to backtest with a ${lookback}-period lookback; only ${returns.length} are usable.`);
  }

  // An optional external signal (e.g. sentiment), one row per return period,
  // available BEFORE that period's return realizes — a leading indicator, not
  // a look-ahead. Must be pre-aligned to `returns`; this backtest does not
  // attempt to re-align it against dropped/missing-price rows itself.
  const signals = opts.signals || null;
  if (signals && signals.length !== returns.length) {
    throw new Error(`signals has ${signals.length} rows but there are ${returns.length} return periods — they must be pre-aligned.`);
  }

  // Log returns are used for estimation (better statistical behaviour); simple
  // returns are used for profit and loss, where they are exactly correct.
  const logReturns = returns.map((row) => row.map((r) => Math.log(1 + r)));

  const portfolioReturns = [];
  const grossReturns = [];
  const outDates = [];
  const weightHistory = [];
  const rebalances = [];
  const turnovers = [];
  const costs = [];

  let weights = null;
  let sinceRebalance = 0;
  let estimationFailures = 0;

  for (let t = lookback; t < returns.length; t++) {
    // ---- decide, using only data strictly before t ----
    const needRebalance = weights === null ||
      (strategy.rebalance !== false && sinceRebalance >= rebalanceEvery);

    if (needRebalance) {
      const window = logReturns.slice(t - lookback, t); // excludes t
      ctx.prevWeights = weights ? weights.slice() : null;
      ctx.currentSignal = signals ? signals[t] : null;
      let target;
      try {
        target = strategy.weights(window, n, ctx);
        if (!target || target.some((w) => !Number.isFinite(w))) throw new Error("non-finite weights");
      } catch (e) {
        estimationFailures++;
        target = weights ? weights.slice() : new Array(n).fill(1 / n);
      }

      const prev = weights || new Array(n).fill(0);
      let traded = 0;
      for (let i = 0; i < n; i++) traded += Math.abs(target[i] - prev[i]);
      const cost = traded * costRate;

      weights = target;
      sinceRebalance = 0;
      rebalances.push({ index: t, date: dates ? dates[priceIndex[t]] : null, weights: target.slice(), traded, cost });
      turnovers.push(traded / 2); // conventional one-way reporting
      costs.push(cost);
    } else {
      costs.push(0);
    }

    // ---- realize the return at t with the weights decided before it ----
    let gross = 0;
    for (let i = 0; i < n; i++) gross += weights[i] * returns[t][i];
    const net = gross - (needRebalance ? rebalances[rebalances.length - 1].cost : 0);

    grossReturns.push(gross);
    portfolioReturns.push(net);
    outDates.push(dates ? dates[priceIndex[t]] : t);
    weightHistory.push(weights.slice());

    // ---- weights drift with the assets until the next rebalance ----
    const grown = weights.map((w, i) => w * (1 + returns[t][i]));
    const total = grown.reduce((a, b) => a + b, 0);
    if (total > 1e-12) weights = grown.map((v) => v / total);
    sinceRebalance++;
  }

  const metrics = computeMetrics(portfolioReturns, periodsPerYear, ctx.rf);
  const grossMetrics = computeMetrics(grossReturns, periodsPerYear, ctx.rf);

  return {
    strategy: strategyKey,
    label: strategy.label,
    dates: outDates,
    returns: portfolioReturns,
    grossReturns,
    weightHistory,
    rebalances,
    metrics,
    grossMetrics,
    totalCost: costs.reduce((a, b) => a + b, 0),
    averageTurnover: turnovers.length ? turnovers.reduce((a, b) => a + b, 0) / turnovers.length : 0,
    rebalanceCount: rebalances.length,
    estimationFailures,
  };
}

// Runs several configurations over identical data so they can be compared
// like for like.
function compareStrategies(prices, dates, configs, shared = {}) {
  const results = [];
  for (const cfg of configs) {
    try {
      const r = walkForward(prices, dates, { ...shared, ...cfg });
      results.push({ name: cfg.name || r.label, ...r });
    } catch (e) {
      results.push({ name: cfg.name || cfg.strategy, error: e.message });
    }
  }
  return results;
}

// Formats a comparison as a plain-text table.
function formatComparison(results) {
  const rows = results.filter((r) => r.metrics);
  if (rows.length === 0) return "No results.";
  const pct = (v) => (v * 100).toFixed(1).padStart(6) + "%";
  const lines = [];
  lines.push("Strategy                      Return     Vol   Sharpe   MaxDD  Turnover");
  lines.push("-".repeat(72));
  for (const r of rows) {
    lines.push(
      r.name.slice(0, 28).padEnd(28) +
      pct(r.metrics.annualReturn) +
      pct(r.metrics.annualVol) +
      r.metrics.sharpe.toFixed(3).padStart(9) +
      pct(r.metrics.maxDrawdown) +
      pct(r.averageTurnover)
    );
  }
  return lines.join("\n");
}

// ---- simulate the browser path ----
function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6d2b79f5)>>>0;let t=Math.imul(s^(s>>>15),1|s);
t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
function normal(r){let sp=null;return()=>{if(sp!==null){const v=sp;sp=null;return v;}
let u,v,q;do{u=2*r()-1;v=2*r()-1;q=u*u+v*v;}while(q===0||q>=1);
const f=Math.sqrt(-2*Math.log(q)/q);sp=v*f;return u*f;};}

const n=4, T=1600, nm=normal(rng(2024));
let px=[[100,100,100,100]];
for(let t=0;t<T;t++){
  const prev=px[px.length-1];
  px.push(prev.map((p,i)=>p*Math.exp(0.0002+(i*0.00005)+0.011*nm())));
}
const dates=px.map((_,t)=>new Date(Date.UTC(2016,0,1)+t*86400000).toISOString().slice(0,10));

let csv="Date,A,B,C,D\n";
px.forEach((row,t)=>{csv+=dates[t]+","+row.map(v=>v.toFixed(4)).join(",")+"\n";});

let failures=0;
const check=(l,c,d="")=>{console.log(`  ${c?"PASS":"FAIL"}  ${l}${d?"  "+d:""}`);if(!c)failures++;};

console.log("--- browser path: CSV import ---");
const parsed = parsePriceCSV(csv);
check("parsePriceCSV works in the bundle", parsed.prices.length===T+1);
const est = estimateFromCSV(csv, {estimator:"ledoit-wolf", method:"log"});
check("estimateFromCSV works in the bundle", est.names.length===4);
check("shrinkage reported", Number.isFinite(est.shrinkage.delta), "delta="+est.shrinkage.delta.toFixed(4));

console.log("\n--- browser path: backtest with inline dependency wiring ---");
const configs=[
  {name:"Equal weight (1/N)",strategy:"equal"},
  {name:"Minimum variance",strategy:"minvar"},
  {name:"Maximum Sharpe",strategy:"maxsharpe"},
];
const results = compareStrategies(parsed.prices, parsed.dates, configs,
  {lookback:500,rebalanceEvery:21,periodsPerYear:252,estimator:"ledoit-wolf",costBps:10,rf:0.02});
check("all three strategies ran", results.every(r=>r.metrics),
  results.filter(r=>!r.metrics).map(r=>r.name+": "+r.error).join("; "));
results.forEach(r=>{ if(r.metrics)
  console.log("        "+r.name.padEnd(22)+" Sharpe "+r.metrics.sharpe.toFixed(3)+
    "  vol "+(r.metrics.annualVol*100).toFixed(1)+"%  DD "+(r.metrics.maxDrawdown*100).toFixed(1)+"%"); });

console.log("\n--- browser path: backtest honours constraints ---");
const constrained = compareStrategies(parsed.prices, parsed.dates,
  [{name:"Max Sharpe capped",strategy:"maxsharpe",bounds:{maxWeight:0.35,groups:[{name:"AB",assets:[0,1],max:0.5}]}}],
  {lookback:500,rebalanceEvery:21,periodsPerYear:252,estimator:"ledoit-wolf",costBps:10,rf:0.02});
const cr = constrained[0];
check("constrained backtest ran", !!cr.metrics, cr.error||"");
if(cr.metrics){
  check("every rebalance respects the per-asset cap",
    cr.rebalances.every(rb=>rb.weights.every(w=>w<=0.35+1e-7)));
  check("every rebalance respects the group cap",
    cr.rebalances.every(rb=>rb.weights[0]+rb.weights[1]<=0.50+1e-7));
}

console.log("\n--- browser path: equity curve construction ---");
const len=results[0].returns.length;
const step=Math.max(1,Math.ceil(len/400));
let wealth=1; const pts=[];
for(let i=0;i<results[0].returns.length;i++){
  wealth*=1+results[0].returns[i];
  if(i%step===0||i===results[0].returns.length-1) pts.push({i,value:wealth});
}
check("equity curve is finite and downsampled", pts.every(p=>Number.isFinite(p.value)) && pts.length<=401,
  pts.length+" points from "+len);
check("final wealth matches the reported total return",
  Math.abs(pts[pts.length-1].value-(1+results[0].metrics.totalReturn))<1e-9);

console.log("\n--- browser path: cost-aware optimization ---");
const prevW=[0.25,0.25,0.25,0.25];
let toLow=null,toHigh=null;
try{
  const wl=efficientWithTurnoverCost([0.10,0.05,0.08,0.06], buildCovariance([0.15,0.06,0.17,0.12],
    [[1,0.1,0.8,0.4],[0.1,1,0.05,0.2],[0.8,0.05,1,0.5],[0.4,0.2,0.5,1]]), 3, prevW, 0.001);
  const wh=efficientWithTurnoverCost([0.10,0.05,0.08,0.06], buildCovariance([0.15,0.06,0.17,0.12],
    [[1,0.1,0.8,0.4],[0.1,1,0.05,0.2],[0.8,0.05,1,0.5],[0.4,0.2,0.5,1]]), 3, prevW, 0.30);
  toLow=wl.reduce((s2,x,i)=>s2+Math.abs(x-prevW[i]),0)/2;
  toHigh=wh.reduce((s2,x,i)=>s2+Math.abs(x-prevW[i]),0)/2;
  check("turnover-cost optimization runs in the bundle", true);
  check("a higher cost trades less in the bundle too", toHigh<=toLow+1e-7,
    (toLow*100).toFixed(1)+"% vs "+(toHigh*100).toFixed(1)+"%");
}catch(e){ check("turnover-cost optimization runs in the bundle", false, e.message); }

try{
  const cr2=compareStrategies(parsed.prices,parsed.dates,
    [{name:"Min variance cost-aware",strategy:"minvarCost"}],
    {lookback:500,rebalanceEvery:21,periodsPerYear:252,estimator:"ledoit-wolf",costBps:50,rf:0.02});
  check("cost-aware backtest strategy runs in the bundle", !!cr2[0].metrics, cr2[0].error||"");
}catch(e){ check("cost-aware backtest strategy runs in the bundle", false, e.message); }

try{
  const f=findFeasiblePoint(4,[1,1,1,1],1,[0.05,0.05,0.05,0.05],[0.35,0.35,0.35,0.35],[],[]);
  check("Phase 1 feasibility search runs in the bundle", f.feasible);
}catch(e){ check("Phase 1 feasibility search runs in the bundle", false, e.message); }

console.log("\n--- browser path: Black-Litterman / sentiment (now wired into the dashboard UI, section 3j) ---");
try{
  const mu0 = [0.08];
  const {P, Q, omega} = sentimentViews(mu0, [0.36], [2], {tiltPerUnit: 0.1512});
  const post = blackLitterman(mu0, [[0.0225]], P, Q, omega);
  check("blackLitterman/sentimentViews run in the bundle", true);
  check("the fixed directional bug (bullish score pulling mu down) does not recur in the bundle",
    post.mu[0] >= mu0[0], `prior ${mu0[0]}, posterior ${post.mu[0]}`);
}catch(e){ check("blackLitterman/sentimentViews run in the bundle", false, e.message); }

console.log("\n--- browser path: approximate-flag propagation ---");
try{
  const mvB = minVarianceLongOnly(buildCovariance([0.15,0.06,0.17,0.19,0.16],
    [[1,-0.1,0.85,0.6,0.1],[-0.1,1,-0.05,0.1,0.2],[0.85,-0.05,1,0.55,0.15],[0.6,0.1,0.55,1,0.2],[0.1,0.2,0.15,0.2,1]]));
  check("minVarianceLongOnly exposes a boolean approximate flag in the bundle",
    typeof mvB.approximate === "boolean");
  const msB = maxSharpeLongOnly([0.10,0.04,0.08,0.09,0.06],
    buildCovariance([0.15,0.06,0.17,0.19,0.16],
    [[1,-0.1,0.85,0.6,0.1],[-0.1,1,-0.05,0.1,0.2],[0.85,-0.05,1,0.55,0.15],[0.6,0.1,0.55,1,0.2],[0.1,0.2,0.15,0.2,1]]), 0.02,
    {maxWeight:0.30});
  check("maxSharpeLongOnly (constrained path) exposes the flag in the bundle",
    typeof msB.approximate === "boolean");
}catch(e){ check("approximate-flag propagation runs in the bundle", false, e.message); }

console.log("\n--- portfolio-optimizer.jsx: verifying the ACTUAL dashboard file, not a copy ---");
// Everything above this point re-implements portfolio-core.js/data-pipeline.js/
// backtest.js independently of portfolio-optimizer.jsx, so it can never catch a
// wiring bug in the real dashboard file — which is exactly what happened here:
// portfolio-optimizer.jsx never actually inlined data-pipeline.js or backtest.js
// at all (estimateFromCSV, walkForward etc. were simply undefined), yet this
// file's "BUNDLE RUNTIME OK" kept passing throughout, because it was only ever
// testing its own hand-maintained duplicate above. This section instead reads
// the real portfolio-optimizer.jsx source, extracts its plain-JS portion (the
// part before the "React application" marker — pure JS, no JSX, so it can run
// directly under Node), and executes THAT to confirm the actual file works.
try {
  const fs = require("fs");
  const path = require("path");
  const vm = require("vm");
  const jsxPath = path.join(__dirname, "portfolio-optimizer.jsx");
  const src = fs.readFileSync(jsxPath, "utf8");
  const endMarker = "// React application";
  const endIdx = src.indexOf(endMarker);
  if (endIdx < 0) throw new Error(`Could not find "${endMarker}" marker — has portfolio-optimizer.jsx's structure changed?`);
  // Skip the leading `import ... from "react"/"recharts"` lines: they are ES
  // module syntax, which vm.runInContext (a classic script, not a module)
  // cannot execute, and the extracted logic below does not need them.
  const startMarker = "const EPS = 1e-10;";
  const startIdx = src.indexOf(startMarker);
  if (startIdx < 0) throw new Error(`Could not find "${startMarker}" — has portfolio-core.js's inlined start point changed?`);
  const required = [
    "minVarianceLongOnly", "maxSharpeLongOnly", "buildCovariance", "blackLitterman", "sentimentViews",
    "parsePriceCSV", "estimateFromCSV", "inferFrequency",
    "walkForward", "compareStrategies", "STRATEGIES",
  ];
  // vm.runInContext attaches top-level `function` declarations to the
  // sandbox object automatically, but NOT top-level `const`/`let` bindings
  // (e.g. STRATEGIES, PERIODS) — those stay in a lexical scope invisible from
  // outside. An epilogue that assigns them onto `this` (the context's global
  // object, from inside the same script) makes them visible afterward too.
  const epilogue = "\n" + required.map((name) => `if (typeof ${name} !== "undefined") { this.${name} = ${name}; }`).join("\n");
  const coreSrc = src.slice(startIdx, endIdx) + epilogue;

  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(coreSrc, sandbox, { filename: "portfolio-optimizer.jsx (extracted, pre-React portion)" });
  const missing = required.filter((name) => typeof sandbox[name] === "undefined");
  check("portfolio-optimizer.jsx actually defines all expected functions (not just this file's copy)",
    missing.length === 0, missing.length ? `missing: ${missing.join(", ")}` : "");

  // A real smoke test against the dashboard's OWN extracted code.
  const smokeCsv = "Date,A,B\n" + Array.from({ length: 300 }, (_, t) => {
    const d = new Date(Date.UTC(2020, 0, 1 + t)).toISOString().slice(0, 10);
    return `${d},${(100 * (1 + t * 0.0005)).toFixed(4)},${(50 * (1 + t * 0.0003)).toFixed(4)}`;
  }).join("\n");

  const est2 = sandbox.estimateFromCSV(smokeCsv);
  check("the dashboard's own estimateFromCSV runs and parses names correctly",
    !!est2 && est2.names.join(",") === "A,B");

  const parsed2 = sandbox.parsePriceCSV(smokeCsv);
  const bt2 = sandbox.compareStrategies(parsed2.prices, parsed2.dates, [
    { name: "Equal weight", strategy: "equal" },
    { name: "Min variance", strategy: "minvar" },
  ], { lookback: 100, rebalanceEvery: 21 });
  check("the dashboard's own walkForward/compareStrategies runs and produces metrics",
    bt2.every((r) => r.metrics), bt2.filter((r) => !r.metrics).map((r) => `${r.name}: ${r.error}`).join("; "));
} catch (e) {
  check("portfolio-optimizer.jsx's real source could be extracted and executed", false, e.message);
}

console.log("\n"+(failures===0?"BUNDLE RUNTIME OK":failures+" FAILURES"));
process.exit(failures?1:0);
