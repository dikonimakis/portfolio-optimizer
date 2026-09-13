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

module.exports = {
  matVec, dot, solveLinear, invertMatrix, symmetricEigen,
  buildCovariance, validateCorrelation, repairCorrelation,
  shrinkCovariance, bayesSteinMu,
  analyticFrontier, quadForm,
  solveQP, checkKKT,
  minVarianceLongOnly, efficientLongOnly, maxSharpeLongOnly, longOnlyFrontier,
  normalizeBounds, buildGroupConstraints, checkGroupFeasibility, findFeasiblePoint,
  refineByProjectedGradient,
  blackLitterman, sentimentViews,
  efficientWithTurnoverCost, maxSharpeWithTurnoverCost,
  sensitivityAnalysis,
};
