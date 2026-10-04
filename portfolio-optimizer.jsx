import React, { useState, useMemo } from "react";
import {
  ComposedChart, Scatter, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from "recharts";

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
//
// INLINING NOTE: the standalone backtest.js (as tested by backtest.test.js)
// uses a require()/globalThis indirection layer so the same file can run
// under Node or be pasted into a non-module browser bundle without
// duplication. This dashboard is an ES module, where top-level function
// declarations do NOT attach to globalThis, so that indirection does not
// apply here — portfolio-core.js's and data-pipeline.js's functions are
// simply already in scope from being inlined above, so this copy calls them
// directly instead.
// ============================================================================

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
      return minVarianceLongOnly(cov, ctx.bounds);
    },
  },
  maxsharpe: {
    label: "Maximum Sharpe",
    weights: (window, n, ctx) => {
      const cov = estimateCovariance(window, ctx);
      const mu = colMeans(window).map((m) => m * ctx.periodsPerYear);
      const covAnnual = cov.map((row) => row.map((v) => v * ctx.periodsPerYear));
      const r = maxSharpeLongOnly(mu, covAnnual, ctx.rf, ctx.bounds);
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
        const { P, Q, omega } = sentimentViews(mu0, ctx.currentSignal, counts, {
          tiltPerUnit: ctx.sentimentTilt != null ? ctx.sentimentTilt : 0.05,
          baseVariance: ctx.sentimentBaseVariance != null ? ctx.sentimentBaseVariance : 0.04,
        });
        if (P.length > 0) {
          mu = blackLitterman(mu0, covAnnual, P, Q, omega, ctx.sentimentTau || 0.05).mu;
        }
      }
      const r = maxSharpeLongOnly(mu, covAnnual, ctx.rf, ctx.bounds);
      if (!r || !r.weights) return new Array(n).fill(1 / n);
      return r.weights;
    },
  },
};

const colMeansOf = (w) => colMeans(w);

function estimateCovariance(window, ctx) {
  if (ctx.estimator === "sample") return sampleCovariance(window, true);
  if (ctx.estimator === "ewma") return ewmaCovariance(window, ctx.lambda || 0.94);
  return ledoitWolfShrinkage(window).covariance;
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
  const pctFmt = (v) => (v * 100).toFixed(1).padStart(6) + "%";
  const lines = [];
  lines.push("Strategy                      Return     Vol   Sharpe   MaxDD  Turnover");
  lines.push("-".repeat(72));
  for (const r of rows) {
    lines.push(
      r.name.slice(0, 28).padEnd(28) +
      pctFmt(r.metrics.annualReturn) +
      pctFmt(r.metrics.annualVol) +
      r.metrics.sharpe.toFixed(3).padStart(9) +
      pctFmt(r.metrics.maxDrawdown) +
      pctFmt(r.averageTurnover)
    );
  }
  return lines.join("\n");
}

// ============================================================================
// React application
// ============================================================================

const PALETTE = [
  "#d4a13c", "#3fa796", "#8b7ec8", "#d4685e", "#5b8def", "#7d9aa8", "#c98fb0", "#6fae6f",
  "#c96f4a", "#4a9bc9", "#a8b34a", "#9c6f9e",
];

const DEFAULT_ASSETS = [
  { name: "US Stocks", ret: 10, vol: 15 },
  { name: "US Bonds", ret: 4, vol: 6 },
  { name: "Intl Stocks", ret: 8, vol: 17 },
  { name: "REITs", ret: 9, vol: 19 },
  { name: "Gold", ret: 6, vol: 16 },
];

const DEFAULT_CORR = [
  [1, -0.10, 0.85, 0.60, 0.10],
  [-0.10, 1, -0.05, 0.10, 0.20],
  [0.85, -0.05, 1, 0.55, 0.15],
  [0.60, 0.10, 0.55, 1, 0.20],
  [0.10, 0.20, 0.15, 0.20, 1],
];

// Three genuinely different slices of the current market, all real data
// fetched once via Tiingo (see README section 3m for the full findings on
// each). Deliberately NOT more illustrative/invented numbers — this project
// never fabricates return or volatility assumptions, so "more variety" here
// means more real datasets, not more guesses.
const SAMPLES = [
  {
    key: "broad", file: "sample-broad.csv", accent: "#3fa796",
    label: "SPY / AGG / EFA / VNQ / GLD — broad asset classes, 2004–2026",
    note: "US stocks, US bonds, international stocks, REITs, gold. The 22-year test behind README §3m's main real-data finding.",
  },
  {
    key: "sectors", file: "sample-sectors.csv", accent: "#5b8def",
    label: "10 SPDR sector ETFs — US equity sectors, 2015–2026",
    note: "Technology, financials, energy, health care, and more. README §3m's sector-ETF test, which complicated rather than confirmed the broad-asset finding.",
  },
  {
    key: "crypto", file: "sample-crypto.csv", accent: "#c98fb0",
    label: "BTC / ETH / LTC / XRP — real daily prices, 2016–2026",
    note: "README §3m's crypto test: minimum variance won outright at every lookback tried.",
  },
];

const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

function Field({ label, children, hint }) {
  return (
    <div>
      <div className="mb-1.5 text-[10px] uppercase tracking-[0.16em] text-neutral-500">{label}</div>
      {children}
      {hint && <div className="mt-1 text-[10px] leading-snug text-neutral-600">{hint}</div>}
    </div>
  );
}

function WeightBar({ weights, names, bands }) {
  if (!weights) return <div className="text-[12px] text-neutral-600">&mdash;</div>;
  return (
    <div className="space-y-2">
      <div className="flex h-2.5 w-full overflow-hidden rounded-sm bg-neutral-800">
        {weights.map((w, i) =>
          w > 0.0005 ? (
            <div
              key={i}
              style={{ width: `${w * 100}%`, backgroundColor: PALETTE[i % PALETTE.length] }}
              title={`${names[i]}: ${pct(w)}`}
            />
          ) : null
        )}
      </div>
      <div className="space-y-1">
        {weights.map((w, i) =>
          w > 0.001 ? (
            <div key={i} className="flex items-center gap-2 text-[12px]">
              <span
                className="inline-block h-2 w-2 shrink-0 rounded-sm"
                style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
              />
              <span className="flex-1 truncate text-neutral-300">{names[i]}</span>
              <span className="font-mono text-neutral-400">{pct(w)}</span>
              {bands && (
                <span className="w-20 shrink-0 text-right font-mono text-[10px] text-neutral-600">
                  {pct(bands.low[i], 0)}&ndash;{pct(bands.high[i], 0)}
                </span>
              )}
            </div>
          ) : null
        )}
      </div>
    </div>
  );
}

function StatCard({ accent, title, subtitle, ret, vol, sharpe, weights, names, bands, note, approximate }) {
  return (
    <div className="rounded-md border border-neutral-800 bg-neutral-950/40 p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[13px]" style={{ color: accent }}>{title}</span>
            {approximate && (
              <span
                className="rounded-sm border border-[#d4a13c]/50 px-1 py-0.5 text-[9px] uppercase tracking-wide text-[#d4a13c]"
                title="The exact solver did not converge on this problem. This result comes from a feasibility-preserving fallback and may be slightly short of optimal."
              >
                approximate
              </span>
            )}
          </div>
          {subtitle && <div className="mt-0.5 text-[10px] text-neutral-600">{subtitle}</div>}
        </div>
        <div className="shrink-0 text-right font-mono text-[11px] leading-relaxed text-neutral-400">
          <div>{ret != null ? pct(ret) : "—"} return</div>
          <div>{vol != null ? pct(vol) : "—"} vol</div>
          {sharpe != null && <div className="text-neutral-500">SR {sharpe.toFixed(3)}</div>}
        </div>
      </div>
      <WeightBar weights={weights} names={names} bands={bands} />
      {approximate && (
        <div className="mt-3 border-t border-neutral-800/70 pt-2 text-[10px] leading-snug text-[#d4a13c]">
          The exact solver did not settle on this problem (this can happen with unusual limit
          combinations). This weighting is feasible — every limit is respected — but is not
          guaranteed to be the true optimum. Simplifying the position or group limits typically
          restores an exact solve.
        </div>
      )}
      {note && (
        <div className="mt-3 border-t border-neutral-800/70 pt-2 text-[10px] leading-snug text-neutral-600">
          {note}
        </div>
      )}
    </div>
  );
}


const CURVE_COLORS = ["#7d9aa8", "#d4685e", "#3fa796", "#c98fb0", "#5b8def"];

function BacktestPanel({ priceData, backtest, running, error, onRun, lookback, setLookback, equityCurves, bounds, estimation }) {
  if (!priceData) {
    return (
      <div className="flex h-[420px] items-center justify-center rounded-md border border-neutral-800 text-center">
        <div className="max-w-md px-8">
          <div className="text-[14px] text-neutral-300">Out-of-sample testing needs price history</div>
          <p className="mt-3 text-[12px] leading-relaxed text-neutral-500">
            Everything on the frontier tab is in-sample: the optimizer is scored on the same assumptions
            it was given. A backtest re-estimates and re-optimizes at each rebalance using only the data
            available at that moment, then measures what the portfolio actually earned afterwards.
          </p>
          <p className="mt-3 text-[12px] leading-relaxed text-neutral-500">
            Load a price CSV on the left to run one.
          </p>
        </div>
      </div>
    );
  }

  const usable = priceData.prices.length - 1;
  const tooShort = usable <= lookback;
  const periodsPerYear = estimation && estimation.frequency ? estimation.frequency.periodsPerYear : 252;

  return (
    <div className="space-y-5">
      <div className="rounded-md border border-neutral-800 bg-neutral-950/40 p-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">Walk-forward test</div>
            <p className="mt-1.5 max-w-lg text-[11px] leading-snug text-neutral-500">
              Re-estimates and re-optimizes every 21 periods using only prior data, charging 10 basis
              points on every unit of weight traded. Your position and group limits are applied at each
              rebalance.
            </p>
          </div>
          <div className="flex items-end gap-2">
            <div>
              <div className="mb-1 text-[10px] text-neutral-600">Lookback</div>
              <input
                type="number" min="30" step="50" value={lookback}
                onChange={(e) => setLookback(parseInt(e.target.value) || 30)}
                className="w-20 rounded-sm border border-neutral-800 bg-neutral-900 px-2 py-1 text-right text-[12px] font-mono text-neutral-200 outline-none focus:border-[#d4a13c]"
                aria-label="Lookback window in periods"
              />
            </div>
            <button
              onClick={onRun}
              disabled={running || tooShort}
              className="rounded-sm border border-[#3fa796] px-3 py-1.5 text-[11px] text-[#3fa796] hover:bg-[#3fa796]/10 disabled:border-neutral-800 disabled:text-neutral-700"
            >
              {running ? "Running…" : backtest ? "Re-run" : "Run backtest"}
            </button>
          </div>
        </div>
        <div className="mt-2 text-[10px] text-neutral-600">
          {usable.toLocaleString()} usable periods
          {tooShort
            ? ` — not enough for a ${lookback}-period lookback. Lower it or load more history.`
            : ` — about ${((usable - lookback) / periodsPerYear).toFixed(1)} years will be tested out of sample.`}
        </div>
      </div>

      {error && (
        <div className="rounded-sm border border-[#d4685e]/40 bg-[#d4685e]/10 p-3 text-[12px] text-[#e08b83]">
          {error}
        </div>
      )}

      {backtest && equityCurves && (
        <>
          <div className="rounded-md border border-neutral-800 bg-neutral-950/40 p-4">
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={equityCurves.merged} margin={{ top: 10, right: 20, bottom: 18, left: 4 }}>
                <CartesianGrid stroke="#1a1f26" />
                <XAxis
                  dataKey="i" type="number" domain={["dataMin", "dataMax"]}
                  stroke="#3d4650" tick={{ fontSize: 10, fill: "#6b7681" }}
                  tickFormatter={(v) => `${(v / periodsPerYear).toFixed(0)}y`}
                  label={{ value: "Years out of sample", position: "insideBottom", offset: -10, fill: "#4d5761", fontSize: 10 }}
                />
                <YAxis
                  stroke="#3d4650" tick={{ fontSize: 10, fill: "#6b7681" }}
                  tickFormatter={(v) => `${v.toFixed(1)}x`}
                  label={{ value: "Growth of 1", angle: -90, position: "insideLeft", offset: 12, fill: "#4d5761", fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{ background: "#12161b", border: "1px solid #262c33", borderRadius: 3, fontSize: 11 }}
                  formatter={(v, name) => [`${Number(v).toFixed(3)}x`, name]}
                  labelFormatter={(v) => {
                    const row = equityCurves.merged.find((m) => m.i === v);
                    return row && row.date ? row.date : `period ${v}`;
                  }}
                />
                {equityCurves.names.map((nm, k) => (
                  <Line
                    key={k} dataKey={`s${k}`} name={nm} type="monotone"
                    stroke={CURVE_COLORS[k % CURVE_COLORS.length]}
                    strokeWidth={k === 0 ? 1.5 : 2} dot={false}
                    strokeDasharray={k === 0 ? "4 3" : undefined}
                    isAnimationActive={false} connectNulls
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
            <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1.5 text-[10px] text-neutral-500">
              {equityCurves.names.map((nm, k) => (
                <span key={k} className="flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-4" style={{ backgroundColor: CURVE_COLORS[k % CURVE_COLORS.length] }} />
                  {nm}
                </span>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto rounded-md border border-neutral-800">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="border-b border-neutral-800 text-[9px] uppercase tracking-wider text-neutral-600">
                  <th className="px-3 py-2 text-left font-normal">Strategy</th>
                  <th className="px-3 py-2 text-right font-normal">Return</th>
                  <th className="px-3 py-2 text-right font-normal">Vol</th>
                  <th className="px-3 py-2 text-right font-normal">Sharpe</th>
                  <th className="px-3 py-2 text-right font-normal">Max DD</th>
                  <th className="px-3 py-2 text-right font-normal">Turnover</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {backtest.map((r, k) => (
                  <tr key={k} className="border-b border-neutral-800/50 last:border-0">
                    <td className="px-3 py-2 font-sans text-neutral-300">
                      <span className="mr-2 inline-block h-2 w-2 rounded-sm align-middle"
                            style={{ backgroundColor: CURVE_COLORS[k % CURVE_COLORS.length] }} />
                      {r.name}
                    </td>
                    <td className="px-3 py-2 text-right text-neutral-400">{pct(r.metrics.annualReturn)}</td>
                    <td className="px-3 py-2 text-right text-neutral-400">{pct(r.metrics.annualVol)}</td>
                    <td className="px-3 py-2 text-right text-neutral-200">{r.metrics.sharpe.toFixed(3)}</td>
                    <td className="px-3 py-2 text-right text-[#d4685e]">{pct(r.metrics.maxDrawdown)}</td>
                    <td className="px-3 py-2 text-right text-neutral-500">{pct(r.averageTurnover)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="rounded-md border border-[#d4a13c]/30 bg-[#d4a13c]/[0.05] p-4">
            <div className="text-[12px] text-[#d4a13c]">Read this table with suspicion</div>
            <p className="mt-2 text-[11.5px] leading-relaxed text-neutral-400">
              This is one historical path. Testing the same strategies across many simulated markets, the
              maximum-Sharpe result ranged from <span className="font-mono text-neutral-200">−0.59</span> to{" "}
              <span className="font-mono text-neutral-200">+1.49</span> Sharpe depending only on which
              history happened to occur. A single number drawn from that spread carries very little
              information, and the ranking above may reverse on a different period.
            </p>
            <p className="mt-2 text-[11.5px] leading-relaxed text-neutral-400">
              Equal weighting is included because it is the benchmark that matters. Beating it by a small
              margin on one sample is not evidence of skill.
            </p>
          </div>
        </>
      )}

      {!backtest && !running && !error && (
        <div className="flex h-[220px] items-center justify-center rounded-md border border-dashed border-neutral-800">
          <div className="max-w-sm px-6 text-center text-[12px] leading-relaxed text-neutral-500">
            Run the test to compare the optimized portfolios against equal weighting on data neither of
            them was fitted to.
          </div>
        </div>
      )}
    </div>
  );
}

export default function PortfolioOptimizer() {
  const [assets, setAssets] = useState(DEFAULT_ASSETS);
  const [corr, setCorr] = useState(DEFAULT_CORR);
  const [rf, setRf] = useState(2);
  const [covDelta, setCovDelta] = useState(0);
  const [useBayesStein, setUseBayesStein] = useState(false);
  const [sampleSize, setSampleSize] = useState(120);
  const [useSentiment, setUseSentiment] = useState(false);
  const [sentimentTilt, setSentimentTilt] = useState(5);
  const [sentimentArticles, setSentimentArticles] = useState(10);
  const [perturb, setPerturb] = useState(15);
  const [showSensitivity, setShowSensitivity] = useState(false);
  const [estimation, setEstimation] = useState(null);
  const [importError, setImportError] = useState(null);
  const [estimator, setEstimator] = useState("ledoit-wolf");
  const [useLimits, setUseLimits] = useState(false);
  const [maxWeight, setMaxWeight] = useState(30);
  const [minWeight, setMinWeight] = useState(0);
  const [groups, setGroups] = useState([]);
  const [priceData, setPriceData] = useState(null);
  const [view, setView] = useState("frontier");
  const [backtest, setBacktest] = useState(null);
  const [backtestRunning, setBacktestRunning] = useState(false);
  const [backtestError, setBacktestError] = useState(null);
  const [lookback, setLookback] = useState(500);
  const [returnMethod, setReturnMethod] = useState("log");
  const [sampleLoading, setSampleLoading] = useState(null);

  const names = assets.map((a) => a.name);
  const n = assets.length;

  const validation = useMemo(() => {
    try {
      return validateCorrelation(corr);
    } catch (e) {
      return { valid: false, isPSD: false, minEigenvalue: NaN, issues: ["Could not analyse the matrix."] };
    }
  }, [corr]);

  const model = useMemo(() => {
    if (!validation.isPSD) return { error: "invalid-corr" };
    const muRaw = assets.map((a) => (Number.isFinite(a.ret) ? a.ret : 0) / 100);
    const vols = assets.map((a) => (Number.isFinite(a.vol) ? Math.max(a.vol, 0.01) : 1) / 100);
    const rfDec = (Number.isFinite(rf) ? rf : 0) / 100;

    let cov;
    try {
      cov = buildCovariance(vols, corr);
    } catch (e) {
      return { error: "Could not build the covariance matrix." };
    }
    if (covDelta > 0) cov = shrinkCovariance(cov, covDelta);

    let mu = muRaw;
    let shrinkInfo = null;
    if (useBayesStein) {
      try {
        const bs = bayesSteinMu(muRaw, cov, sampleSize);
        mu = bs.mu;
        shrinkInfo = bs;
      } catch (e) { /* fall back to raw estimates */ }
    }

    // Sentiment tilt (Black-Litterman): the shrunk historical mean above is
    // the PRIOR, not the final answer — a per-asset score nudges it without
    // overwriting it. Applied last, after Bayes-Stein, so it tilts whatever
    // estimate the rest of the pipeline already produced.
    let sentimentInfo = null;
    if (useSentiment) {
      const scores = assets.map((a) => (a.sentiment === "" || a.sentiment == null ? NaN : Number(a.sentiment)));
      const counts = assets.map(() => sentimentArticles);
      try {
        const { P, Q, omega, covered } = sentimentViews(mu, scores, counts, {
          tiltPerUnit: sentimentTilt / 100,
          baseVariance: 0.04,
        });
        if (covered.length > 0) {
          const priorMu = mu;
          const post = blackLitterman(mu, cov, P, Q, omega, 0.05);
          sentimentInfo = { covered, priorMu, posteriorMu: post.mu };
          mu = post.mu;
        }
      } catch (e) { /* leave mu untilted */ }
    }

    let analytic = null;
    try { analytic = analyticFrontier(mu, cov, rfDec); } catch (e) { analytic = null; }

    const activeGroups = groups
      .filter((g) => g.assets.length > 0 && (g.max !== "" || g.min !== ""))
      .map((g) => ({
        name: g.name,
        assets: g.assets.filter((i) => i < assets.length),
        ...(g.max !== "" ? { max: parseFloat(g.max) / 100 } : {}),
        ...(g.min !== "" ? { min: parseFloat(g.min) / 100 } : {}),
      }))
      .filter((g) => g.assets.length > 0);

    const bounds = (useLimits || activeGroups.length > 0)
      ? {
          lower: assets.map(() => (useLimits ? minWeight / 100 : 0)),
          upper: assets.map(() => (useLimits ? maxWeight / 100 : 1)),
          groups: activeGroups,
        }
      : null;

    let frontier = [];
    let maxSharpe = null;
    let minVar = null;
    let error = null;
    let constraintError = false;
    try {
      frontier = longOnlyFrontier(mu, cov, 70, bounds);
      maxSharpe = maxSharpeLongOnly(mu, cov, rfDec, bounds);
      const mv = minVarianceLongOnly(cov, bounds);
      const mvRet = dot(mv, mu);
      const mvVol = Math.sqrt(quadForm(mv, cov));
      minVar = { weights: mv, ret: mvRet, vol: mvVol, sharpe: mvVol > 0 ? (mvRet - rfDec) / mvVol : 0, approximate: !!mv.approximate };
    } catch (e) {
      error = e.message;
      constraintError = /cannot be satisfied|exceeds its maximum|at most|at least|fully invested/.test(e.message);
    }

    return { mu, cov, rfDec, analytic, frontier, maxSharpe, minVar, shrinkInfo, sentimentInfo, error, constraintError, bounds };
  }, [assets, corr, rf, covDelta, useBayesStein, sampleSize, validation.isPSD,
      useLimits, maxWeight, minWeight, groups, useSentiment, sentimentTilt, sentimentArticles]);

  const sensitivity = useMemo(() => {
    if (!showSensitivity || model.error || !model.maxSharpe || !model.maxSharpe.weights) return null;
    try {
      return sensitivityAnalysis(model.mu, model.cov, model.rfDec, perturb / 100, 200, 12345, model.bounds);
    } catch (e) {
      return null;
    }
  }, [showSensitivity, model, perturb]);

  const chartData = useMemo(() => {
    if (model.error) return { longOnly: [], analytic: [], assets: [] };
    const longOnly = (model.frontier || []).map((p) => ({ vol: p.vol * 100, ret: p.ret * 100 }));
    let analyticPts = [];
    if (model.analytic) {
      const lo = Math.min(...model.mu) - 0.03;
      const hi = Math.max(...model.mu) + 0.04;
      for (let i = 0; i <= 70; i++) {
        const r = lo + ((hi - lo) * i) / 70;
        const v = model.analytic.varianceAt(r);
        if (v >= 0) analyticPts.push({ vol: Math.sqrt(v) * 100, ret: r * 100 });
      }
      analyticPts.sort((a, b) => a.vol - b.vol);
    }
    const assetPts = assets.map((a, i) => ({ vol: a.vol, ret: model.mu[i] * 100, name: a.name }));
    return { longOnly, analytic: analyticPts, assets: assetPts };
  }, [model, assets]);

  const equityCurves = useMemo(() => {
    if (!backtest) return null;
    const len = backtest[0].returns.length;
    const step = Math.max(1, Math.ceil(len / 400)); // ceil, not floor: floor overshoots the cap
    const series = backtest.map((r) => {
      const pts = [];
      let wealth = 1;
      for (let i = 0; i < r.returns.length; i++) {
        wealth *= 1 + r.returns[i];
        if (i % step === 0 || i === r.returns.length - 1) {
          pts.push({ i, value: wealth, date: r.dates[i] });
        }
      }
      return { name: r.name, points: pts };
    });
    const merged = series[0].points.map((p, idx) => {
      const row = { i: p.i, date: p.date };
      series.forEach((sr, k) => { row[`s${k}`] = sr.points[idx] ? sr.points[idx].value : null; });
      return row;
    });
    return { merged, names: series.map((sr) => sr.name) };
  }, [backtest]);

  function updateAsset(i, field, value) {
    markManual();
    const next = assets.slice();
    next[i] = { ...next[i], [field]: field === "name" ? value : (value === "" ? "" : parseFloat(value)) };
    setAssets(next);
  }

  function addAsset() {
    if (n >= 12) return;
    setAssets([...assets, { name: `Asset ${n + 1}`, ret: 7, vol: 12 }]);
    setCorr([...corr.map((r) => [...r, 0]), [...Array(n).fill(0), 1]]);
  }

  function removeAsset(i) {
    if (n <= 2) return;
    setAssets(assets.filter((_, k) => k !== i));
    setCorr(corr.filter((_, r) => r !== i).map((row) => row.filter((_, c) => c !== i)));
  }

  function updateCorr(i, j, raw) {
    markManual();
    const v = raw === "" || raw === "-" ? 0 : Math.max(-1, Math.min(1, parseFloat(raw) || 0));
    const next = corr.map((r) => [...r]);
    next[i][j] = v;
    next[j][i] = v;
    setCorr(next);
  }

  function addGroup() {
    if (groups.length >= 4) return;
    setGroups([...groups, { name: `Group ${groups.length + 1}`, assets: [], min: "", max: "" }]);
  }
  function updateGroup(i, patch) {
    setGroups(groups.map((g, k) => (k === i ? { ...g, ...patch } : g)));
  }
  function removeGroup(i) {
    setGroups(groups.filter((_, k) => k !== i));
  }
  function toggleGroupAsset(gi, ai) {
    setGroups(groups.map((g, k) => {
      if (k !== gi) return g;
      const has = g.assets.includes(ai);
      return { ...g, assets: has ? g.assets.filter((x) => x !== ai) : [...g.assets, ai].sort((a, b) => a - b) };
    }));
  }

  function runBacktest() {
    if (!priceData) return;
    setBacktestRunning(true);
    setBacktestError(null);
    // Deferred so the browser can paint the running state before the work starts.
    setTimeout(() => {
      try {
        const shared = {
          lookback,
          rebalanceEvery: 21,
          periodsPerYear: estimation && estimation.frequency ? estimation.frequency.periodsPerYear : 252,
          estimator,
          costBps: 10,
          rf: (Number.isFinite(rf) ? rf : 0) / 100,
        };
        const configs = [
          { name: "Equal weight (1/N)", strategy: "equal" },
          { name: "Minimum variance", strategy: "minvar", bounds: model.bounds },
          { name: "Maximum Sharpe", strategy: "maxsharpe", bounds: model.bounds },
          { name: "Min variance, cost-aware", strategy: "minvarCost", bounds: model.bounds },
          { name: "Max Sharpe, cost-aware", strategy: "maxsharpeCost", bounds: model.bounds },
        ];
        const results = compareStrategies(priceData.prices, priceData.dates, configs, shared);
        const usable = results.filter((r) => r.metrics);
        if (usable.length === 0) {
          setBacktestError(results[0] && results[0].error ? results[0].error : "The backtest produced no results.");
          setBacktest(null);
        } else {
          setBacktest(usable);
        }
      } catch (err) {
        setBacktestError(err.message || "The backtest failed.");
        setBacktest(null);
      } finally {
        setBacktestRunning(false);
      }
    }, 30);
  }

  function repairMatrix() {
    setCorr(repairCorrelation(corr).map((r) => r.map((v) => Math.round(v * 1000) / 1000)));
  }

  function applyEstimate(est) {
    const round = (v, d) => Math.round(v * d) / d;
    setAssets(est.names.map((nm, i) => ({
      name: nm,
      ret: round(est.mu[i] * 100, 100),
      vol: round(est.vols[i] * 100, 100),
    })));
    setCorr(est.corr.map((row) => row.map((v) => round(v, 1000))));
    setEstimation(est);
    setImportError(null);
  }

  function loadCSVText(text) {
    try {
      const est = estimateFromCSV(text, { estimator, method: returnMethod });
      const parsed = parsePriceCSV(text);
      if (est.names.length > 12) {
        setImportError(`The file has ${est.names.length} assets; this build handles up to 12. Trim some columns and try again.`);
        return;
      }
      applyEstimate(est);
      setPriceData(parsed);
      setBacktest(null);
      setBacktestError(null);
    } catch (err) {
      setImportError(err.message || "The file could not be parsed.");
    }
  }

  function handleFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onerror = () => setImportError("The file could not be read.");
    reader.onload = (e) => loadCSVText(String(e.target.result));
    reader.readAsText(file);
  }

  function loadSample(path, key) {
    setSampleLoading(key);
    fetch(path)
      .then((r) => {
        if (!r.ok) throw new Error(`Could not fetch ${path} (HTTP ${r.status}).`);
        return r.text();
      })
      .then((text) => loadCSVText(text))
      .catch((err) => setImportError(err.message || "The sample could not be loaded."))
      .finally(() => setSampleLoading(null));
  }

  function clearEstimate() {
    setEstimation(null);
    setImportError(null);
    setPriceData(null);
    setBacktest(null);
    setBacktestError(null);
    setView("frontier");
    setAssets(DEFAULT_ASSETS);
    setCorr(DEFAULT_CORR);
  }

  // Editing an input by hand means the numbers are no longer purely estimated,
  // so the provenance badge must not keep claiming they are.
  function markManual() {
    if (estimation) setEstimation({ ...estimation, edited: true });
  }

  const inputCls =
    "rounded-sm border border-neutral-800 bg-neutral-900 px-2 py-1 text-[12px] text-neutral-200 outline-none focus:border-[#d4a13c]";

  return (
    <div className="min-h-screen bg-[#0a0c0f] text-neutral-200" style={{ fontFamily: "ui-sans-serif, system-ui, sans-serif" }}>
      <div className="mx-auto max-w-[1200px] px-5 py-9 sm:px-8">

        <header className="mb-8 border-b border-neutral-800 pb-6">
          <div className="text-[10px] uppercase tracking-[0.22em] text-[#d4a13c]">Mean-Variance Optimization</div>
          <h1 className="mt-2 text-[28px] leading-tight text-neutral-50 sm:text-[34px]" style={{ fontFamily: "ui-serif, Georgia, serif" }}>
            Efficient Frontier Optimizer
          </h1>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-neutral-400">
            Long-only portfolios are solved exactly with an active-set quadratic programming solver, not
            approximated by random sampling. Inputs are checked for internal consistency, and both covariance
            and expected-return estimates can be shrunk to reduce the estimation error that mean-variance
            optimization is known to amplify.
          </p>
          <div className="mt-5 flex gap-1">
            {[["frontier", "Efficient frontier"], ["backtest", "Out-of-sample test"]].map(([key, label]) => (
              <button
                key={key}
                onClick={() => setView(key)}
                className={`rounded-sm border px-3 py-1.5 text-[11px] ${
                  view === key
                    ? "border-[#d4a13c] bg-[#d4a13c]/10 text-[#d4a13c]"
                    : "border-neutral-800 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </header>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[320px_1fr]">

          {/* ---------------------------------------------- inputs */}
          <div className="space-y-6">

            {/* ---- data source ---- */}
            <div className="rounded-md border border-neutral-800 p-3">
              <div className="mb-2 text-[10px] uppercase tracking-[0.16em] text-neutral-500">Data source</div>

              {!estimation ? (
                <>
                  <p className="mb-1.5 text-[11px] leading-snug text-neutral-500">
                    The figures below are illustrative. Load a price history to estimate them from
                    real data instead.
                  </p>
                  <p className="mb-2.5 text-[10px] leading-snug text-neutral-600">
                    These labels are categories, not tickers — the default weights are what to allocate
                    to each, not what to buy. The standard, liquid ETF for each: US Stocks → SPY,
                    US Bonds → AGG, Intl Stocks → VXUS or EFA, REITs → VNQ, Gold → GLD.
                  </p>
                  <label className="block cursor-pointer rounded-sm border border-dashed border-neutral-700 px-3 py-3 text-center text-[11px] text-neutral-400 hover:border-[#3fa796] hover:text-[#3fa796]">
                    Choose a price CSV
                    <input
                      type="file"
                      accept=".csv,text/csv,text/plain"
                      className="hidden"
                      onChange={(e) => handleFile(e.target.files && e.target.files[0])}
                    />
                  </label>
                  <div className="mt-2 font-mono text-[9.5px] leading-relaxed text-neutral-600">
                    Date,SPY,AGG,GLD<br />
                    2020-01-02,321.55,112.30,143.10
                  </div>
                  <div className="mt-3 space-y-2.5 border-t border-neutral-800 pt-3">
                    <div className="text-[10px] text-neutral-600">
                      Or load a real sample — three different slices of the current market, each
                      fetched once via Tiingo and shipped with this page. No live call, no key needed.
                    </div>
                    {SAMPLES.map((s) => (
                      <div key={s.key}>
                        <button
                          onClick={() => loadSample(s.file, s.key)}
                          disabled={sampleLoading === s.key}
                          style={{ "--accent": s.accent }}
                          className="w-full rounded-sm border border-neutral-800 px-2.5 py-2 text-left text-[11px] text-neutral-400 hover:border-[var(--accent)] hover:text-[var(--accent)] disabled:opacity-50"
                        >
                          {sampleLoading === s.key ? "Loading…" : s.label}
                        </button>
                        <div className="mt-1 text-[10px] leading-snug text-neutral-600">{s.note}</div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-[11px] text-[#3fa796]">
                        {estimation.edited ? "Estimated, then edited" : "Estimated from price history"}
                      </div>
                      <div className="mt-1 font-mono text-[10px] leading-relaxed text-neutral-500">
                        {estimation.observations.toLocaleString()} {estimation.frequency.frequency} returns<br />
                        {estimation.periodStart} to {estimation.periodEnd}
                      </div>
                    </div>
                    <button onClick={clearEstimate} className="shrink-0 text-[10px] text-neutral-600 hover:text-[#d4685e]">
                      clear
                    </button>
                  </div>
                  {estimation.shrinkage && (
                    <div className="mt-2 border-t border-neutral-800 pt-2 text-[10px] leading-snug text-neutral-500">
                      Ledoit-Wolf chose a shrinkage of{" "}
                      <span className="font-mono text-neutral-300">{estimation.shrinkage.delta.toFixed(3)}</span>{" "}
                      from the data itself.
                      {estimation.shrinkage.delta > 0.5
                        ? " That is high, meaning this history is short or noisy relative to the number of assets."
                        : " That is modest, meaning the sample covariance is reasonably well determined."}
                    </div>
                  )}
                  {estimation.edited && (
                    <div className="mt-2 text-[10px] leading-snug text-[#d4a13c]">
                      You have changed some inputs by hand, so these no longer match the file.
                    </div>
                  )}
                </>
              )}

              {importError && (
                <div className="mt-2 rounded-sm border border-[#d4685e]/40 bg-[#d4685e]/10 p-2 text-[10.5px] leading-snug text-[#e08b83]">
                  {importError}
                </div>
              )}

              <div className="mt-3 grid grid-cols-2 gap-2 border-t border-neutral-800 pt-2.5">
                <div>
                  <div className="mb-1 text-[9px] uppercase tracking-wider text-neutral-600">Covariance</div>
                  <select
                    value={estimator}
                    onChange={(e) => setEstimator(e.target.value)}
                    className={`${inputCls} w-full`}
                    aria-label="Covariance estimator"
                  >
                    <option value="ledoit-wolf">Ledoit-Wolf</option>
                    <option value="sample">Sample</option>
                    <option value="ewma">EWMA</option>
                  </select>
                </div>
                <div>
                  <div className="mb-1 text-[9px] uppercase tracking-wider text-neutral-600">Returns</div>
                  <select
                    value={returnMethod}
                    onChange={(e) => setReturnMethod(e.target.value)}
                    className={`${inputCls} w-full`}
                    aria-label="Return calculation method"
                  >
                    <option value="log">Log</option>
                    <option value="simple">Simple</option>
                  </select>
                </div>
              </div>
              <div className="mt-1.5 text-[9.5px] leading-snug text-neutral-600">
                Applied when a file is loaded. Change these, then reload the file to re-estimate.
              </div>
            </div>

            {estimation && estimation.quality && estimation.quality.warnings.length > 0 && (
              <div className="rounded-md border border-[#d4a13c]/30 bg-[#d4a13c]/[0.05] p-3">
                <div className="text-[11px] text-[#d4a13c]">Check your data</div>
                <ul className="mt-1.5 space-y-1">
                  {estimation.quality.warnings.map((w, i) => (
                    <li key={i} className="text-[10.5px] leading-snug text-neutral-400">{w}</li>
                  ))}
                </ul>
              </div>
            )}

            <div>
              <div className="mb-2 flex items-baseline justify-between">
                <span className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">Assets</span>
                <button
                  onClick={addAsset}
                  disabled={n >= 12}
                  className="text-[11px] text-[#3fa796] hover:text-[#5fc4b3] disabled:text-neutral-700"
                >
                  + add asset
                </button>
              </div>
              <div className="mb-1.5 flex gap-1.5 pr-6 text-[9px] uppercase tracking-wider text-neutral-600">
                <span className="ml-3.5 flex-1">Name</span>
                <span className="w-14 text-right">Return %</span>
                <span className="w-14 text-right">Vol %</span>
                {useSentiment && <span className="w-14 text-right">Sent.</span>}
              </div>
              <div className="space-y-1.5">
                {assets.map((a, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <span className="h-2 w-2 shrink-0 rounded-sm" style={{ backgroundColor: PALETTE[i % PALETTE.length] }} />
                    <input
                      value={a.name}
                      onChange={(e) => updateAsset(i, "name", e.target.value)}
                      className={`${inputCls} w-full min-w-0 flex-1`}
                      aria-label={`Asset ${i + 1} name`}
                    />
                    <input
                      type="number"
                      value={a.ret}
                      onChange={(e) => updateAsset(i, "ret", e.target.value)}
                      className={`${inputCls} w-14 shrink-0 text-right font-mono`}
                      aria-label={`${a.name} expected return`}
                    />
                    <input
                      type="number"
                      value={a.vol}
                      onChange={(e) => updateAsset(i, "vol", e.target.value)}
                      className={`${inputCls} w-14 shrink-0 text-right font-mono`}
                      aria-label={`${a.name} volatility`}
                    />
                    {useSentiment && (
                      <input
                        type="number"
                        step="0.1"
                        min="-1"
                        max="1"
                        value={a.sentiment ?? ""}
                        onChange={(e) => updateAsset(i, "sentiment", e.target.value)}
                        placeholder="—"
                        className={`${inputCls} w-14 shrink-0 text-right font-mono`}
                        aria-label={`${a.name} sentiment score`}
                      />
                    )}
                    <button
                      onClick={() => removeAsset(i)}
                      disabled={n <= 2}
                      aria-label={`Remove ${a.name}`}
                      className="shrink-0 px-0.5 text-[14px] text-neutral-600 hover:text-[#d4685e] disabled:text-neutral-800"
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-2 text-[10px] uppercase tracking-[0.16em] text-neutral-500">Correlations</div>
              <div className="overflow-x-auto">
                <table className="text-[10px] font-mono">
                  <tbody>
                    <tr>
                      <td />
                      {names.map((nm, j) => (
                        <td key={j} className="px-0.5 pb-1 text-center text-neutral-600">{nm.slice(0, 4)}</td>
                      ))}
                    </tr>
                    {corr.map((row, i) => (
                      <tr key={i}>
                        <td className="pr-1 text-right text-neutral-600">{names[i] ? names[i].slice(0, 4) : ""}</td>
                        {row.map((val, j) => (
                          <td key={j} className="p-0.5">
                            {i === j ? (
                              <div className="flex h-6 w-10 items-center justify-center text-neutral-700">1</div>
                            ) : j < i ? (
                              <div className="flex h-6 w-10 items-center justify-center text-neutral-800">·</div>
                            ) : (
                              <input
                                type="number"
                                step="0.05"
                                min="-1"
                                max="1"
                                value={val}
                                onChange={(e) => updateCorr(i, j, e.target.value)}
                                aria-label={`Correlation between ${names[i]} and ${names[j]}`}
                                className="h-6 w-10 rounded-sm border border-neutral-800 bg-neutral-900 text-center text-neutral-300 outline-none focus:border-[#d4a13c]"
                              />
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {!validation.isPSD ? (
                <div className="mt-2 rounded-sm border border-[#d4685e]/40 bg-[#d4685e]/10 p-2.5">
                  <div className="text-[11px] font-medium text-[#e08b83]">These correlations are impossible</div>
                  <div className="mt-1 text-[10.5px] leading-snug text-neutral-400">
                    No set of real assets can have all of these correlations at once (smallest eigenvalue{" "}
                    {Number.isFinite(validation.minEigenvalue) ? validation.minEigenvalue.toFixed(3) : "n/a"}).
                    Optimizing on them would produce weights that look precise but mean nothing, so results are paused.
                  </div>
                  <button
                    onClick={repairMatrix}
                    className="mt-2 rounded-sm border border-[#d4685e]/50 px-2 py-1 text-[11px] text-[#e08b83] hover:bg-[#d4685e]/15"
                  >
                    Adjust to the nearest valid matrix
                  </button>
                </div>
              ) : (
                <div className="mt-1.5 text-[10px] text-neutral-600">
                  Consistent · smallest eigenvalue {validation.minEigenvalue.toFixed(3)}
                </div>
              )}
            </div>

            <Field label="Risk-free rate %">
              <input
                type="number"
                step="0.25"
                value={rf}
                onChange={(e) => setRf(parseFloat(e.target.value) || 0)}
                className={`${inputCls} w-20 text-right font-mono`}
              />
            </Field>

            <div className="space-y-4 rounded-md border border-neutral-800 p-3">
              <div className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">Estimation controls</div>

              <Field
                label={`Covariance shrinkage · ${covDelta.toFixed(2)}`}
                hint="Blends your correlations toward their average. Higher values give more stable, less concentrated portfolios."
              >
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={covDelta}
                  onChange={(e) => setCovDelta(parseFloat(e.target.value))}
                  className="w-full accent-[#d4a13c]"
                />
              </Field>

              <div>
                <label className="flex cursor-pointer items-center gap-2 text-[12px] text-neutral-300">
                  <input
                    type="checkbox"
                    checked={useBayesStein}
                    onChange={(e) => setUseBayesStein(e.target.checked)}
                    className="accent-[#3fa796]"
                  />
                  Shrink expected returns (Bayes-Stein)
                </label>
                {useBayesStein && (
                  <div className="mt-2 pl-6">
                    <div className="mb-1 text-[10px] text-neutral-600">Observations behind your estimates</div>
                    <input
                      type="number"
                      min="1"
                      value={sampleSize}
                      onChange={(e) => setSampleSize(parseInt(e.target.value) || 1)}
                      className={`${inputCls} w-20 text-right font-mono`}
                    />
                    {model.shrinkInfo && (
                      <div className="mt-1.5 text-[10px] leading-snug text-neutral-600">
                        {model.shrinkInfo.phi === 0
                          ? "Too few observations to shrink reliably — estimates left unchanged."
                          : `Pulling returns ${pct(model.shrinkInfo.phi, 0)} of the way toward ${pct(model.shrinkInfo.muTarget)}.`}
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="border-t border-neutral-800 pt-3">
                <label className="flex cursor-pointer items-center gap-2 text-[12px] text-neutral-300">
                  <input
                    type="checkbox"
                    checked={useSentiment}
                    onChange={(e) => setUseSentiment(e.target.checked)}
                    className="accent-[#c98fb0]"
                  />
                  Apply sentiment tilt (Black-Litterman)
                </label>
                {useSentiment && (
                  <div className="mt-2 space-y-2 pl-6">
                    <div className="text-[10px] leading-snug text-neutral-600">
                      Score each asset below from -1 (bearish) to +1 (bullish). Blank leaves that
                      asset's estimate untouched — this tilts the return above, it does not
                      replace it. A signal has to be genuinely informative to help: this
                      project's own measurements (README §3l) found a pure-noise signal
                      measurably <em>hurts</em> performance, and a plausible real-world news
                      feed's informativeness was still net negative. Only use this with a signal
                      you have reason to trust.
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-32 shrink-0 text-[10px] text-neutral-600">Tilt per unit score</span>
                      <input
                        type="number" min="1" max="30" step="1"
                        value={sentimentTilt}
                        onChange={(e) => setSentimentTilt(parseFloat(e.target.value) || 0)}
                        className={`${inputCls} w-16 text-right font-mono`}
                      />
                      <span className="text-[10px] text-neutral-600">%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-32 shrink-0 text-[10px] text-neutral-600">Articles per score</span>
                      <input
                        type="number" min="1"
                        value={sentimentArticles}
                        onChange={(e) => setSentimentArticles(parseInt(e.target.value) || 1)}
                        className={`${inputCls} w-16 text-right font-mono`}
                      />
                    </div>
                    {model.sentimentInfo && model.sentimentInfo.covered.length > 0 && (
                      <div className="space-y-0.5 text-[10px] leading-snug text-neutral-500">
                        {model.sentimentInfo.covered.map((idx) => {
                          const delta = (model.sentimentInfo.posteriorMu[idx] - model.sentimentInfo.priorMu[idx]) * 100;
                          return (
                            <div key={idx}>
                              {names[idx]}: {delta >= 0 ? "+" : ""}{delta.toFixed(1)}pp
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="border-t border-neutral-800 pt-3">
                <label className="flex cursor-pointer items-center gap-2 text-[12px] text-neutral-300">
                  <input
                    type="checkbox"
                    checked={useLimits}
                    onChange={(e) => setUseLimits(e.target.checked)}
                    className="accent-[#5b8def]"
                  />
                  Apply position limits
                </label>
                {useLimits && (
                  <div className="mt-2 space-y-2 pl-6">
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-[10px] text-neutral-600">Max per asset</span>
                      <input
                        type="number" min="1" max="100" step="5" value={maxWeight}
                        onChange={(e) => setMaxWeight(parseFloat(e.target.value) || 0)}
                        className={`${inputCls} w-16 text-right font-mono`}
                        aria-label="Maximum weight per asset"
                      />
                      <span className="text-[10px] text-neutral-600">%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-[10px] text-neutral-600">Min per asset</span>
                      <input
                        type="number" min="0" max="100" step="1" value={minWeight}
                        onChange={(e) => setMinWeight(parseFloat(e.target.value) || 0)}
                        className={`${inputCls} w-16 text-right font-mono`}
                        aria-label="Minimum weight per asset"
                      />
                      <span className="text-[10px] text-neutral-600">%</span>
                    </div>
                    <div className="text-[10px] leading-snug text-neutral-600">
                      {minWeight * n > 100
                        ? `A ${minWeight}% minimum across ${n} assets needs ${minWeight * n}% of the portfolio, which is impossible.`
                        : maxWeight * n < 100
                        ? `A ${maxWeight}% cap across ${n} assets allows at most ${maxWeight * n}%, so the portfolio cannot be fully invested.`
                        : `Every asset held between ${minWeight}% and ${maxWeight}%.`}
                    </div>
                  </div>
                )}
              </div>

              <div className="border-t border-neutral-800 pt-3">
                <div className="mb-2 flex items-baseline justify-between">
                  <span className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">Group limits</span>
                  <button
                    onClick={addGroup}
                    disabled={groups.length >= 4}
                    className="text-[11px] text-[#5b8def] hover:text-[#7ba5f5] disabled:text-neutral-700"
                  >
                    + add group
                  </button>
                </div>
                {groups.length === 0 ? (
                  <p className="text-[10px] leading-snug text-neutral-600">
                    Limits across several assets at once, such as "at most 60% in equities".
                    Per-asset caps cannot express this.
                  </p>
                ) : (
                  <div className="space-y-3">
                    {groups.map((g, gi) => (
                      <div key={gi} className="rounded-sm border border-neutral-800 bg-neutral-900/40 p-2">
                        <div className="flex items-center gap-1.5">
                          <input
                            value={g.name}
                            onChange={(e) => updateGroup(gi, { name: e.target.value })}
                            className={`${inputCls} min-w-0 flex-1`}
                            aria-label={`Group ${gi + 1} name`}
                          />
                          <button
                            onClick={() => removeGroup(gi)}
                            className="px-0.5 text-[14px] text-neutral-600 hover:text-[#d4685e]"
                            aria-label={`Remove group ${g.name}`}
                          >
                            &times;
                          </button>
                        </div>
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {assets.map((a, ai) => {
                            const on = g.assets.includes(ai);
                            return (
                              <button
                                key={ai}
                                onClick={() => toggleGroupAsset(gi, ai)}
                                className={`rounded-sm border px-1.5 py-0.5 text-[10px] ${
                                  on
                                    ? "border-[#5b8def] bg-[#5b8def]/15 text-[#8fb4f7]"
                                    : "border-neutral-800 text-neutral-600 hover:border-neutral-700"
                                }`}
                              >
                                {a.name.slice(0, 10)}
                              </button>
                            );
                          })}
                        </div>
                        <div className="mt-1.5 flex items-center gap-2">
                          <span className="text-[10px] text-neutral-600">min</span>
                          <input
                            type="number" min="0" max="100" step="5" placeholder="—"
                            value={g.min}
                            onChange={(e) => updateGroup(gi, { min: e.target.value })}
                            className={`${inputCls} w-14 text-right font-mono`}
                            aria-label={`${g.name} minimum`}
                          />
                          <span className="text-[10px] text-neutral-600">max</span>
                          <input
                            type="number" min="0" max="100" step="5" placeholder="—"
                            value={g.max}
                            onChange={(e) => updateGroup(gi, { max: e.target.value })}
                            className={`${inputCls} w-14 text-right font-mono`}
                            aria-label={`${g.name} maximum`}
                          />
                          <span className="text-[10px] text-neutral-600">%</span>
                        </div>
                        {g.assets.length === 0 && (
                          <div className="mt-1 text-[10px] text-[#d4a13c]">
                            Select at least one asset for this limit to apply.
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div>
                <label className="flex cursor-pointer items-center gap-2 text-[12px] text-neutral-300">
                  <input
                    type="checkbox"
                    checked={showSensitivity}
                    onChange={(e) => setShowSensitivity(e.target.checked)}
                    className="accent-[#8b7ec8]"
                  />
                  Run sensitivity analysis
                </label>
                {showSensitivity && (
                  <div className="mt-2 pl-6">
                    <div className="mb-1 text-[10px] text-neutral-600">Assumed error in returns · ±{perturb}%</div>
                    <input
                      type="range"
                      min="5"
                      max="50"
                      step="5"
                      value={perturb}
                      onChange={(e) => setPerturb(parseInt(e.target.value))}
                      className="w-full accent-[#8b7ec8]"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ---------------------------------------------- results */}
          <div className="space-y-6">
            {view === "backtest" ? (
              <BacktestPanel
                priceData={priceData}
                backtest={backtest}
                running={backtestRunning}
                error={backtestError}
                onRun={runBacktest}
                lookback={lookback}
                setLookback={setLookback}
                equityCurves={equityCurves}
                bounds={model.bounds}
                estimation={estimation}
              />
            ) : model.error === "invalid-corr" ? (
              <div className="flex h-[400px] items-center justify-center rounded-md border border-neutral-800 text-center">
                <div className="max-w-sm px-6">
                  <div className="text-[14px] text-neutral-300">Results paused</div>
                  <div className="mt-2 text-[12px] leading-relaxed text-neutral-500">
                    Fix the correlation matrix on the left to continue.
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="rounded-md border border-neutral-800 bg-neutral-950/40 p-4">
                  <ResponsiveContainer width="100%" height={400}>
                    <ComposedChart margin={{ top: 12, right: 24, bottom: 22, left: 4 }}>
                      <CartesianGrid stroke="#1a1f26" />
                      <XAxis
                        type="number"
                        dataKey="vol"
                        domain={["dataMin - 1", "dataMax + 1"]}
                        stroke="#3d4650"
                        tick={{ fontSize: 10, fill: "#6b7681" }}
                        tickFormatter={(v) => `${v.toFixed(0)}%`}
                        label={{ value: "Volatility (annualized σ)", position: "insideBottom", offset: -12, fill: "#4d5761", fontSize: 10 }}
                      />
                      <YAxis
                        type="number"
                        dataKey="ret"
                        domain={["dataMin - 1", "dataMax + 1"]}
                        stroke="#3d4650"
                        tick={{ fontSize: 10, fill: "#6b7681" }}
                        tickFormatter={(v) => `${v.toFixed(0)}%`}
                        label={{ value: "Expected return", angle: -90, position: "insideLeft", offset: 12, fill: "#4d5761", fontSize: 10 }}
                      />
                      <Tooltip
                        cursor={{ stroke: "#2a3038" }}
                        contentStyle={{ background: "#12161b", border: "1px solid #262c33", borderRadius: 3, fontSize: 11 }}
                        formatter={(v, name) => [`${Number(v).toFixed(2)}%`, name]}
                      />
                      <Line
                        data={chartData.analytic}
                        dataKey="ret"
                        type="monotone"
                        stroke="#5d5236"
                        strokeWidth={1.5}
                        strokeDasharray="4 4"
                        dot={false}
                        name="Shorts allowed"
                        isAnimationActive={false}
                      />
                      <Line
                        data={chartData.longOnly}
                        dataKey="ret"
                        type="monotone"
                        stroke="#d4a13c"
                        strokeWidth={2.5}
                        dot={false}
                        name="Long-only frontier"
                        isAnimationActive={false}
                      />
                      <Scatter data={chartData.assets} fill="#4d5761" name="Individual assets" isAnimationActive={false} />
                      {model.maxSharpe && model.maxSharpe.weights && (
                        <Scatter
                          data={[{ vol: model.maxSharpe.vol * 100, ret: model.maxSharpe.ret * 100 }]}
                          fill="#3fa796"
                          shape="star"
                          name="Max Sharpe"
                          isAnimationActive={false}
                        />
                      )}
                      {model.minVar && (
                        <Scatter
                          data={[{ vol: model.minVar.vol * 100, ret: model.minVar.ret * 100 }]}
                          fill="#d4685e"
                          shape="diamond"
                          name="Min volatility"
                          isAnimationActive={false}
                        />
                      )}
                    </ComposedChart>
                  </ResponsiveContainer>
                  <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1.5 text-[10px] text-neutral-500">
                    <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#d4a13c]" /> long-only frontier (exact)</span>
                    <span className="flex items-center gap-1.5"><span className="inline-block h-0 w-4 border-t border-dashed border-[#6d6046]" /> theoretical, shorts allowed</span>
                    <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-full bg-[#3fa796]" /> max Sharpe</span>
                    <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-full bg-[#d4685e]" /> min volatility</span>
                    <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-full bg-[#4d5761]" /> single assets</span>
                  </div>
                  {model.frontier && model.frontier.approximate && (
                    <div className="mt-1.5 text-[10px] leading-snug text-[#d4a13c]">
                      Part of this frontier comes from an approximate fallback rather than the exact
                      solver (see the "approximate" badge below) — treat that stretch as indicative,
                      not exact.
                    </div>
                  )}
                </div>

                {model.error && model.error !== "invalid-corr" && (
                  <div className="rounded-sm border border-[#d4685e]/40 bg-[#d4685e]/10 p-3">
                    <div className="text-[12px] text-[#e08b83]">
                      {model.constraintError ? "These limits cannot all be met" : "Optimization failed"}
                    </div>
                    <div className="mt-1 text-[11px] leading-snug text-neutral-400">{model.error}</div>
                    {model.constraintError && (
                      <div className="mt-1.5 text-[10px] leading-snug text-neutral-500">
                        Nothing is shown rather than a portfolio that quietly breaks one of your rules.
                      </div>
                    )}
                  </div>
                )}

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <StatCard
                    accent="#3fa796"
                    title="Maximum Sharpe ratio"
                    subtitle={model.maxSharpe && model.maxSharpe.method === "golden-section"
                      ? "Best return per unit of risk, within your limits"
                      : "Best return per unit of risk"}
                    ret={model.maxSharpe ? model.maxSharpe.ret : null}
                    vol={model.maxSharpe ? model.maxSharpe.vol : null}
                    sharpe={model.maxSharpe ? model.maxSharpe.sharpe : null}
                    weights={model.maxSharpe ? model.maxSharpe.weights : null}
                    names={names}
                    bands={sensitivity}
                    note={model.maxSharpe ? model.maxSharpe.reason : null}
                    approximate={!!(model.maxSharpe && model.maxSharpe.approximate)}
                  />
                  <StatCard
                    accent="#d4685e"
                    title="Minimum volatility"
                    subtitle="Lowest achievable risk"
                    ret={model.minVar ? model.minVar.ret : null}
                    vol={model.minVar ? model.minVar.vol : null}
                    sharpe={model.minVar ? model.minVar.sharpe : null}
                    weights={model.minVar ? model.minVar.weights : null}
                    names={names}
                    approximate={!!(model.minVar && model.minVar.approximate)}
                  />
                </div>

                {sensitivity && (
                  <div className="rounded-md border border-[#8b7ec8]/30 bg-[#8b7ec8]/[0.04] p-4">
                    <div className="text-[13px] text-[#a99adb]">How much do these weights actually mean?</div>
                    <div className="mt-2 text-[12px] leading-relaxed text-neutral-400">
                      Re-running the optimization {sensitivity.trials} times with expected returns varied by
                      ±{perturb}% moves a median of{" "}
                      <span className="font-mono text-neutral-200">{pct(sensitivity.medianTurnover, 0)}</span>{" "}
                      of the portfolio, and up to{" "}
                      <span className="font-mono text-neutral-200">{pct(sensitivity.maxTurnover, 0)}</span> in the
                      worst case. The ranges beside each weight above show where it landed across those runs.
                    </div>
                    <div className="mt-3 border-t border-[#8b7ec8]/20 pt-2.5 text-[10.5px] leading-snug text-neutral-500">
                      {sensitivity.medianTurnover > 0.25
                        ? "These allocations are highly sensitive to your return assumptions. Treat the weights as one scenario among many rather than a precise answer — raising covariance shrinkage or enabling Bayes-Stein will stabilise them."
                        : "These allocations hold up reasonably well under estimation error."}
                    </div>
                  </div>
                )}

                {model.analytic && model.analytic.tanWeights && (
                  <details className="rounded-md border border-neutral-800 p-4">
                    <summary className="cursor-pointer text-[12px] text-neutral-400 hover:text-neutral-200">
                      Theoretical tangency portfolio (short positions allowed)
                    </summary>
                    <div className="mt-3 font-mono text-[11px] text-neutral-500">
                      {pct(model.analytic.tanRet)} return · {pct(model.analytic.tanVol)} vol
                    </div>
                    <div className="mt-2 space-y-1">
                      {model.analytic.tanWeights.map((w, i) => (
                        <div key={i} className="flex items-center gap-2 text-[12px]">
                          <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: PALETTE[i % PALETTE.length] }} />
                          <span className="flex-1 text-neutral-400">{names[i]}</span>
                          <span className={`font-mono ${w < 0 ? "text-[#d4685e]" : "text-neutral-400"}`}>{pct(w)}</span>
                        </div>
                      ))}
                    </div>
                    <div className="mt-3 text-[10px] leading-snug text-neutral-600">
                      Closed-form solution via matrix inversion. Negative weights are short positions. Shown for
                      comparison — the long-only results above are what most investors can actually hold.
                    </div>
                  </details>
                )}
              </>
            )}
          </div>
        </div>

        <footer className="mt-10 border-t border-neutral-800 pt-4 text-[10px] leading-relaxed text-neutral-600">
          Long-only portfolios solved exactly by active-set quadratic programming; maximum Sharpe via the
          Cornuejols–Tütüncü transformation. Correlation matrices are checked for positive semi-definiteness
          before use. Covariance shrinkage targets constant correlation; expected-return shrinkage follows
          Jorion (1986). Default figures are illustrative long-run asset-class assumptions, not market data —
          replace them with your own estimates. This is an analytical tool, not investment advice.
        </footer>
      </div>
    </div>
  );
}
