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
  gaps.sort((a, b) => a - b);
  const med = gaps[Math.floor(gaps.length / 2)];
  let frequency;
  if (med <= 4) frequency = "daily";
  else if (med <= 10) frequency = "weekly";
  else if (med <= 45) frequency = "monthly";
  else if (med <= 120) frequency = "quarterly";
  else frequency = "annual";
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
      warnings.push(`${names[i]}: ${extreme} single-period moves above 25%. Check for unadjusted stock splits or dividends.`);
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

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    splitCSVLine, parseNumber, parsePriceCSV,
    pricesToReturns, inferFrequency, PERIODS,
    colMeans, sampleCovariance, ewmaCovariance, ledoitWolfShrinkage,
    annualize, dataQualityReport, estimateFromCSV,
  };
}
