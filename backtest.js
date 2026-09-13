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

const CORE = typeof require !== "undefined" ? require("./portfolio-core.js") : null;
const PIPE = typeof require !== "undefined" ? require("./data-pipeline.js") : null;

const _core = () => (typeof minVarianceLongOnly !== "undefined"
  ? { minVarianceLongOnly, maxSharpeLongOnly, buildCovariance, blackLitterman, sentimentViews }
  : CORE);
const efficientWithTurnoverCost = (...args) =>
  (typeof globalThis.efficientWithTurnoverCost === "function" && CORE === null
    ? globalThis.efficientWithTurnoverCost(...args)
    : CORE.efficientWithTurnoverCost(...args));
const maxSharpeWithTurnoverCost = (...args) =>
  (CORE ? CORE.maxSharpeWithTurnoverCost(...args) : globalThis.maxSharpeWithTurnoverCost(...args));
const _pipe = () => (typeof sampleCovariance !== "undefined"
  ? { sampleCovariance, ledoitWolfShrinkage, ewmaCovariance, colMeans, pricesToReturns }
  : PIPE);

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

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    walkForward, compareStrategies, computeMetrics, maxDrawdown,
    simpleReturnsFrom, formatComparison, STRATEGIES,
  };
}
