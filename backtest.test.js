const B = require("./backtest.js");
const C = require("./portfolio-core.js");

let pass = 0, fail = 0;
const failures = [];
function ok(name, cond, detail = "") {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; failures.push(name); console.log(`  FAIL  ${name}  ${detail}`); }
}
const close = (a, b, tol = 1e-9) => Math.abs(a - b) < tol;
const section = (s) => console.log(`\n== ${s}`);

// ---------------------------------------------------------------- synthetic data
function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-14)) : s / L[j][j];
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
    let u, v, q;
    do { u = 2 * rng() - 1; v = 2 * rng() - 1; q = u * u + v * v; } while (q === 0 || q >= 1);
    const f = Math.sqrt(-2 * Math.log(q) / q);
    spare = v * f;
    return u * f;
  };
}
function genPrices(mu, cov, T, seed, start = 100) {
  const n = mu.length, L = cholesky(cov), norm = makeNormal(makeRng(seed));
  const prices = [new Array(n).fill(start)];
  for (let t = 0; t < T; t++) {
    const z = Array.from({ length: n }, () => norm());
    const prev = prices[prices.length - 1];
    const row = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k <= i; k++) s += L[i][k] * z[k];
      row[i] = prev[i] * Math.exp(mu[i] + s);
    }
    prices.push(row);
  }
  return prices;
}
function datesFor(len, startYear = 2010) {
  const out = [];
  for (let i = 0; i < len; i++) {
    out.push(new Date(Date.UTC(startYear, 0, 1) + i * 86400000).toISOString().slice(0, 10));
  }
  return out;
}

// ---------------------------------------------------------------- metrics
section("Performance metrics");
{
  // 10% then -10% leaves you below where you started: 1.1 * 0.9 = 0.99
  const m = B.computeMetrics([0.10, -0.10], 2, 0);
  ok("compounding is geometric, not arithmetic", close(m.finalWealth, 0.99, 1e-12));
  ok("total return reflects the compounded loss", close(m.totalReturn, -0.01, 1e-12));
  ok("arithmetic mean of zero still yields a negative annual return", m.annualReturn < 0);
}
{
  const flat = B.computeMetrics(new Array(252).fill(0.0004), 252, 0);
  ok("a constant return series has zero volatility", close(flat.annualVol, 0, 1e-12));
  ok("its annual return compounds correctly",
    close(flat.annualReturn, Math.pow(1.0004, 252) - 1, 1e-9));
  ok("win rate is 100% when every period is positive", close(flat.winRate, 1, 1e-12));
}
{
  const dd = B.maxDrawdown([0.5, -0.5, -0.2, 0.3]);
  // wealth: 1.5, 0.75, 0.60, 0.78  -> peak 1.5, trough 0.60 -> -60%
  ok("maximum drawdown is measured peak to trough", close(dd.maxDrawdown, 0.60 / 1.50 - 1, 1e-12),
    `got ${(dd.maxDrawdown * 100).toFixed(2)}%`);
  ok("the trough index is identified", dd.troughIndex === 2);

  const rising = B.maxDrawdown([0.1, 0.1, 0.1]);
  ok("a series that only rises has no drawdown", close(rising.maxDrawdown, 0, 1e-12));
}
{
  const m = B.computeMetrics([0.02, -0.01, 0.03, -0.02, 0.01], 12, 0);
  ok("Sortino exceeds Sharpe when downside moves are the smaller ones",
    m.sortino > m.sharpe, `sortino ${m.sortino.toFixed(3)} sharpe ${m.sharpe.toFixed(3)}`);
  ok("win rate is counted correctly", close(m.winRate, 3 / 5, 1e-12));
}

// ---------------------------------------------------------------- look-ahead bias
section("Look-ahead bias — the failure that invalidates a backtest");
{
  const cov = [[0.0001, 0.00002], [0.00002, 0.00012]];
  const prices = genPrices([0.0003, 0.0004], cov, 900, 1234);
  const dates = datesFor(prices.length);

  const base = B.walkForward(prices, dates, {
    lookback: 200, rebalanceEvery: 20, strategy: "maxsharpe", periodsPerYear: 252,
  });

  // Inject an enormous move late in the sample. A correct backtest cannot know
  // about it in advance, so every return before that point must be identical.
  const spikeAt = 700;
  const tampered = prices.map((row) => row.slice());
  for (let t = spikeAt; t < tampered.length; t++) tampered[t][0] *= 3;

  const after = B.walkForward(tampered, dates, {
    lookback: 200, rebalanceEvery: 20, strategy: "maxsharpe", periodsPerYear: 252,
  });

  const firstAffected = spikeAt - 200 - 1;
  let identicalBefore = true;
  let firstDiff = -1;
  for (let i = 0; i < firstAffected; i++) {
    if (Math.abs(base.returns[i] - after.returns[i]) > 1e-12) {
      identicalBefore = false; firstDiff = i; break;
    }
  }
  ok("a future price spike does not change any earlier backtest return",
    identicalBefore, `first difference at index ${firstDiff} of ${firstAffected} pre-spike periods`);
  ok("the spike does change later returns, so the test is not vacuous",
    Math.abs(base.metrics.totalReturn - after.metrics.totalReturn) > 1e-6);

  let weightsIdentical = true;
  for (let i = 0; i < firstAffected; i++) {
    for (let k = 0; k < 2; k++) {
      if (Math.abs(base.weightHistory[i][k] - after.weightHistory[i][k]) > 1e-12) weightsIdentical = false;
    }
  }
  ok("weights chosen before the spike are also unchanged", weightsIdentical);
}
{
  // The return earned at t must use weights decided from data before t.
  const prices = genPrices([0.0002, 0.0002], [[0.0001, 0], [0, 0.0001]], 400, 77);
  const r = B.walkForward(prices, null, { lookback: 100, rebalanceEvery: 1000, strategy: "equal", periodsPerYear: 252 });
  const { returns } = B.simpleReturnsFrom(prices);
  const expectedFirst = 0.5 * returns[100][0] + 0.5 * returns[100][1];
  ok("the first backtest return equals the portfolio return at the first out-of-sample period",
    close(r.returns[0], expectedFirst, 1e-12),
    `got ${r.returns[0].toExponential(4)}, expected ${expectedFirst.toExponential(4)}`);
  ok("the number of out-of-sample periods is returns minus lookback",
    r.returns.length === returns.length - 100);
}

// ---------------------------------------------------------------- mechanics
section("Backtest mechanics");
{
  // A single asset held throughout must reproduce that asset's own return.
  const prices = [];
  let p = 100;
  for (let t = 0; t < 300; t++) { prices.push([p]); p *= 1.001; }
  const r = B.walkForward(prices, null, { lookback: 50, rebalanceEvery: 10, strategy: "equal", periodsPerYear: 252 });
  const expected = Math.pow(1.001, prices.length - 1 - 50) - 1;
  ok("a one-asset portfolio reproduces the asset's compounded return",
    close(r.metrics.totalReturn, expected, 1e-9),
    `got ${(r.metrics.totalReturn * 100).toFixed(4)}%, expected ${(expected * 100).toFixed(4)}%`);
  ok("the first rebalance trades from cash into the position",
    close(r.rebalances[0].traded, 1, 1e-12));
  ok("no trading is needed at any rebalance after the first",
    r.rebalances.slice(1).every((rb) => rb.traded < 1e-12),
    `largest later trade ${Math.max(0, ...r.rebalances.slice(1).map((rb) => rb.traded)).toExponential(2)}`);
}
{
  const prices = genPrices([0.0003, 0.0002, 0.0004], [[0.0001,0,0],[0,0.00008,0],[0,0,0.00015]], 800, 555);
  const cheap = B.walkForward(prices, null, { lookback: 150, rebalanceEvery: 5, strategy: "minvar", costBps: 0, periodsPerYear: 252 });
  const dear  = B.walkForward(prices, null, { lookback: 150, rebalanceEvery: 5, strategy: "minvar", costBps: 50, periodsPerYear: 252 });
  ok("transaction costs reduce net returns", dear.metrics.totalReturn < cheap.metrics.totalReturn,
    `0bps ${(cheap.metrics.totalReturn*100).toFixed(2)}% vs 50bps ${(dear.metrics.totalReturn*100).toFixed(2)}%`);
  ok("costs do not affect the gross series",
    close(cheap.grossMetrics.totalReturn, dear.grossMetrics.totalReturn, 1e-12));
  ok("total cost paid is reported", dear.totalCost > 0);

  const rare = B.walkForward(prices, null, { lookback: 150, rebalanceEvery: 100, strategy: "minvar", costBps: 50, periodsPerYear: 252 });
  ok("rebalancing less often costs less", rare.totalCost < dear.totalCost,
    `every 5 periods ${(dear.totalCost*100).toFixed(3)}% vs every 100 ${(rare.totalCost*100).toFixed(3)}%`);
}
{
  const prices = genPrices([0.0003, 0.0002], [[0.0001,0],[0,0.0002]], 600, 909);
  const held = B.walkForward(prices, null, { lookback: 100, rebalanceEvery: 20, strategy: "hold", periodsPerYear: 252 });
  ok("buy and hold rebalances exactly once", held.rebalanceCount === 1);
  ok("buy and hold pays cost only once",
    held.rebalances.length === 1 && held.rebalances[0].traded > 0);

  const drifted = held.weightHistory[held.weightHistory.length - 1];
  ok("weights drift away from equal as prices move",
    Math.abs(drifted[0] - 0.5) > 1e-4, `ended at ${drifted.map(w=>(w*100).toFixed(1)+"%").join(", ")}`);
  ok("drifted weights still sum to 1", close(drifted.reduce((a,b)=>a+b,0), 1, 1e-9));
}
{
  const prices = genPrices([0.0003, 0.0002, 0.0004, 0.0001], [
    [0.00012,0.00002,0.00004,0.00001],
    [0.00002,0.00006,0.00001,0.00001],
    [0.00004,0.00001,0.00018,0.00002],
    [0.00001,0.00001,0.00002,0.00005]], 900, 3131);
  const r = B.walkForward(prices, null, {
    lookback: 200, rebalanceEvery: 20, strategy: "maxsharpe", periodsPerYear: 252,
    bounds: { maxWeight: 0.40, minWeight: 0.05 },
  });
  ok("every rebalance respects the position limits",
    r.rebalances.every((rb) => rb.weights.every((w) => w >= 0.05 - 1e-7 && w <= 0.40 + 1e-7)),
    `worst weight ${Math.max(...r.rebalances.map(rb=>Math.max(...rb.weights))).toFixed(4)}`);
  ok("every rebalance is fully invested",
    r.rebalances.every((rb) => close(rb.weights.reduce((a,b)=>a+b,0), 1, 1e-8)));

  const g = B.walkForward(prices, null, {
    lookback: 200, rebalanceEvery: 20, strategy: "maxsharpe", periodsPerYear: 252,
    bounds: { groups: [{ name: "First two", assets: [0, 1], max: 0.50 }] },
  });
  ok("every rebalance respects the group constraint",
    g.rebalances.every((rb) => rb.weights[0] + rb.weights[1] <= 0.50 + 1e-7));
}
{
  const prices = genPrices([0.0003, 0.0002], [[0.0001,0],[0,0.0001]], 120, 4);
  let msg = "";
  try { B.walkForward(prices, null, { lookback: 500 }); } catch (e) { msg = e.message; }
  ok("too little history for the lookback is refused with both numbers, not a crash",
    msg.includes("500") && msg.includes("120"), msg);
}
{
  // Near-identical assets make covariance nearly singular. The engine must keep
  // running rather than abandoning the backtest partway.
  const prices = genPrices([0.0003, 0.0003, 0.0003],
    [[0.0001,0.0000999,0.0000999],[0.0000999,0.0001,0.0000999],[0.0000999,0.0000999,0.0001]], 500, 8);
  let ran = true, res = null;
  try { res = B.walkForward(prices, null, { lookback: 100, rebalanceEvery: 20, strategy: "minvar", periodsPerYear: 252 }); }
  catch (e) { ran = false; }
  ok("a near-singular covariance does not abort the backtest", ran);
  ok("any estimation failures are counted and reported",
    res && Number.isFinite(res.estimationFailures));
  ok("weights remain valid throughout",
    res && res.rebalances.every((rb) => close(rb.weights.reduce((a,b)=>a+b,0), 1, 1e-6) &&
      rb.weights.every((w) => w >= -1e-9)));
}

// ---------------------------------------------------------------- comparison
section("Comparing strategies on identical data");
{
  const n = 5;
  const load = [[0.010,0.003],[0.003,-0.005],[0.012,0.004],[0.008,-0.002],[0.005,0.007]];
  const idio = [0.00007,0.00002,0.00010,0.00005,0.00008];
  const cov = Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>{
    let v = load[i][0]*load[j][0] + load[i][1]*load[j][1];
    if (i===j) v += idio[i];
    return v;
  }));
  const mu = [0.00035, 0.00012, 0.00040, 0.00025, 0.00018];
  const prices = genPrices(mu, cov, 2600, 20260815);
  const dates = datesFor(prices.length);

  const results = B.compareStrategies(prices, dates, [
    { name: "Equal weight (1/N)", strategy: "equal" },
    { name: "Min variance, sample", strategy: "minvar", estimator: "sample" },
    { name: "Min variance, shrunk", strategy: "minvar", estimator: "ledoit-wolf" },
    { name: "Max Sharpe, sample", strategy: "maxsharpe", estimator: "sample" },
    { name: "Max Sharpe, shrunk", strategy: "maxsharpe", estimator: "ledoit-wolf" },
    { name: "Max Sharpe, capped 30%", strategy: "maxsharpe", estimator: "ledoit-wolf", bounds: { maxWeight: 0.30 } },
  ], { lookback: 500, rebalanceEvery: 21, periodsPerYear: 252, costBps: 10, rf: 0.02 });

  ok("every configuration produced a result", results.every((r) => r.metrics), 
    results.filter(r=>!r.metrics).map(r=>r.name+": "+r.error).join("; "));
  ok("all configurations cover the same number of periods",
    new Set(results.map((r) => r.metrics.periods)).size === 1);

  console.log("");
  console.log(B.formatComparison(results).split("\n").map((l) => "      " + l).join("\n"));

  const byName = Object.fromEntries(results.map((r) => [r.name, r]));
  ok("minimum variance delivers lower volatility than equal weight",
    byName["Min variance, shrunk"].metrics.annualVol < byName["Equal weight (1/N)"].metrics.annualVol,
    `${(byName["Min variance, shrunk"].metrics.annualVol*100).toFixed(2)}% vs ${(byName["Equal weight (1/N)"].metrics.annualVol*100).toFixed(2)}%`);
  ok("equal weight requires the least trading",
    byName["Equal weight (1/N)"].averageTurnover <= byName["Max Sharpe, sample"].averageTurnover,
    `${(byName["Equal weight (1/N)"].averageTurnover*100).toFixed(2)}% vs ${(byName["Max Sharpe, sample"].averageTurnover*100).toFixed(2)}%`);
  ok("capping positions reduces turnover relative to the uncapped optimizer",
    byName["Max Sharpe, capped 30%"].averageTurnover <= byName["Max Sharpe, shrunk"].averageTurnover + 1e-9,
    `${(byName["Max Sharpe, capped 30%"].averageTurnover*100).toFixed(2)}% vs ${(byName["Max Sharpe, shrunk"].averageTurnover*100).toFixed(2)}%`);
}

// ---------------------------------------------------------------- cost awareness
section("Does telling the optimizer about costs actually help?");
{
  // Costs were previously charged after optimizing, so the optimizer produced
  // portfolios it could not afford to reach: 14.8% turnover per rebalance for
  // maximum Sharpe against 2.3% for minimum variance. Putting the cost inside
  // the objective lets it decline trades that do not pay for themselves.
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const sd = (a) => {
    const m = mean(a);
    return Math.sqrt(a.reduce((t, v) => t + (v - m) ** 2, 0) / (a.length - 1));
  };

  const cov = [
    [0.00012, 0.00003, 0.00005, 0.00002, 0.00003, 0.00001],
    [0.00003, 0.00006, 0.00001, 0.00001, 0.00002, 0.00001],
    [0.00005, 0.00001, 0.00020, 0.00003, 0.00004, 0.00002],
    [0.00002, 0.00001, 0.00003, 0.00009, 0.00002, 0.00001],
    [0.00003, 0.00002, 0.00004, 0.00002, 0.00014, 0.00002],
    [0.00001, 0.00001, 0.00002, 0.00001, 0.00002, 0.00007],
  ];
  const mu = [0.00035, 0.00012, 0.00040, 0.00025, 0.00018, 0.00022];

  function paired(blind, aware, costBps, trials = 25) {
    const diffs = [], tBlind = [], tAware = [];
    for (let k = 0; k < trials; k++) {
      const prices = genPrices(mu, cov, 3000, 900 + k * 61);
      const run = (strat) => B.walkForward(prices, null, {
        lookback: 500, rebalanceEvery: 21, strategy: strat,
        estimator: "ledoit-wolf", periodsPerYear: 252, costBps, rf: 0.02,
      });
      let a = null, b = null;
      try { a = run(blind); } catch (e) { /* skip */ }
      try { b = run(aware); } catch (e) { /* skip */ }
      if (a && b && a.metrics && b.metrics) {
        diffs.push(b.metrics.sharpe - a.metrics.sharpe);
        tBlind.push(a.averageTurnover);
        tAware.push(b.averageTurnover);
      }
    }
    const m = mean(diffs);
    return {
      mean: m, t: m / (sd(diffs) / Math.sqrt(diffs.length)),
      n: diffs.length, turnoverBlind: mean(tBlind), turnoverAware: mean(tAware),
    };
  }

  const mv10 = paired("minvar", "minvarCost", 10);
  ok("the cost-aware strategies run at all", mv10.n >= 20, `${mv10.n} paired trials`);
  ok("cost awareness cuts turnover sharply",
    mv10.turnoverAware < mv10.turnoverBlind * 0.6,
    `${(mv10.turnoverBlind * 100).toFixed(1)}% → ${(mv10.turnoverAware * 100).toFixed(1)}%`);
  // Reported as measured. At 10 basis points the improvement is positive and
  // consistent in direction but only marginally significant at this sample size
  // (t around 1.9-3.7 depending on the generated universe). Raising the trial
  // count until it crossed a threshold would be choosing the answer, so the
  // assertion is on the direction, with the t-statistic printed for judgement.
  ok("at 10 basis points minimum variance improves in the expected direction",
    mv10.mean > 0, `+${mv10.mean.toFixed(4)}, t = ${mv10.t.toFixed(2)}`);
  ok("that improvement is at best marginal at this cost level",
    mv10.t < 5, `t = ${mv10.t.toFixed(2)} — real but small next to the 200bps result below`);
  console.log(`      10bps min variance: ${mv10.mean >= 0 ? "+" : ""}${mv10.mean.toFixed(4)} Sharpe, t = ${mv10.t.toFixed(2)}, turnover ${(mv10.turnoverBlind*100).toFixed(1)}% → ${(mv10.turnoverAware*100).toFixed(1)}%`);

  const ms10 = paired("maxsharpe", "maxsharpeCost", 10);
  const ms200 = paired("maxsharpe", "maxsharpeCost", 200);
  console.log(`      10bps max Sharpe:   ${ms10.mean >= 0 ? "+" : ""}${ms10.mean.toFixed(4)} Sharpe, t = ${ms10.t.toFixed(2)}, turnover ${(ms10.turnoverBlind*100).toFixed(1)}% → ${(ms10.turnoverAware*100).toFixed(1)}%`);
  console.log(`      200bps max Sharpe:  ${ms200.mean >= 0 ? "+" : ""}${ms200.mean.toFixed(4)} Sharpe, t = ${ms200.t.toFixed(2)}, turnover ${(ms200.turnoverBlind*100).toFixed(1)}% → ${(ms200.turnoverAware*100).toFixed(1)}%`);

  ok("the benefit grows with the cost being charged",
    ms200.mean > ms10.mean,
    `10bps ${ms10.mean.toFixed(4)} vs 200bps ${ms200.mean.toFixed(4)}`);
  ok("at 200 basis points the improvement is unambiguous", ms200.t > 3,
    `t = ${ms200.t.toFixed(2)}`);
  ok("at 10 basis points the effect on maximum Sharpe is not significant",
    Math.abs(ms10.t) < 2,
    `t = ${ms10.t.toFixed(2)} — cost awareness is worth little when costs are small relative to the noise`);
}

// ---------------------------------------------------------------- robustness
section("One backtest is not evidence");
{
  // The comparison above shows 1/N ahead of every optimized strategy, with
  // maximum Sharpe posting a NEGATIVE out-of-sample Sharpe ratio. That looks
  // like a finding. It is one draw. This section re-runs the same comparison
  // across independent universes to see whether the ranking survives.
  function makeUniverse(n, seed) {
    const r = makeRng(seed);
    const load = Array.from({ length: n }, () => [0.004 + r() * 0.010, (r() - 0.5) * 0.010]);
    const idio = Array.from({ length: n }, () => 0.00002 + r() * 0.00010);
    const cov = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
      let v = load[i][0] * load[j][0] + load[i][1] * load[j][1];
      if (i === j) v += idio[i];
      return v;
    }));
    const mu = Array.from({ length: n }, () => 0.0001 + r() * 0.0004);
    return { mu, cov };
  }

  const TRIALS = 20;
  const sharpes = { equal: [], minvar: [], maxsharpe: [] };
  let equalWins = 0, completed = 0;

  for (let trial = 0; trial < TRIALS; trial++) {
    const { mu, cov } = makeUniverse(5, 1000 + trial * 97);
    const prices = genPrices(mu, cov, 2520, 50000 + trial * 131);
    const got = {};
    let ok = true;
    for (const strat of ["equal", "minvar", "maxsharpe"]) {
      try {
        const r = B.walkForward(prices, null, {
          lookback: 500, rebalanceEvery: 21, strategy: strat,
          estimator: "ledoit-wolf", periodsPerYear: 252, costBps: 10, rf: 0.02,
        });
        got[strat] = r.metrics.sharpe;
      } catch (e) { ok = false; }
    }
    if (!ok) continue;
    completed++;
    for (const k of Object.keys(got)) sharpes[k].push(got[k]);
    if (got.equal >= got.minvar && got.equal >= got.maxsharpe) equalWins++;
  }

  ok("all trials completed", completed === TRIALS, `${completed}/${TRIALS}`);
  ok("the ranking is not stable across universes — 1/N does not always win",
    equalWins > 0 && equalWins < completed,
    `1/N was best in ${equalWins} of ${completed} universes`);

  const spread = (a) => Math.max(...a) - Math.min(...a);
  ok("Sharpe varies enormously between universes, so a single number means little",
    spread(sharpes.maxsharpe) > 0.3,
    `max Sharpe strategy ranged ${Math.min(...sharpes.maxsharpe).toFixed(2)} to ${Math.max(...sharpes.maxsharpe).toFixed(2)}`);
  console.log(`      1/N was best in ${equalWins} of ${completed} universes`);
  console.log(`      max-Sharpe strategy ranged ${Math.min(...sharpes.maxsharpe).toFixed(3)} to ${Math.max(...sharpes.maxsharpe).toFixed(3)} across universes`);
}
{
  // Estimation error is the mechanism, so more estimation data should help the
  // strategy that depends most on estimates.
  function makeUniverse(n, seed) {
    const r = makeRng(seed);
    const load = Array.from({ length: n }, () => [0.004 + r() * 0.010, (r() - 0.5) * 0.010]);
    const idio = Array.from({ length: n }, () => 0.00002 + r() * 0.00010);
    const cov = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
      let v = load[i][0] * load[j][0] + load[i][1] * load[j][1];
      if (i === j) v += idio[i];
      return v;
    }));
    return { mu: Array.from({ length: n }, () => 0.0001 + r() * 0.0004), cov };
  }
  const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  const byLookback = {};
  for (const lookback of [250, 1000]) {
    const vals = [];
    for (let trial = 0; trial < 20; trial++) {
      const { mu, cov } = makeUniverse(5, 1000 + trial * 97);
      const prices = genPrices(mu, cov, 2520, 50000 + trial * 131);
      try {
        const r = B.walkForward(prices, null, {
          lookback, rebalanceEvery: 21, strategy: "maxsharpe",
          estimator: "ledoit-wolf", periodsPerYear: 252, costBps: 10, rf: 0.02,
        });
        vals.push(r.metrics.sharpe);
      } catch (e) { /* skip */ }
    }
    byLookback[lookback] = median(vals);
  }
  ok("a longer estimation window improves the estimate-dependent strategy",
    byLookback[1000] > byLookback[250],
    `median Sharpe ${byLookback[250].toFixed(3)} at 250 periods vs ${byLookback[1000].toFixed(3)} at 1000`);
  console.log(`      max-Sharpe median: ${byLookback[250].toFixed(3)} with 250-period lookback, ${byLookback[1000].toFixed(3)} with 1000`);
  console.log(`      caveat: this data is stationary by construction. Real markets are not,`);
  console.log(`      which punishes estimation harder than anything measured here.`);
}

// ---------------------------------------------------------------- summary
console.log(`\n${"=".repeat(64)}`);
console.log(`${pass} passed, ${fail} failed`);
if (fail) { console.log("Failures:"); failures.forEach((f) => console.log(`  - ${f}`)); }
console.log("=".repeat(64));
process.exit(fail ? 1 : 0);
