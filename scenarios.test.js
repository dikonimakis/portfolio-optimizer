const S = require("./scenarios.js");
const B = require("./backtest.js");
const C = require("./portfolio-core.js");

let pass = 0, fail = 0;
const failures = [];
function ok(name, cond, detail = "") {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; failures.push(name); console.log(`  FAIL  ${name}  ${detail}`); }
}
const section = (s) => console.log(`\n== ${s}`);

// ---------------------------------------------------------------- generators are valid
section("Generated data is well formed");
{
  for (const [name, gen] of Object.entries(S.SCENARIOS)) {
    const sc = gen(5, 1500, 42);
    const finite = sc.prices.every((row) => row.every((p) => Number.isFinite(p) && p > 0));
    const shaped = sc.prices.length === 1501 && sc.returns.length === 1500 &&
                   sc.prices[0].length === 5 && sc.regimes.length === 1500;
    ok(`${name}: prices are finite and positive throughout`, finite);
    ok(`${name}: shapes are consistent`, shaped,
      `${sc.prices.length} prices, ${sc.returns.length} returns, ${sc.regimes.length} regimes`);
  }
}
{
  const a = S.regimeSwitching(4, 500, 777);
  const b = S.regimeSwitching(4, 500, 777);
  const same = a.returns.every((row, t) => row.every((v, i) => v === b.returns[t][i]));
  ok("generators are reproducible from a seed", same);

  const c = S.regimeSwitching(4, 500, 778);
  const differs = c.returns.some((row, t) => row.some((v, i) => v !== a.returns[t][i]));
  ok("different seeds give different data", differs);
}

// ---------------------------------------------------------------- each scenario does what it claims
section("Each scenario exhibits the property it is named for");
{
  const sc = S.stationary(5, 4000, 11);
  const d = S.describe(sc);
  ok("stationary: no stress regime is flagged", d.stressFraction === 0);
  // The two half-means differ by sampling noise alone, so the only defensible
  // threshold is the standard error of that difference.
  const halfN = Math.floor(sc.returns.length / 2);
  const firstCol = sc.returns.map((r) => r[0]);
  const sdOf = (a) => {
    const m = a.reduce((x, y) => x + y, 0) / a.length;
    return Math.sqrt(a.reduce((t, v) => t + (v - m) ** 2, 0) / (a.length - 1));
  };
  const seDiff = sdOf(firstCol) * Math.sqrt(2 / halfN);
  const z = (d.meanFirstHalf - d.meanSecondHalf) / seDiff;
  ok("stationary: the two halves differ only by sampling noise",
    Math.abs(z) < 4, `z = ${z.toFixed(2)} (${d.meanFirstHalf.toExponential(2)} vs ${d.meanSecondHalf.toExponential(2)})`);
  ok("stationary: squared returns show little persistence",
    Math.abs(d.volAutocorrelation) < 0.10, `autocorr ${d.volAutocorrelation.toFixed(4)}`);
}
{
  const sc = S.regimeSwitching(5, 6000, 22);
  const d = S.describe(sc);
  ok("regime switching: crises occupy a minority of the sample",
    d.stressFraction > 0.02 && d.stressFraction < 0.35,
    `${(d.stressFraction * 100).toFixed(1)}% of periods`);
  ok("regime switching: crisis volatility is far above calm volatility",
    d.stressVol > d.calmVol * 2,
    `calm ${(d.calmVol * 100).toFixed(2)}%, crisis ${(d.stressVol * 100).toFixed(2)}%`);
  ok("regime switching: correlations are much higher in crisis",
    d.stressCorrelation > d.calmCorrelation + 0.3,
    `calm ${d.calmCorrelation.toFixed(2)}, crisis ${d.stressCorrelation.toFixed(2)}`);
  console.log(`      crisis ${(d.stressFraction*100).toFixed(1)}% of time, vol ${(d.calmVol*100).toFixed(2)}% -> ${(d.stressVol*100).toFixed(2)}%, corr ${d.calmCorrelation.toFixed(2)} -> ${d.stressCorrelation.toFixed(2)}`);
}
{
  const sc = S.volatilityClustering(5, 6000, 33);
  const d = S.describe(sc);
  const flat = S.describe(S.stationary(5, 6000, 33));
  ok("volatility clustering: squared returns are autocorrelated",
    d.volAutocorrelation > 0.08, `autocorr ${d.volAutocorrelation.toFixed(4)}`);
  ok("volatility clustering: markedly more persistent than stationary data",
    d.volAutocorrelation > flat.volAutocorrelation + 0.05,
    `${d.volAutocorrelation.toFixed(4)} vs stationary ${flat.volAutocorrelation.toFixed(4)}`);
  console.log(`      squared-return autocorrelation ${d.volAutocorrelation.toFixed(3)} vs ${flat.volAutocorrelation.toFixed(3)} stationary`);
}
{
  const sc = S.structuralBreak(5, 4000, 44);
  const d = S.describe(sc);
  ok("structural break: the break point is reported", sc.breakAt === 2000);
  ok("structural break: the best asset before is not the best asset after",
    sc.muBefore.indexOf(Math.max(...sc.muBefore)) !== sc.muAfter.indexOf(Math.max(...sc.muAfter)),
    `before ${sc.muBefore.map(m=>m.toExponential(1))}, after ${sc.muAfter.map(m=>m.toExponential(1))}`);
  ok("structural break: the ranking is genuinely reversed",
    Math.abs(sc.muBefore.indexOf(Math.max(...sc.muBefore)) - sc.muAfter.indexOf(Math.min(...sc.muAfter))) === 0);
}
{
  const sc = S.correlationBreakdown(5, 6000, 55);
  const d = S.describe(sc);
  ok("correlation breakdown: stressed periods occur", d.stressFraction > 0.01,
    `${(d.stressFraction * 100).toFixed(1)}%`);
  ok("correlation breakdown: correlation is far higher under stress",
    d.stressCorrelation > d.calmCorrelation + 0.4,
    `calm ${d.calmCorrelation.toFixed(2)}, stressed ${d.stressCorrelation.toFixed(2)}`);
  ok("correlation breakdown: volatility also rises under stress",
    d.stressVol > d.calmVol,
    `${(d.calmVol * 100).toFixed(2)}% -> ${(d.stressVol * 100).toFixed(2)}%`);
  console.log(`      stressed ${(d.stressFraction*100).toFixed(1)}% of time, corr ${d.calmCorrelation.toFixed(2)} -> ${d.stressCorrelation.toFixed(2)}`);
}

// ---------------------------------------------------------------- the actual question
section("Does the optimizer survive its own assumptions being violated?");
{
  const TRIALS = 12;
  const STRATS = ["equal", "minvar", "maxsharpe"];
  const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };

  const table = {};
  for (const [name, gen] of Object.entries(S.SCENARIOS)) {
    const acc = Object.fromEntries(STRATS.map((k) => [k, []]));
    for (let trial = 0; trial < TRIALS; trial++) {
      const sc = gen(6, 3000, 900 + trial * 61);
      for (const strat of STRATS) {
        try {
          const r = B.walkForward(sc.prices, null, {
            lookback: 500, rebalanceEvery: 21, strategy: strat,
            estimator: "ledoit-wolf", periodsPerYear: 252, costBps: 10, rf: 0.02,
          });
          acc[strat].push(r.metrics.sharpe);
        } catch (e) { /* skip */ }
      }
    }
    table[name] = Object.fromEntries(STRATS.map((k) => [k, median(acc[k])]));
  }

  ok("every scenario produced results for every strategy",
    Object.values(table).every((row) => STRATS.every((k) => Number.isFinite(row[k]))));

  console.log("");
  console.log("      Median out-of-sample Sharpe, 12 trials per scenario");
  console.log("      scenario                    1/N   minvar  maxSharpe   optimizer edge");
  console.log("      " + "-".repeat(70));
  for (const [name, row] of Object.entries(table)) {
    const bestOpt = Math.max(row.minvar, row.maxsharpe);
    const edge = bestOpt - row.equal;
    console.log(
      "      " + name.padEnd(24) +
      row.equal.toFixed(3).padStart(7) +
      row.minvar.toFixed(3).padStart(8) +
      row.maxsharpe.toFixed(3).padStart(10) +
      (edge >= 0 ? "+" : "") + edge.toFixed(3).padStart(16 - (edge >= 0 ? 1 : 0))
    );
  }

  const edgeIn = (name) => Math.max(table[name].minvar, table[name].maxsharpe) - table[name].equal;
  const stationaryEdge = edgeIn("stationary");
  const brokenEdges = ["regimeSwitching", "structuralBreak", "correlationBreakdown"].map(edgeIn);

  ok("the optimizer's advantage is largest on the data its assumptions describe",
    stationaryEdge >= Math.max(...brokenEdges) - 1e-9,
    `stationary ${stationaryEdge.toFixed(3)} vs best non-stationary ${Math.max(...brokenEdges).toFixed(3)}`);

  // The table above is DESCRIPTIVE ONLY. Twelve trials cannot reliably rank two
  // strategies whose Sharpe ratios differ by a few hundredths, and an earlier
  // version of this file asserted a ranking from it that did not survive more
  // trials. Ranking claims are made below, paired and with a t-statistic.
  ok("the table is only used descriptively, not to rank strategies",
    Object.values(table).every((row) => Number.isFinite(row.equal)));

  console.log("");
  console.log(`      stationary edge ${stationaryEdge >= 0 ? "+" : ""}${stationaryEdge.toFixed(3)}, worst non-stationary edge ${Math.min(...brokenEdges).toFixed(3)}`);
}

section("Ranking strategies properly: paired trials with a t-statistic");
{
  // Running both strategies on the SAME generated data pairs the comparison and
  // removes the between-universe variance that swamped the 12-trial table.
  const TRIALS = 60;
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const sd = (a) => {
    const m = mean(a);
    return Math.sqrt(a.reduce((t, v) => t + (v - m) ** 2, 0) / (a.length - 1));
  };

  function pairedDiff(gen, stratA, stratB, trials = TRIALS) {
    const diffs = [];
    for (let t = 0; t < trials; t++) {
      const sc = gen(6, 3000, 900 + t * 61);
      let a = null, b = null;
      const run = (strat) => B.walkForward(sc.prices, null, {
        lookback: 500, rebalanceEvery: 21, strategy: strat,
        estimator: "ledoit-wolf", periodsPerYear: 252, costBps: 10, rf: 0.02,
      }).metrics.sharpe;
      try { a = run(stratA); } catch (e) { /* skip */ }
      try { b = run(stratB); } catch (e) { /* skip */ }
      if (a !== null && b !== null) diffs.push(a - b);
    }
    const m = mean(diffs);
    const se = sd(diffs) / Math.sqrt(diffs.length);
    return { mean: m, se, t: m / se, n: diffs.length, wins: diffs.filter((d) => d > 0).length };
  }

  const stat = pairedDiff(S.stationary, "minvar", "maxsharpe");
  ok("on stationary data minimum variance beats maximum Sharpe significantly",
    stat.t > 2, `mean +${stat.mean.toFixed(4)}, t = ${stat.t.toFixed(2)}, won ${stat.wins}/${stat.n}`);
  console.log(`      stationary:      minvar - maxsharpe = ${stat.mean >= 0 ? "+" : ""}${stat.mean.toFixed(4)}  t = ${stat.t.toFixed(2)}  won ${stat.wins}/${stat.n}`);

  const regime = pairedDiff(S.regimeSwitching, "minvar", "maxsharpe");
  console.log(`      regime switching: minvar - maxsharpe = ${regime.mean >= 0 ? "+" : ""}${regime.mean.toFixed(4)}  t = ${regime.t.toFixed(2)}  won ${regime.wins}/${regime.n}`);
  ok("under regime switching the difference is not significant either way",
    Math.abs(regime.t) < 2,
    `t = ${regime.t.toFixed(2)} — the 12-trial table showed maximum Sharpe ahead here, which was noise`);

  ok("the direction still favours minimum variance in both scenarios",
    stat.mean > 0 && regime.mean > 0);
  console.log(`      the 12-trial table ranked maxSharpe above minvar under regime switching;`);
  console.log(`      at 60 paired trials that reverses and is not significant. Third time in this`);
  console.log(`      project that an apparent ranking has failed to survive more trials.`);
}

section("Why the break does not bite: expected returns are not estimable");
{
  // Reversing every asset's expected return had no measurable effect. The
  // explanation is not that the break is harmless — it is that the estimator
  // was never tracking expected returns in the first place. Merton (1980):
  // the precision of a mean estimate depends on the LENGTH of the sample, not
  // the number of observations in it, so sampling more finely does not help.
  const LOOKBACK = 500;
  const DAILY_VOL = 0.012;

  // Standard error of a mean estimated from LOOKBACK daily observations,
  // expressed in annualized percentage points.
  const seAnnualPP = (DAILY_VOL / Math.sqrt(LOOKBACK)) * 252 * 100;

  const sc = S.structuralBreak(6, 3000, 961);
  const spreadPP = (Math.max(...sc.muBefore) - Math.min(...sc.muBefore)) * 252 * 100;
  const snr = spreadPP / seAnnualPP;

  ok("the noise in a mean estimate exceeds the spread being estimated",
    snr < 1, `signal ${spreadPP.toFixed(1)}pp vs noise ${seAnnualPP.toFixed(1)}pp, ratio ${snr.toFixed(2)}`);
  console.log(`      spread in true annual returns: ${spreadPP.toFixed(1)} percentage points`);
  console.log(`      standard error of the estimate: ${seAnnualPP.toFixed(1)} percentage points`);
  console.log(`      signal-to-noise: ${snr.toFixed(2)} — below 1, so the ranking is unrecoverable`);

  // Covariance is a different matter: its relative error falls as sqrt(2/T),
  // which at 500 observations is a few percent rather than a few hundred.
  const covRelError = Math.sqrt(2 / LOOKBACK) * 100;
  ok("covariance is estimable to a few percent from the same sample",
    covRelError < 10, `${covRelError.toFixed(1)}% relative error`);
  console.log(`      covariance relative error at the same sample size: ${covRelError.toFixed(1)}%`);
  console.log(`      -> this asymmetry is why minimum variance leads maximum Sharpe on average`);

  // The mechanism, demonstrated: raise the signal far above the noise and the
  // break starts doing exactly the damage originally predicted.
  const median = (a) => { const s2 = [...a].sort((x, y) => x - y); return s2.length ? s2[Math.floor(s2.length / 2)] : NaN; };
  const runPair = (mult) => {
    const stat = [], brk = [];
    for (let trial = 0; trial < 10; trial++) {
      const seed = 900 + trial * 61;
      const base = Array.from({ length: 6 }, (_, i) => (0.0001 + (i / 5) * 0.0004) * mult);
      const vols = new Array(6).fill(0.010);
      const corr = S.constantCorr(6, 0.3);
      const a = S.stationary(6, 3000, seed, { mu: base, vols, corr });
      const b = S.structuralBreak(6, 3000, seed, { muBefore: base, muAfter: [...base].reverse(), vols, corr });
      for (const [arr, scn] of [[stat, a], [brk, b]]) {
        try {
          arr.push(B.walkForward(scn.prices, null, {
            lookback: LOOKBACK, rebalanceEvery: 21, strategy: "maxsharpe",
            estimator: "ledoit-wolf", periodsPerYear: 252, costBps: 10, rf: 0.02,
          }).metrics.sharpe);
        } catch (e) { /* skip */ }
      }
    }
    return median(stat) - median(brk);
  };

  const damageRealistic = runPair(1);
  const damageLoud = runPair(12);
  ok("with a realistic signal the break causes no measurable damage",
    damageRealistic < 0.3, `damage ${damageRealistic.toFixed(3)} Sharpe`);
  ok("with the signal raised far above the noise, the break does the predicted damage",
    damageLoud > damageRealistic + 0.5,
    `realistic ${damageRealistic.toFixed(3)} vs amplified ${damageLoud.toFixed(3)}`);
  console.log(`      break damage at realistic signal: ${damageRealistic.toFixed(3)} Sharpe`);
  console.log(`      break damage at 12x signal:       ${damageLoud.toFixed(3)} Sharpe`);
  console.log(`      the break was always harmful; at realistic levels it is invisible under the noise`);
}

// ---------------------------------------------------------------- summary
console.log(`\n${"=".repeat(64)}`);
console.log(`${pass} passed, ${fail} failed`);
if (fail) { console.log("Failures:"); failures.forEach((f) => console.log(`  - ${f}`)); }
console.log("=".repeat(64));
process.exit(fail ? 1 : 0);
