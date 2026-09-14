// Ad hoc script: runs the walk-forward backtest comparison on real historical
// price data (fetched via fetch-market-data.js --source=tiingo), rather than
// the generated data backtest.test.js and scenarios.test.js use. Not part of
// the test suite — this is a one-off validation exercise, and its result is
// a single draw from actual history, not a repeatable statistical claim.
// Runs at several lookback windows to check whether any finding is robust to
// that choice or an artifact of one parameter pick (the closest available
// analog, on real data, to the project's "check a different seed range"
// discipline for synthetic data).
const fs = require("fs");
const D = require("./data-pipeline.js");
const B = require("./backtest.js");

const csv = fs.readFileSync(process.argv[2] || "prices_real_full.csv", "utf8");
const { dates, names, prices } = D.parsePriceCSV(csv);

// Auto-detect periods/year (252 weekday market vs. 365 24/7 calendar, e.g.
// crypto) rather than assuming 252 -- the same class of bug fixed in
// data-pipeline.js's inferFrequency applies here too: walkForward takes
// periodsPerYear as an explicit option and does not infer it itself, so a
// hardcoded 252 would silently misstate every annualized figure for crypto.
const freq = D.inferFrequency(dates);
const periodsPerYear = freq.periodsPerYear;

console.log(`Loaded ${prices.length} daily observations for ${names.join(", ")}`);
console.log(`${dates[0]} to ${dates[dates.length - 1]} (~${(prices.length / periodsPerYear).toFixed(1)} years)`);
console.log(`Detected calendar: ${freq.frequency} (${periodsPerYear} periods/year)\n`);

// A fixed 30% cap is infeasible outright for n <= 3 assets (n * cap must
// exceed 1 to reach full investment) and only weakly meaningful just above
// that. Scale the cap to the universe size instead of hardcoding 30%, which
// silently produced a 100%-estimation-failure, equal-weight-in-disguise
// "capped" row for a 2-asset crypto test -- caught only by checking
// estimationFailures directly, not from the output table, which gave no
// visible sign anything was wrong.
const n = names.length;
const cap = Math.max(0.30, Math.ceil((1 / n + 0.1) * 20) / 20);

for (const lookback of [250, 500, 1000]) {
  const results = B.compareStrategies(prices, dates, [
    { name: "Equal weight (1/N)", strategy: "equal" },
    { name: "Min variance, shrunk", strategy: "minvar", estimator: "ledoit-wolf" },
    { name: "Max Sharpe, shrunk", strategy: "maxsharpe", estimator: "ledoit-wolf" },
    { name: `Max Sharpe, capped ${(cap * 100).toFixed(0)}%`, strategy: "maxsharpe", estimator: "ledoit-wolf", bounds: { maxWeight: cap } },
  ], { lookback, rebalanceEvery: 21, periodsPerYear, costBps: 10, rf: 0.02 });

  console.log(`\n=== lookback=${lookback} periods (~${(lookback/periodsPerYear).toFixed(1)}y estimation window) ===`);
  for (const r of results) {
    if (!r.metrics) { console.log(`${r.name}: ERROR ${r.error}`); continue; }
    if (r.rebalanceCount > 0 && r.estimationFailures === r.rebalanceCount) {
      console.log(`  WARNING: "${r.name}" failed to solve at EVERY rebalance (${r.estimationFailures}/${r.rebalanceCount}) and silently fell back to prior/equal weights throughout -- its row below is not a real result.`);
    } else if (r.estimationFailures > 0) {
      console.log(`  note: "${r.name}" fell back to prior/equal weights on ${r.estimationFailures}/${r.rebalanceCount} rebalances.`);
    }
  }
  console.log(B.formatComparison(results));
  console.log(`(${results[0].metrics.periods} out-of-sample periods, ~${(results[0].metrics.periods / periodsPerYear).toFixed(1)} years)`);
}
