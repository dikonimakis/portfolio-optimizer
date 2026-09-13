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

console.log(`Loaded ${prices.length} daily observations for ${names.join(", ")}`);
console.log(`${dates[0]} to ${dates[dates.length - 1]} (~${(prices.length / 252).toFixed(1)} years)\n`);

for (const lookback of [250, 500, 1000]) {
  const results = B.compareStrategies(prices, dates, [
    { name: "Equal weight (1/N)", strategy: "equal" },
    { name: "Min variance, shrunk", strategy: "minvar", estimator: "ledoit-wolf" },
    { name: "Max Sharpe, shrunk", strategy: "maxsharpe", estimator: "ledoit-wolf" },
    { name: "Max Sharpe, capped 30%", strategy: "maxsharpe", estimator: "ledoit-wolf", bounds: { maxWeight: 0.30 } },
  ], { lookback, rebalanceEvery: 21, periodsPerYear: 252, costBps: 10, rf: 0.02 });

  console.log(`\n=== lookback=${lookback} days (~${(lookback/252).toFixed(1)}y estimation window) ===`);
  for (const r of results) {
    if (!r.metrics) { console.log(`${r.name}: ERROR ${r.error}`); continue; }
  }
  console.log(B.formatComparison(results));
  console.log(`(${results[0].metrics.periods} out-of-sample periods, ~${(results[0].metrics.periods / 252).toFixed(1)} years)`);
}
