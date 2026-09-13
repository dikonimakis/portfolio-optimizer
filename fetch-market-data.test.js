const F = require("./fetch-market-data.js");
const D = require("./data-pipeline.js");

let pass = 0, fail = 0;
const failures = [];
function ok(name, cond, detail = "") {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; failures.push(name); console.log(`  FAIL  ${name}  ${detail}`); }
}
function throwsWith(fn, substring) {
  try { fn(); return false; }
  catch (e) { return e.message.toLowerCase().includes(substring.toLowerCase()); }
}
const section = (s) => console.log(`\n== ${s}`);

// ---------------------------------------------------------------- Stooq
section("Stooq response parsing");
{
  const body = `Date,Open,High,Low,Close,Volume
2024-01-02,470.10,472.50,469.00,471.20,75000000
2024-01-03,471.00,473.10,468.20,469.80,68000000
2024-01-04,469.50,474.00,469.10,473.55,71000000`;
  const m = F.parseStooqCSV(body, "SPY");
  ok("reads every row", m.size === 3);
  ok("takes the close column, not open or high", m.get("2024-01-02") === 471.20);
  ok("keys are the dates", m.has("2024-01-04") && m.get("2024-01-04") === 473.55);
}
{
  // Column order is not guaranteed across providers or over time.
  const body = `Date,Close,Open,Volume\n2024-01-02,100.5,99.0,1000\n2024-01-03,101.5,100.0,1200`;
  const m = F.parseStooqCSV(body, "X");
  ok("locates columns by header name rather than position", m.get("2024-01-02") === 100.5);
}
{
  ok("an empty body is rejected clearly",
    throwsWith(() => F.parseStooqCSV("", "SPY"), "no data"));
  ok("a 'No data' body is rejected clearly",
    throwsWith(() => F.parseStooqCSV("No data found", "BADSYM"), "no data"));
  ok("a body without a Close column is rejected",
    throwsWith(() => F.parseStooqCSV("Date,Open,Volume\n2024-01-02,1,2", "X"), "unexpected"));
  ok("the failing ticker is named in the error",
    throwsWith(() => F.parseStooqCSV("", "WEIRDTICKER"), "WEIRDTICKER"));
  // Discovered via live testing (2026-09): Stooq's endpoint now returns this
  // HTML challenge page, with HTTP 200, instead of CSV for a plain HTTP client.
  const antiBotPage = '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>' +
    '<noscript>This site requires JavaScript to verify your browser.</noscript>' +
    '<script>/* proof-of-work challenge */</script></body></html>';
  ok("an anti-bot JavaScript challenge page is recognized, not dumped as a format error",
    throwsWith(() => F.parseStooqCSV(antiBotPage, "SPY"), "anti-bot"));
  ok("a plain unexpected-format body is still reported as such, not misdiagnosed as anti-bot",
    throwsWith(() => F.parseStooqCSV("garbage,nonsense\n1,2", "SPY"), "unexpected"));
}
{
  const body = `Date,Close\n2024-01-02,100\n2024-01-03,\n2024-01-04,N/A\n2024-01-05,0\n2024-01-06,-5\n2024-01-07,105`;
  const m = F.parseStooqCSV(body, "X");
  ok("blank, non-numeric, zero and negative prices are all dropped", m.size === 2,
    `kept ${[...m.keys()].join(",")}`);
  ok("valid rows around the bad ones survive", m.get("2024-01-02") === 100 && m.get("2024-01-07") === 105);
  ok("a body with no usable rows is rejected",
    throwsWith(() => F.parseStooqCSV("Date,Close\n2024-01-02,\n2024-01-03,N/A", "X"), "no usable"));
}

// ---------------------------------------------------------------- Alpha Vantage
section("Alpha Vantage response parsing");
{
  const body = JSON.stringify({
    "Meta Data": { "2. Symbol": "SPY" },
    "Time Series (Daily)": {
      "2024-01-04": { "1. open": "469.50", "4. close": "473.55", "5. adjusted close": "472.10" },
      "2024-01-03": { "1. open": "471.00", "4. close": "469.80", "5. adjusted close": "468.40" },
    },
  });
  const raw = F.parseAlphaVantageJSON(body, false, "SPY");
  ok("reads the unadjusted close", raw.get("2024-01-04") === 473.55);
  const adj = F.parseAlphaVantageJSON(body, true, "SPY");
  ok("reads the adjusted close when asked", adj.get("2024-01-04") === 472.10);
  ok("adjusted and unadjusted genuinely differ", raw.get("2024-01-03") !== adj.get("2024-01-03"));
  ok("accepts an already-parsed object as well as a string",
    F.parseAlphaVantageJSON(JSON.parse(body), false, "SPY").size === 2);
}
{
  // Alpha Vantage signals all of these with HTTP 200, so the body must be read.
  ok("an API error message is surfaced",
    throwsWith(() => F.parseAlphaVantageJSON(
      JSON.stringify({ "Error Message": "Invalid API call." }), false, "X"), "Invalid API call"));
  ok("a rate-limit note is surfaced as a rate limit, not a parse failure",
    throwsWith(() => F.parseAlphaVantageJSON(
      JSON.stringify({ "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute" }),
      false, "X"), "rate limit"));
  ok("an 'Information' response (invalid key) is surfaced",
    throwsWith(() => F.parseAlphaVantageJSON(
      JSON.stringify({ "Information": "the parameter apikey is invalid" }), false, "X"), "apikey"));
  ok("non-JSON is rejected clearly",
    throwsWith(() => F.parseAlphaVantageJSON("<html>502 Bad Gateway</html>", false, "X"), "non-json"));
  ok("a response with no time series is rejected",
    throwsWith(() => F.parseAlphaVantageJSON(JSON.stringify({ "Meta Data": {} }), false, "X"), "no time series"));
}
{
  // The free tier commonly omits adjusted closes; the fallback must not be silent
  // about producing something different from what was requested.
  const noAdj = JSON.stringify({
    "Time Series (Daily)": { "2024-01-04": { "4. close": "473.55" } },
  });
  const m = F.parseAlphaVantageJSON(noAdj, true, "SPY");
  ok("falls back to unadjusted close when adjusted is absent", m.get("2024-01-04") === 473.55);

  const empty = JSON.stringify({ "Time Series (Daily)": { "2024-01-04": { "4. close": "0" } } });
  ok("an adjusted request yielding nothing usable explains the free-tier cause",
    throwsWith(() => F.parseAlphaVantageJSON(empty, true, "SPY"), "free tier"));
}

// ---------------------------------------------------------------- Tiingo
section("Tiingo response parsing");
{
  // Shape recorded from a live call in 2026-09 (trimmed to the fields used).
  const body = JSON.stringify([
    { date: "2024-01-03T00:00:00.000Z", close: 469.80, adjClose: 468.40 },
    { date: "2024-01-04T00:00:00.000Z", close: 473.55, adjClose: 472.10 },
  ]);
  const raw = F.parseTiingoJSON(body, 200, false, "SPY");
  ok("reads the unadjusted close", raw.get("2024-01-04") === 473.55);
  ok("date is truncated to YYYY-MM-DD, dropping the time component", raw.has("2024-01-04") && !raw.has("2024-01-04T00:00:00.000Z"));
  const adj = F.parseTiingoJSON(body, 200, true, "SPY");
  ok("reads the adjusted close when asked", adj.get("2024-01-04") === 472.10);
  ok("adjusted and unadjusted genuinely differ", raw.get("2024-01-03") !== adj.get("2024-01-03"));
  ok("accepts an already-parsed object as well as a string",
    F.parseTiingoJSON(JSON.parse(body), 200, false, "SPY").size === 2);
}
{
  // Tiingo signals errors with a real HTTP status and a {"detail": "..."}
  // body — unlike Stooq/Alpha Vantage, which always return 200.
  ok("an unknown ticker (404) is surfaced with Tiingo's own explanation",
    throwsWith(() => F.parseTiingoJSON(
      JSON.stringify({ detail: "Error: Ticker 'BADSYM' not found" }), 404, false, "BADSYM"), "not found"));
  ok("an invalid token (403) is surfaced",
    throwsWith(() => F.parseTiingoJSON(
      JSON.stringify({ detail: "Invalid token." }), 403, false, "SPY"), "invalid token"));
  ok("a missing token (403) is surfaced",
    throwsWith(() => F.parseTiingoJSON(
      JSON.stringify({ detail: "Please supply a token" }), 403, false, "SPY"), "supply a token"));
  ok("a non-200 status with no usable detail still fails clearly, not silently",
    throwsWith(() => F.parseTiingoJSON("{}", 500, false, "SPY"), "http 500"));
  ok("non-JSON is rejected clearly",
    throwsWith(() => F.parseTiingoJSON("<html>502 Bad Gateway</html>", 502, false, "SPY"), "non-json"));
  ok("a 200 response that isn't an array is rejected",
    throwsWith(() => F.parseTiingoJSON(JSON.stringify({ not: "an array" }), 200, false, "SPY"), "not an array"));
  ok("a 200 response with an empty array is rejected",
    throwsWith(() => F.parseTiingoJSON("[]", 200, false, "SPY"), "no data"));
}
{
  const body = JSON.stringify([
    { date: "2024-01-02T00:00:00.000Z", close: 100, adjClose: 100 },
    { date: "2024-01-03T00:00:00.000Z", close: null, adjClose: null },
    { date: "2024-01-04T00:00:00.000Z", close: 0, adjClose: 0 },
    { date: "2024-01-05T00:00:00.000Z", close: -5, adjClose: -5 },
    { date: "2024-01-08T00:00:00.000Z", close: 105, adjClose: 106 },
  ]);
  const m = F.parseTiingoJSON(body, 200, false, "X");
  ok("null, zero and negative prices are all dropped", m.size === 2, `kept ${[...m.keys()].join(",")}`);
  ok("valid rows around the bad ones survive", m.get("2024-01-02") === 100 && m.get("2024-01-08") === 105);
}

// ---------------------------------------------------------------- Tiingo crypto
section("Tiingo crypto response parsing");
{
  // Shape recorded from a live call in 2026-09: an array with one entry PER
  // TICKER (even for a single ticker), each holding a nested priceData array
  // — a different shape from the equities endpoint's flat array.
  const body = JSON.stringify([
    {
      ticker: "btcusd", baseCurrency: "btc", quoteCurrency: "usd",
      priceData: [
        { date: "2024-01-02T00:00:00+00:00", close: 44974.05 },
        { date: "2024-01-03T00:00:00+00:00", close: 42862.35 },
      ],
    },
  ]);
  const m = F.parseTiingoCryptoJSON(body, 200, "btcusd");
  ok("reads the close price", m.get("2024-01-02") === 44974.05);
  ok("date is truncated to YYYY-MM-DD", m.has("2024-01-03") && !m.has("2024-01-03T00:00:00+00:00"));
  ok("accepts an already-parsed object as well as a string",
    F.parseTiingoCryptoJSON(JSON.parse(body), 200, "btcusd").size === 2);
}
{
  // Measured live (2026-09): an unknown ticker is HTTP 200 with an EMPTY
  // array — unlike the equities endpoint's 404 + detail for the same case.
  ok("an unknown ticker (empty array) is surfaced, not silently treated as zero rows",
    throwsWith(() => F.parseTiingoCryptoJSON("[]", 200, "notarealcoin"), "check the format"));
  ok("an entry with no priceData is rejected",
    throwsWith(() => F.parseTiingoCryptoJSON(JSON.stringify([{ ticker: "btcusd" }]), 200, "btcusd"), "no price data"));
  ok("an entry with an empty priceData array is rejected",
    throwsWith(() => F.parseTiingoCryptoJSON(JSON.stringify([{ ticker: "btcusd", priceData: [] }]), 200, "btcusd"), "no price data"));
  ok("a real HTTP error status surfaces Tiingo's detail message",
    throwsWith(() => F.parseTiingoCryptoJSON(JSON.stringify({ detail: "Please supply a token" }), 403, "btcusd"), "supply a token"));
  ok("non-JSON is rejected clearly",
    throwsWith(() => F.parseTiingoCryptoJSON("<html>502 Bad Gateway</html>", 502, "btcusd"), "non-json"));
  ok("a 200 response that isn't an array is rejected",
    throwsWith(() => F.parseTiingoCryptoJSON(JSON.stringify({ not: "an array" }), 200, "btcusd"), "not an array"));
}
{
  const body = JSON.stringify([{
    ticker: "btcusd",
    priceData: [
      { date: "2024-01-02T00:00:00+00:00", close: 100 },
      { date: "2024-01-03T00:00:00+00:00", close: null },
      { date: "2024-01-04T00:00:00+00:00", close: 0 },
      { date: "2024-01-05T00:00:00+00:00", close: -5 },
      { date: "2024-01-08T00:00:00+00:00", close: 105 },
    ],
  }]);
  const m = F.parseTiingoCryptoJSON(body, 200, "btcusd");
  ok("null, zero and negative prices are all dropped", m.size === 2, `kept ${[...m.keys()].join(",")}`);
  ok("valid rows around the bad ones survive", m.get("2024-01-02") === 100 && m.get("2024-01-08") === 105);
}

// ---------------------------------------------------------------- alignment
section("Date alignment across tickers");
{
  const a = new Map([["2024-01-02", 100], ["2024-01-03", 101], ["2024-01-04", 102]]);
  const b = new Map([["2024-01-02", 50], ["2024-01-04", 52], ["2024-01-05", 53]]);
  const dates = F.alignSeries([{ ticker: "A", series: a }, { ticker: "B", series: b }]);
  ok("keeps only dates present for every ticker", dates.join(",") === "2024-01-02,2024-01-04",
    dates.join(","));
  ok("a date missing from one ticker is dropped for all", !dates.includes("2024-01-03"));
  ok("output is chronologically sorted",
    dates.every((d, i) => i === 0 || d >= dates[i - 1]));
}
{
  const a = new Map([["2024-01-02", 1], ["2024-02-02", 2], ["2024-03-02", 3]]);
  const b = new Map([["2024-01-02", 1], ["2024-02-02", 2], ["2024-03-02", 3]]);
  const dates = F.alignSeries([{ ticker: "A", series: a }, { ticker: "B", series: b }], "2024-02-01");
  ok("the start date filter is applied", dates.length === 2 && dates[0] === "2024-02-02");
}
{
  const a = new Map([["2024-01-02", 1]]);
  const b = new Map([["2024-06-02", 1]]);
  ok("no overlapping history yields an empty result rather than an exception",
    F.alignSeries([{ ticker: "A", series: a }, { ticker: "B", series: b }]).length === 0);
  ok("aligning nothing is rejected",
    throwsWith(() => F.alignSeries([]), "no series"));
}

// ---------------------------------------------------------------- CSV output
section("CSV output feeds the pipeline");
{
  const a = new Map([["2024-01-02", 100], ["2024-01-03", 101]]);
  const b = new Map([["2024-01-02", 50], ["2024-01-03", 50.5]]);
  const series = [{ ticker: "SPY", series: a }, { ticker: "AGG", series: b }];
  const csv = F.buildCSV(series, F.alignSeries(series));
  ok("header names the tickers", csv.split("\n")[0] === "Date,SPY,AGG");
  ok("rows are date-then-prices", csv.split("\n")[1] === "2024-01-02,100,50");

  // The contract that matters: output of this script must be input to the pipeline.
  const parsed = D.parsePriceCSV(csv);
  ok("the pipeline parses this script's output", parsed.names.join(",") === "SPY,AGG");
  ok("prices survive the round trip", parsed.prices[1][0] === 101 && parsed.prices[1][1] === 50.5);
}
{
  // Full round trip on enough data to actually estimate from. Weekday-only
  // dates (skipping Sat/Sun), matching a real stock market calendar — a
  // fully-consecutive (weekend-inclusive) series would be misread as a
  // 24/7 market by inferFrequency's 365-day detection, same as a stock
  // ticker's data never should be.
  const n = 300;
  const spy = new Map(), agg = new Map();
  let p1 = 400, p2 = 100;
  let i = 0;
  for (let d = new Date(Date.UTC(2023, 0, 1)); spy.size < n; d.setUTCDate(d.getUTCDate() + 1)) {
    const day = d.getUTCDay();
    if (day === 0 || day === 6) continue;
    const ds = d.toISOString().slice(0, 10);
    p1 *= 1 + (Math.sin(i * 1.7) * 0.01);
    p2 *= 1 + (Math.cos(i * 2.3) * 0.003);
    spy.set(ds, Math.round(p1 * 100) / 100);
    agg.set(ds, Math.round(p2 * 100) / 100);
    i++;
  }
  const series = [{ ticker: "SPY", series: spy }, { ticker: "AGG", series: agg }];
  const csv = F.buildCSV(series, F.alignSeries(series));
  const est = D.estimateFromCSV(csv);
  ok("a full download-to-estimate round trip works", est.names.join(",") === "SPY,AGG");
  ok("observation count matches the download", est.observations === n - 1);
  ok("frequency is inferred from the generated dates", est.frequency.periodsPerYear === 252);
  ok("estimates are finite and usable",
    est.mu.every(Number.isFinite) && est.vols.every((v) => v > 0));
}

// ---------------------------------------------------------------- CLI arguments
section("Command line parsing");
{
  const a = F.parseArgs(["node", "s.js", "--source=stooq", "--tickers=SPY,AGG", "--adjusted"]);
  ok("key=value arguments are parsed", a.source === "stooq" && a.tickers === "SPY,AGG");
  ok("bare flags become true", a.adjusted === true);
  ok("absent arguments are undefined", a.key === undefined);

  const b = F.parseArgs(["node", "s.js", "--start=2020-01-01"]);
  ok("dates containing dashes are not mangled", b.start === "2020-01-01");
}

// ---------------------------------------------------------------- summary
console.log(`\n${"=".repeat(64)}`);
console.log(`${pass} passed, ${fail} failed`);
if (fail) { console.log("Failures:"); failures.forEach((f) => console.log(`  - ${f}`)); }
console.log("=".repeat(64));
console.log("\nNot covered: the HTTPS request itself, which needs network access to");
console.log("the providers. Everything that interprets their responses is tested above.");
process.exit(fail ? 1 : 0);
