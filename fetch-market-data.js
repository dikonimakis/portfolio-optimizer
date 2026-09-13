#!/usr/bin/env node
// ============================================================================
// fetch-market-data.js
//
// Downloads historical prices and writes a CSV in the format the optimizer's
// data pipeline expects:
//
//     Date,SPY,AGG,GLD
//     2020-01-02,321.55,112.30,143.10
//     ...
//
// Run this on your own machine with your own API key. Keys must never be
// placed in browser code — anyone can read them there, and most data licences
// forbid redistribution, which client-side fetching would amount to.
//
// USAGE
//   node fetch-market-data.js --source=stooq --tickers=SPY,AGG,GLD --out=prices.csv
//   node fetch-market-data.js --source=alphavantage --tickers=SPY,AGG --key=YOUR_KEY
//   node fetch-market-data.js --source=tiingo --tickers=SPY,AGG --key=YOUR_TOKEN
//   node fetch-market-data.js --source=tiingocrypto --tickers=btcusd,ethusd --key=YOUR_TOKEN
//
// A NOTE ON MIXING CRYPTO WITH STOCKS/BONDS
//   Crypto trades every calendar day; stocks/bonds trade ~252 weekdays a
//   year. This script only ever fetches one asset class per run, so it
//   cannot silently mix the two calendars itself — but if you fetch each
//   separately and then combine the two CSVs yourself before feeding them
//   to data-pipeline.js, its date-intersection logic will keep only the
//   weekday dates common to both, discarding crypto's weekend price moves
//   from the volatility/correlation estimate. That is a real distortion,
//   not a bug this script can fix for you: decide deliberately whether you
//   want that behavior (comparable, weekday-only estimates) or a different
//   convention (e.g. resampling crypto onto its own calendar).
//
// SOURCES
//   stooq         No key required in principle, but as of live testing in
//                 2026-09, Stooq's public CSV endpoint returns an HTTP 200
//                 JavaScript proof-of-work anti-bot challenge page instead of
//                 data for any plain HTTPS client, regardless of User-Agent.
//                 That is a server-side access control, not a parsing gap,
//                 and this project does not attempt to solve it. Left in
//                 place (and still unit-tested against recorded CSV) in case
//                 Stooq's policy changes again, but alphavantage/tiingo are
//                 the sources that actually work today.
//   alphavantage  Free key from alphavantage.co. As of live testing in
//                 2026-09, the free tier no longer allows outputsize=full on
//                 TIME_SERIES_DAILY (premium-only) and TIME_SERIES_DAILY_ADJUSTED
//                 is premium-only outright regardless of outputsize. This
//                 script defaults to outputsize=compact (~100 most recent
//                 observations) and --adjusted will fail with a clear error
//                 on a free key. The free tier is also capped at 25
//                 requests/day, not just a per-minute rate, so budget calls
//                 accordingly.
//   tiingo        Free token from tiingo.com/account/api/token. As of live
//                 testing in 2026-09, the free tier returned full multi-year
//                 daily history (8,462 rows for SPY back to 1993, its
//                 inception) with no outputsize-style restriction observed.
//                 Both raw and split/dividend-adjusted closes are included in
//                 every response (adjClose field), so --adjusted works
//                 without the premium wall Alpha Vantage now has. Errors come
//                 back as real HTTP status codes (404 unknown ticker, 403
//                 bad/missing token) with a JSON {"detail": "..."} body,
//                 unlike the other two providers which signal errors via
//                 HTTP 200. MEASURED GOTCHA: omitting startDate does not
//                 default to full history — it returns exactly one row, the
//                 latest quote. This script always sends a startDate
//                 (defaulting to 1900-01-01) so --source=tiingo with no
//                 --start still gets full history rather than silently one row.
//   tiingocrypto  Same free token as tiingo, different endpoint
//                 (tiingo/crypto/prices), different response shape, and
//                 different quirks — confirmed via live testing in 2026-09:
//                 - Tickers are exchange pairs, lowercase, e.g. "btcusd",
//                   "ethusd" — not bare symbols.
//                 - Response is an array with one entry per ticker, each
//                   holding a nested `priceData` array — not the flat array
//                   the equities endpoint returns.
//                 - An unknown ticker comes back as HTTP 200 with an EMPTY
//                   array, not a 404 + detail like tiingo's equities endpoint.
//                 - There is no adjusted-close field (no splits/dividends
//                   for crypto), so --adjusted has no effect on this source.
//                 - Omitting startDate does not return full history either
//                   — it returns only a partial day of recent intraday bars.
//                   Omitting resampleFreq while passing an old startDate can
//                   silently return an empty array. This script always sends
//                   both explicitly (resampleFreq default "1day").
//                 - THE BIG ONE: the free tier silently caps a single request
//                   at roughly 5,000 daily rows, truncating from the END
//                   (most recent data), not the start. Requesting BTC from
//                   its real 2011-08-19 inception returns data that stops
//                   in 2023 — over 3 years stale — while requesting from
//                   2015-01-01 reaches today. "Ask for the earliest possible
//                   date" is backwards here. This script defaults startDate
//                   to ~10 years back from today (not an arbitrarily early
//                   date) for exactly that reason; it does not paginate
//                   around the cap, so anyone wanting more than ~10-13 years
//                   of daily crypto history needs to fetch it in chunks
//                   themselves and verify each chunk's actual end date.
//                 - Daily history genuinely goes back to each asset's real
//                   start (BTC to 2011-08-19, ETH to 2015-08-08, confirmed
//                   live within the row cap) — and even that early daily
//                   history has real gaps (missing days), unlike equities'
//                   regular weekday calendar.
//                 - Free tier goes down to 1-minute bars (confirmed live),
//                   far finer than the daily-only equities sources — but
//                   this script's CSV format is one row per calendar date,
//                   so requesting sub-daily resampleFreq here will silently
//                   collapse to one (the last) bar per day. Fine-grained
//                   data is a different feature (a live price ticker) from
//                   what this script's output feeds (periodic estimation).
//
// TESTING NOTE
//   Everything that interprets a provider's response is a pure function
//   (parseStooqCSV, parseAlphaVantageJSON, parseTiingoJSON,
//   parseTiingoCryptoJSON, alignSeries, buildCSV) and is covered by
//   fetch-market-data.test.js. Verified against all four providers' live
//   endpoints in 2026-09 (see findings above); provider policies can change
//   again, so re-verify if a source starts failing.
// ============================================================================

const fs = require("fs");
const https = require("https");

// ---------------------------------------------------------------- arguments

function parseArgs(argv) {
  const args = {};
  for (const a of argv.slice(2)) {
    const m = a.match(/^--([^=]+)=(.*)$/);
    if (m) args[m[1]] = m[2];
    else if (a.startsWith("--")) args[a.slice(2)] = true;
  }
  return args;
}

function usage(msg) {
  if (msg) console.error(`\nError: ${msg}`);
  console.error(`
Usage:
  node fetch-market-data.js --source=<stooq|alphavantage|tiingo|tiingocrypto> --tickers=A,B,C [options]

Options:
  --source=NAME       Data provider. Default: stooq
  --tickers=LIST      Comma-separated symbols. Required, at least two.
                      tiingocrypto tickers are lowercase exchange pairs, e.g.
                      btcusd,ethusd — not bare symbols.
  --out=FILE          Output path. Default: prices.csv
  --key=KEY           API key/token. Required for alphavantage, tiingo, and
                      tiingocrypto (same token as tiingo).
  --start=YYYY-MM-DD  Earliest date to keep. Optional. Also passed to Tiingo/
                      tiingocrypto as startDate, so it need not fetch history
                      it will discard.
  --adjusted          Use split/dividend-adjusted closes where available.
                      (Alpha Vantage: premium-only as of 2026-09 — will fail
                      with a clear error on a free key. Tiingo: works on the
                      free tier. tiingocrypto: no effect — crypto has no
                      splits/dividends to adjust for.)
  --full              alphavantage only: request outputsize=full (premium-only
                      as of 2026-09). Default is compact (~100 observations).
                      tiingo/tiingocrypto always fetch full history (from
                      --start, or a provider-appropriate default) regardless
                      of this flag.
  --resample=FREQ     tiingocrypto only: Tiingo's resampleFreq. Default:
                      1day. This script writes one CSV row per calendar date,
                      so a sub-daily value silently collapses to the last bar
                      of each day — fine granularity is a different feature
                      (a live ticker) from what this CSV feeds.
`);
  process.exit(1);
}

// ---------------------------------------------------------------- pure parsers
// Separated from the network so they can be tested against recorded responses.

// Stooq returns a plain CSV: Date,Open,High,Low,Close,Volume
function parseStooqCSV(body, ticker = "series") {
  const text = String(body);
  if (body == null || /^\s*$/.test(text) || /No data/i.test(text)) {
    throw new Error(`Stooq returned no data for "${ticker}". Check the symbol.`);
  }
  if (/<!DOCTYPE|<html/i.test(text) && /javascript/i.test(text)) {
    throw new Error(
      `Stooq returned an anti-bot JavaScript challenge page instead of CSV data for "${ticker}". ` +
      `This is a server-side block (seen in live testing as of 2026-09), not a parsing bug — ` +
      `Stooq no longer serves its public CSV endpoint to plain HTTP clients. Use --source=alphavantage instead.`
    );
  }
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const dateIdx = header.indexOf("date");
  const closeIdx = header.indexOf("close");
  if (dateIdx < 0 || closeIdx < 0) {
    throw new Error(`Unexpected Stooq format for "${ticker}": ${lines[0]}`);
  }
  const out = new Map();
  for (let i = 1; i < lines.length; i++) {
    const c = lines[i].split(",");
    const v = parseFloat(c[closeIdx]);
    if (c[dateIdx] && Number.isFinite(v) && v > 0) out.set(c[dateIdx].trim(), v);
  }
  if (out.size === 0) throw new Error(`No usable closes for "${ticker}".`);
  return out;
}

// Alpha Vantage returns JSON keyed by date, and signals problems in-band with
// an HTTP 200, so the body must be inspected rather than the status code.
function parseAlphaVantageJSON(body, adjusted, ticker = "series") {
  let json;
  try { json = typeof body === "string" ? JSON.parse(body) : body; }
  catch (e) { throw new Error(`Alpha Vantage returned non-JSON for "${ticker}".`); }

  if (json["Error Message"]) throw new Error(`Alpha Vantage: ${json["Error Message"]}`);
  if (json["Note"]) throw new Error(`Alpha Vantage rate limit hit. Wait a minute and retry.`);
  if (json["Information"]) throw new Error(`Alpha Vantage: ${json["Information"]}`);

  const seriesKey = Object.keys(json).find((k) => k.toLowerCase().includes("time series"));
  if (!seriesKey) throw new Error(`No time series in the Alpha Vantage response for "${ticker}".`);

  const series = json[seriesKey];
  const field = adjusted ? "5. adjusted close" : "4. close";
  const out = new Map();
  for (const [date, row] of Object.entries(series)) {
    const raw = row[field] != null ? row[field] : row["4. close"];
    const v = parseFloat(raw);
    if (Number.isFinite(v) && v > 0) out.set(date, v);
  }
  if (out.size === 0) {
    if (adjusted) {
      throw new Error(`No adjusted closes for "${ticker}". The free tier may not include them — retry without --adjusted.`);
    }
    throw new Error(`No usable closes for "${ticker}".`);
  }
  return out;
}

// Tiingo returns a JSON array of daily bars, or (unlike Stooq/Alpha Vantage)
// signals errors with a real non-200 HTTP status plus a {"detail": "..."}
// body, so the status code must be inspected alongside the body.
function parseTiingoJSON(body, status, adjusted, ticker = "series") {
  let json;
  try { json = typeof body === "string" ? JSON.parse(body) : body; }
  catch (e) { throw new Error(`Tiingo returned non-JSON for "${ticker}".`); }

  if (status !== 200) {
    const detail = json && typeof json === "object" && json.detail;
    throw new Error(`Tiingo: ${detail || `HTTP ${status}`} (ticker "${ticker}").`);
  }
  if (!Array.isArray(json)) throw new Error(`Unexpected Tiingo format for "${ticker}": not an array.`);
  if (json.length === 0) throw new Error(`Tiingo returned no data for "${ticker}". Check the symbol.`);

  const field = adjusted ? "adjClose" : "close";
  const out = new Map();
  for (const row of json) {
    if (!row || typeof row.date !== "string") continue;
    const date = row.date.slice(0, 10);
    const raw = row[field] != null ? row[field] : row.close;
    const v = parseFloat(raw);
    if (date && Number.isFinite(v) && v > 0) out.set(date, v);
  }
  if (out.size === 0) throw new Error(`No usable closes for "${ticker}".`);
  return out;
}

// Tiingo's crypto endpoint has a genuinely different shape from its equities
// endpoint: a JSON array with one entry PER TICKER (even for a single
// ticker), each holding a nested `priceData` array of bars, rather than a
// flat array of bars. It also signals "not found" as HTTP 200 with an EMPTY
// array, not a 404 + detail like the equities endpoint — both measured live.
// There is no adjusted-close concept for crypto (no splits/dividends).
function parseTiingoCryptoJSON(body, status, ticker = "series") {
  let json;
  try { json = typeof body === "string" ? JSON.parse(body) : body; }
  catch (e) { throw new Error(`Tiingo returned non-JSON for "${ticker}".`); }

  if (status !== 200) {
    const detail = json && typeof json === "object" && json.detail;
    throw new Error(`Tiingo: ${detail || `HTTP ${status}`} (ticker "${ticker}").`);
  }
  if (!Array.isArray(json)) throw new Error(`Unexpected Tiingo crypto format for "${ticker}": not an array.`);
  if (json.length === 0) {
    throw new Error(`Tiingo returned no data for "${ticker}". Crypto tickers are exchange pairs (e.g. "btcusd"), not bare symbols — check the format.`);
  }

  const priceData = Array.isArray(json[0].priceData) ? json[0].priceData : [];
  if (priceData.length === 0) throw new Error(`Tiingo returned no price data for "${ticker}".`);

  const out = new Map();
  for (const row of priceData) {
    if (!row || typeof row.date !== "string") continue;
    const date = row.date.slice(0, 10);
    const v = parseFloat(row.close);
    if (date && Number.isFinite(v) && v > 0) out.set(date, v);
  }
  if (out.size === 0) throw new Error(`No usable closes for "${ticker}".`);
  return out;
}

// Keeps only dates present for EVERY ticker. Forward-filling would be easier
// but understates volatility and correlation, flattering the portfolio.
function alignSeries(seriesByTicker, startDate) {
  if (seriesByTicker.length === 0) throw new Error("No series to align.");
  let common = new Set(seriesByTicker[0].series.keys());
  for (const { series } of seriesByTicker.slice(1)) {
    common = new Set([...common].filter((d) => series.has(d)));
  }
  let dates = [...common].sort();
  if (startDate) dates = dates.filter((d) => d >= startDate);
  return dates;
}

function buildCSV(seriesByTicker, dates) {
  let csv = `Date,${seriesByTicker.map((s) => s.ticker).join(",")}\n`;
  for (const d of dates) {
    csv += `${d},${seriesByTicker.map(({ series }) => series.get(d)).join(",")}\n`;
  }
  return csv;
}

// ---------------------------------------------------------------- network

// Resolves with { status, body } rather than rejecting on a non-200 status:
// Tiingo signals real errors (bad ticker, bad token) via HTTP status with an
// informative JSON body, and discarding that body (as this used to do) would
// throw away the one useful thing in the response. Stooq and Alpha Vantage
// both always return 200 even on error, so this is a no-op change for them.
function get(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { "User-Agent": "portfolio-optimizer/1.0" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        return resolve(get(res.headers.location));
      }
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (c) => (body += c));
      res.on("end", () => resolve({ status: res.statusCode, body }));
    });
    req.on("error", reject);
    req.setTimeout(30000, () => req.destroy(new Error("Request timed out after 30s")));
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Thin wrappers: all interpretation lives in the tested parsers above.
async function fetchStooq(ticker) {
  const sym = ticker.toLowerCase().includes(".") ? ticker.toLowerCase() : `${ticker.toLowerCase()}.us`;
  const { body } = await get(`https://stooq.com/q/d/l/?s=${encodeURIComponent(sym)}&i=d`);
  return parseStooqCSV(body, ticker);
}

async function fetchAlphaVantage(ticker, key, adjusted, full) {
  const fn = adjusted ? "TIME_SERIES_DAILY_ADJUSTED" : "TIME_SERIES_DAILY";
  // outputsize=full is premium-only as of 2026-09 (confirmed via live testing);
  // compact (~100 observations) is what the free tier actually serves.
  const outputsize = full ? "full" : "compact";
  const url = `https://www.alphavantage.co/query?function=${fn}&symbol=${encodeURIComponent(ticker)}` +
              `&outputsize=${outputsize}&apikey=${encodeURIComponent(key)}`;
  const { body } = await get(url);
  return parseAlphaVantageJSON(body, adjusted, ticker);
}

async function fetchTiingo(ticker, key, adjusted, startDate) {
  // MEASURED (2026-09): omitting startDate does NOT return full history —
  // it returns exactly one row, the latest quote. A startDate is required
  // to get a history at all, so this always sends one, defaulting to a date
  // early enough to predate any US-listed ETF/equity.
  const params = new URLSearchParams({ token: key, startDate: startDate || "1900-01-01" });
  const url = `https://api.tiingo.com/tiingo/daily/${encodeURIComponent(ticker.toLowerCase())}/prices?${params.toString()}`;
  const { status, body } = await get(url);
  return parseTiingoJSON(body, status, adjusted, ticker);
}

async function fetchTiingoCrypto(ticker, key, startDate, resampleFreq) {
  // MEASURED (2026-09): omitting startDate returns only a partial day of
  // recent intraday bars, not full history; omitting resampleFreq while
  // passing an old startDate can silently return an empty array. Both are
  // always sent explicitly for that reason.
  //
  // A THIRD, more serious gotcha: the free tier silently caps a single
  // request at roughly 5,000 daily rows, and truncates from the END (most
  // recent data), not the start. Requesting BTC from its real 2011-08-19
  // inception returned data that stopped at 2023-09-10 — three-year-stale
  // — while requesting from 2015-01-01 reached today. So "ask for the
  // earliest possible date to get full history" is exactly backwards here:
  // it silently loses the recent past instead. Defaulting to ~10 years back
  // (comfortably under the cap for one asset) rather than an arbitrarily
  // early date, so --source=tiingocrypto with no --start reaches today.
  // Anyone wanting deeper history than that should pass --start explicitly
  // and verify the resulting end date themselves — this script does not
  // paginate around the cap.
  let defaultStart = startDate;
  if (!defaultStart) {
    const d = new Date();
    d.setUTCFullYear(d.getUTCFullYear() - 10);
    defaultStart = d.toISOString().slice(0, 10);
  }
  const params = new URLSearchParams({
    tickers: ticker.toLowerCase(),
    token: key,
    startDate: defaultStart,
    resampleFreq: resampleFreq || "1day",
  });
  const url = `https://api.tiingo.com/tiingo/crypto/prices?${params.toString()}`;
  const { status, body } = await get(url);
  return parseTiingoCryptoJSON(body, status, ticker);
}

// ---------------------------------------------------------------- main

async function main() {
  const args = parseArgs(process.argv);
  const source = String(args.source || "stooq").toLowerCase();

  if (!args.tickers) usage("--tickers is required.");
  const tickers = String(args.tickers).split(",").map((t) => t.trim()).filter(Boolean);
  if (tickers.length < 2) usage("Give at least two tickers — a portfolio of one has nothing to optimize.");
  if (!["stooq", "alphavantage", "tiingo", "tiingocrypto"].includes(source)) usage(`Unknown source "${source}".`);
  if (["alphavantage", "tiingo", "tiingocrypto"].includes(source) && !args.key) usage(`--key is required for ${source}.`);

  const outPath = args.out || "prices.csv";
  const adjusted = !!args.adjusted;

  if (!adjusted && source !== "tiingocrypto") {
    console.log("Note: fetching unadjusted closes. Splits and dividends will appear as");
    console.log("      spurious large returns. Pass --adjusted if your source supports it.\n");
  }
  if (adjusted && source === "tiingocrypto") {
    console.log("Note: --adjusted has no effect for tiingocrypto — crypto has no splits/dividends.\n");
  }

  const seriesByTicker = [];
  for (let i = 0; i < tickers.length; i++) {
    const t = tickers[i];
    process.stdout.write(`Fetching ${t} (${i + 1}/${tickers.length}) ... `);
    try {
      const series = source === "stooq" ? await fetchStooq(t)
        : source === "tiingo" ? await fetchTiingo(t, args.key, adjusted, args.start)
        : source === "tiingocrypto" ? await fetchTiingoCrypto(t, args.key, args.start, args.resample)
        : await fetchAlphaVantage(t, args.key, adjusted, !!args.full);
      seriesByTicker.push({ ticker: t, series });
      console.log(`${series.size} rows`);
    } catch (e) {
      console.log("FAILED");
      console.error(`  ${e.message}`);
      process.exit(1);
    }
    if (source === "alphavantage" && i < tickers.length - 1) {
      process.stdout.write("  pausing 13s for the rate limit ... ");
      await sleep(13000);
      console.log("ok");
    }
  }

  const dates = alignSeries(seriesByTicker, args.start);

  if (dates.length < 30) {
    console.error(`\nOnly ${dates.length} dates are common to all tickers — too few for a`);
    console.error("usable estimate. The symbols may trade on different calendars or have");
    console.error("little overlapping history.");
    process.exit(1);
  }

  const maxRows = Math.max(...seriesByTicker.map(({ series }) => series.size));
  const dropped = maxRows - dates.length;
  if (dropped > 0) console.log(`\n${dropped} dates dropped: not every ticker traded on them.`);

  fs.writeFileSync(outPath, buildCSV(seriesByTicker, dates));

  const years = dates.length / 252;
  console.log(`\nWrote ${outPath}`);
  console.log(`  ${dates.length} dates from ${dates[0]} to ${dates[dates.length - 1]} (~${years.toFixed(1)} years)`);
  console.log(`  ${tickers.length} assets: ${tickers.join(", ")}`);
  if (years < 3) {
    console.log(`\n  Warning: under three years of history. Expected-return estimates from`);
    console.log(`  so short a window carry very large standard errors.`);
  }
  console.log(`\nLoad this file into the optimizer, or estimate from the command line:`);
  console.log(`  node -e "const D=require('./data-pipeline.js'),fs=require('fs');`);
  console.log(`    const e=D.estimateFromCSV(fs.readFileSync('${outPath}','utf8'));`);
  console.log(`    console.log(e.names, e.mu, e.vols); e.quality.warnings.forEach(w=>console.log(w));"`);
}

module.exports = {
  parseStooqCSV, parseAlphaVantageJSON, parseTiingoJSON, parseTiingoCryptoJSON,
  alignSeries, buildCSV, parseArgs,
};

if (require.main === module) {
  main().catch((e) => {
    console.error(`\nUnexpected failure: ${e.message}`);
    process.exit(1);
  });
}
