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
//
// TESTING NOTE
//   Everything that interprets a provider's response is a pure function
//   (parseStooqCSV, parseAlphaVantageJSON, parseTiingoJSON, alignSeries,
//   buildCSV) and is covered by fetch-market-data.test.js. Verified against
//   all three providers' live endpoints in 2026-09 (see findings above);
//   provider policies can change again, so re-verify if a source starts
//   failing.
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
  node fetch-market-data.js --source=<stooq|alphavantage|tiingo> --tickers=A,B,C [options]

Options:
  --source=NAME       Data provider. Default: stooq
  --tickers=LIST      Comma-separated symbols. Required, at least two.
  --out=FILE          Output path. Default: prices.csv
  --key=KEY           API key/token. Required for alphavantage and tiingo.
  --start=YYYY-MM-DD  Earliest date to keep. Optional. Also passed to Tiingo
                      as startDate, so it need not fetch history it will
                      discard.
  --adjusted          Use split/dividend-adjusted closes where available.
                      (Alpha Vantage: premium-only as of 2026-09 — will fail
                      with a clear error on a free key. Tiingo: works on the
                      free tier.)
  --full              alphavantage only: request outputsize=full (premium-only
                      as of 2026-09). Default is compact (~100 observations).
                      tiingo always fetches full history (from 1900-01-01, or
                      --start) regardless of this flag.
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

// ---------------------------------------------------------------- main

async function main() {
  const args = parseArgs(process.argv);
  const source = String(args.source || "stooq").toLowerCase();

  if (!args.tickers) usage("--tickers is required.");
  const tickers = String(args.tickers).split(",").map((t) => t.trim()).filter(Boolean);
  if (tickers.length < 2) usage("Give at least two tickers — a portfolio of one has nothing to optimize.");
  if (!["stooq", "alphavantage", "tiingo"].includes(source)) usage(`Unknown source "${source}".`);
  if ((source === "alphavantage" || source === "tiingo") && !args.key) usage(`--key is required for ${source}.`);

  const outPath = args.out || "prices.csv";
  const adjusted = !!args.adjusted;

  if (!adjusted) {
    console.log("Note: fetching unadjusted closes. Splits and dividends will appear as");
    console.log("      spurious large returns. Pass --adjusted if your source supports it.\n");
  }

  const seriesByTicker = [];
  for (let i = 0; i < tickers.length; i++) {
    const t = tickers[i];
    process.stdout.write(`Fetching ${t} (${i + 1}/${tickers.length}) ... `);
    try {
      const series = source === "stooq" ? await fetchStooq(t)
        : source === "tiingo" ? await fetchTiingo(t, args.key, adjusted, args.start)
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

module.exports = { parseStooqCSV, parseAlphaVantageJSON, parseTiingoJSON, alignSeries, buildCSV, parseArgs };

if (require.main === module) {
  main().catch((e) => {
    console.error(`\nUnexpected failure: ${e.message}`);
    process.exit(1);
  });
}
