#!/usr/bin/env node
// ============================================================================
// fetch-news-sentiment.js
//
// Downloads per-ticker news sentiment and writes a JSON file the dashboard's
// Black-Litterman sentiment tilt can consume directly:
//
//   { "generatedAt": "...", "tickers": {
//       "SPY": { "score": 0.18, "articles": 34, "bullish": 21, "bearish": 6 },
//       ...
//   }}
//
// Run this on your own machine with your own API key. Keys must never be
// placed in browser code, and this environment cannot verify a live news feed
// (no network route to one here — see the TESTING NOTE below).
//
// USAGE
//   node fetch-news-sentiment.js --tickers=SPY,AGG,GLD --key=YOUR_KEY
//
// SOURCE
//   Alpha Vantage NEWS_SENTIMENT. Free tier, same key as fetch-market-data.js.
//   Chosen deliberately over building a sentiment scorer from scratch: a
//   bag-of-words sentiment model written here would be low-quality and would
//   dress up noise as a measurement. Alpha Vantage's endpoint returns
//   per-article, per-ticker relevance and sentiment scores from an existing
//   scoring service, which is a more honest foundation than inventing one.
//
// WHAT THIS DOES NOT DO
//   It does not decide whether the resulting score is worth acting on. Section
//   3i / test-sentiment-value.js measured that a sentiment-style signal only
//   improves performance when its informativeness is roughly 0.6+ out of 1.0,
//   and a plausible real-world estimate (0.15) is net negative once trading
//   costs are included. Nothing in this script establishes where a real feed
//   falls on that scale — that would require its own backtest against the
//   feed's actual historical accuracy, which is a separate, harder project.
//
// TESTING NOTE
//   As with fetch-market-data.js, everything that INTERPRETS a response is a
//   pure function (parseNewsSentimentJSON, aggregateSentiment) and is covered
//   by fetch-news-sentiment.test.js. Verified against the live endpoint in
//   2026-09 (a real key, AAPL, 50 mentions returned and correctly aggregated).
//
//   MEASURED: --limit does not appear to take effect on a free key. Requesting
//   limit=10 and limit=200 against the same ticker both returned exactly 50
//   articles ("items": 50 in both raw responses). Whether that is a free-tier
//   cap or a documented minimum/behavior on Alpha Vantage's side was not
//   established — only the observed behavior was. Do not rely on --limit to
//   reduce or increase article count on a free key.
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
  node fetch-news-sentiment.js --tickers=A,B,C --key=YOUR_KEY [options]

Options:
  --tickers=LIST   Comma-separated symbols. Required.
  --key=KEY        Alpha Vantage API key. Required.
  --out=FILE       Output path. Default: sentiment.json
  --topics=LIST    Optional Alpha Vantage topic filter (e.g. technology,earnings)
  --limit=N        Articles per ticker to request. Default: 50. NOTE: measured
                   to have no effect on a free key as of 2026-09 (10 and 200
                   both returned exactly 50 articles) — see TESTING NOTE above.
`);
  process.exit(1);
}

// ---------------------------------------------------------------- pure parsing
// Separated from the network call so it can be tested against a recorded
// response shape without needing a live connection.

// Alpha Vantage returns one "feed" of articles; each article carries a
// ticker_sentiment array with a relevance_score and ticker_sentiment_score per
// mentioned ticker. An article can mention several tickers at once, so
// aggregation happens PER TICKER across the whole feed, not per article.
function parseNewsSentimentJSON(body, ticker = "series") {
  let json;
  try { json = typeof body === "string" ? JSON.parse(body) : body; }
  catch (e) { throw new Error(`Alpha Vantage returned non-JSON for "${ticker}".`); }

  if (json["Error Message"]) throw new Error(`Alpha Vantage: ${json["Error Message"]}`);
  if (json["Note"]) throw new Error(`Alpha Vantage rate limit hit. Wait a minute and retry.`);
  if (json["Information"]) throw new Error(`Alpha Vantage: ${json["Information"]}`);

  const feed = json.feed;
  if (!Array.isArray(feed)) throw new Error(`No article feed in the Alpha Vantage response for "${ticker}".`);

  const rows = [];
  for (const article of feed) {
    const mentions = Array.isArray(article.ticker_sentiment) ? article.ticker_sentiment : [];
    for (const m of mentions) {
      const relevance = parseFloat(m.relevance_score);
      const score = parseFloat(m.ticker_sentiment_score);
      if (!Number.isFinite(relevance) || !Number.isFinite(score)) continue;
      rows.push({
        ticker: m.ticker,
        relevance,
        score,
        label: m.ticker_sentiment_label || null,
        publishedAt: article.time_published || null,
      });
    }
  }
  return rows;
}

// Aggregates raw per-article rows into one summary per ticker: a
// relevance-weighted mean score (an article barely about the ticker should
// count for less than one squarely about it), plus counts a user can use to
// judge how much to trust the score.
function aggregateSentiment(rows) {
  const byTicker = new Map();
  for (const r of rows) {
    if (!byTicker.has(r.ticker)) {
      byTicker.set(r.ticker, { weightedSum: 0, weightSum: 0, articles: 0, bullish: 0, bearish: 0 });
    }
    const acc = byTicker.get(r.ticker);
    acc.weightedSum += r.relevance * r.score;
    acc.weightSum += r.relevance;
    acc.articles += 1;
    if (r.score > 0.15) acc.bullish += 1;
    else if (r.score < -0.15) acc.bearish += 1;
  }
  const out = {};
  for (const [ticker, acc] of byTicker) {
    out[ticker] = {
      score: acc.weightSum > 0 ? acc.weightedSum / acc.weightSum : 0,
      articles: acc.articles,
      bullish: acc.bullish,
      bearish: acc.bearish,
    };
  }
  return out;
}

function buildOutputJSON(sentimentByTicker, tickers) {
  const tickersOut = {};
  for (const t of tickers) {
    tickersOut[t] = sentimentByTicker[t] || { score: null, articles: 0, bullish: 0, bearish: 0 };
  }
  return { generatedAt: new Date().toISOString(), tickers: tickersOut };
}

// ---------------------------------------------------------------- network

function get(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { "User-Agent": "portfolio-optimizer/1.0" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        return resolve(get(res.headers.location));
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode} from ${url}`));
      }
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (c) => (body += c));
      res.on("end", () => resolve(body));
    });
    req.on("error", reject);
    req.setTimeout(30000, () => req.destroy(new Error("Request timed out after 30s")));
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchNewsSentiment(ticker, key, opts) {
  const params = new URLSearchParams({
    function: "NEWS_SENTIMENT",
    tickers: ticker,
    apikey: key,
    limit: String(opts.limit || 50),
  });
  if (opts.topics) params.set("topics", opts.topics);
  const body = await get(`https://www.alphavantage.co/query?${params.toString()}`);
  return parseNewsSentimentJSON(body, ticker);
}

// ---------------------------------------------------------------- main

async function main() {
  const args = parseArgs(process.argv);
  if (!args.tickers) usage("--tickers is required.");
  if (!args.key) usage("--key is required.");
  const tickers = String(args.tickers).split(",").map((t) => t.trim()).filter(Boolean);
  if (tickers.length === 0) usage("No valid tickers given.");
  const outPath = args.out || "sentiment.json";

  console.log("Note: sentiment scores are noisy and, per this project's own");
  console.log("      measurements, only help a portfolio when they are substantially");
  console.log("      more informative than a typical news feed is likely to be.");
  console.log("      See test-sentiment-value.js and README section 3l.\n");

  let allRows = [];
  for (let i = 0; i < tickers.length; i++) {
    const t = tickers[i];
    process.stdout.write(`Fetching sentiment for ${t} (${i + 1}/${tickers.length}) ... `);
    try {
      const rows = await fetchNewsSentiment(t, args.key, { limit: args.limit, topics: args.topics });
      allRows = allRows.concat(rows);
      console.log(`${rows.filter((r) => r.ticker === t).length} mentions`);
    } catch (e) {
      console.log("FAILED");
      console.error(`  ${e.message}`);
      process.exit(1);
    }
    if (i < tickers.length - 1) {
      process.stdout.write("  pausing 13s for the rate limit ... ");
      await sleep(13000);
      console.log("ok");
    }
  }

  const aggregated = aggregateSentiment(allRows);
  const output = buildOutputJSON(aggregated, tickers);
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2));

  console.log(`\nWrote ${outPath}`);
  for (const t of tickers) {
    const s = output.tickers[t];
    console.log(`  ${t.padEnd(8)} score=${s.score != null ? s.score.toFixed(3) : "n/a"}  articles=${s.articles}  bullish=${s.bullish}  bearish=${s.bearish}`);
  }
  const uncovered = tickers.filter((t) => output.tickers[t].articles === 0);
  if (uncovered.length > 0) {
    console.log(`\n${uncovered.length} ticker(s) had no news mentions: ${uncovered.join(", ")}`);
  }
}

module.exports = { parseArgs, parseNewsSentimentJSON, aggregateSentiment, buildOutputJSON };

if (require.main === module) {
  main().catch((e) => {
    console.error(`\nUnexpected failure: ${e.message}`);
    process.exit(1);
  });
}
