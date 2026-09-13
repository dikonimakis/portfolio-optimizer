const F = require("./fetch-news-sentiment.js");

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
const close = (a, b, tol = 1e-9) => Math.abs(a - b) < tol;
const section = (s) => console.log(`\n== ${s}`);

// ---------------------------------------------------------------- response parsing
section("Alpha Vantage NEWS_SENTIMENT response parsing");
{
  const body = JSON.stringify({
    feed: [
      {
        time_published: "20240102T093000",
        ticker_sentiment: [
          { ticker: "SPY", relevance_score: "0.8", ticker_sentiment_score: "0.35", ticker_sentiment_label: "Bullish" },
          { ticker: "AGG", relevance_score: "0.2", ticker_sentiment_score: "-0.05", ticker_sentiment_label: "Neutral" },
        ],
      },
      {
        time_published: "20240102T140000",
        ticker_sentiment: [
          { ticker: "SPY", relevance_score: "0.5", ticker_sentiment_score: "-0.20", ticker_sentiment_label: "Bearish" },
        ],
      },
    ],
  });
  const rows = F.parseNewsSentimentJSON(body, "SPY");
  ok("extracts one row per ticker mention, not per article", rows.length === 3);
  ok("reads relevance and score as numbers", rows[0].relevance === 0.8 && rows[0].score === 0.35);
  ok("a single article mentioning two tickers produces a row for each",
    rows.filter((r) => r.time_published === undefined || true).some((r) => r.ticker === "AGG"));
  ok("carries the sentiment label through", rows[0].label === "Bullish");
  ok("carries the publish time through", rows[0].publishedAt === "20240102T093000");
}
{
  ok("an empty feed produces no rows",
    F.parseNewsSentimentJSON(JSON.stringify({ feed: [] }), "X").length === 0);
  ok("an article with no ticker_sentiment array is skipped without crashing",
    F.parseNewsSentimentJSON(JSON.stringify({ feed: [{ time_published: "t" }] }), "X").length === 0);
}
{
  const body = JSON.stringify({
    feed: [{ ticker_sentiment: [
      { ticker: "SPY", relevance_score: "not-a-number", ticker_sentiment_score: "0.2" },
      { ticker: "SPY", relevance_score: "0.5", ticker_sentiment_score: "also-bad" },
      { ticker: "SPY", relevance_score: "0.5", ticker_sentiment_score: "0.2" },
    ] }],
  });
  const rows = F.parseNewsSentimentJSON(body, "SPY");
  ok("mentions with non-numeric relevance or score are dropped, not NaN-poisoned",
    rows.length === 1 && rows[0].score === 0.2);
}
{
  // Alpha Vantage signals problems with HTTP 200, so the body must be read.
  ok("an API error message is surfaced",
    throwsWith(() => F.parseNewsSentimentJSON(
      JSON.stringify({ "Error Message": "Invalid API call." }), "X"), "Invalid API call"));
  ok("a rate-limit note is surfaced as a rate limit",
    throwsWith(() => F.parseNewsSentimentJSON(
      JSON.stringify({ "Note": "5 calls per minute" }), "X"), "rate limit"));
  ok("an invalid-key Information response is surfaced",
    throwsWith(() => F.parseNewsSentimentJSON(
      JSON.stringify({ "Information": "the parameter apikey is invalid" }), "X"), "apikey"));
  ok("non-JSON is rejected clearly",
    throwsWith(() => F.parseNewsSentimentJSON("<html>502</html>", "X"), "non-json"));
  ok("a response with no feed array is rejected",
    throwsWith(() => F.parseNewsSentimentJSON(JSON.stringify({ meta: {} }), "SPY"), "no article feed"));
  ok("accepts an already-parsed object as well as a string",
    F.parseNewsSentimentJSON({ feed: [] }, "X").length === 0);
}

// ---------------------------------------------------------------- aggregation
section("Aggregation across articles");
{
  const rows = [
    { ticker: "SPY", relevance: 0.8, score: 0.4 },
    { ticker: "SPY", relevance: 0.2, score: -0.4 },
  ];
  const agg = F.aggregateSentiment(rows);
  // relevance-weighted mean: (0.8*0.4 + 0.2*-0.4) / (0.8+0.2) = (0.32-0.08)/1.0 = 0.24
  ok("aggregation is relevance-weighted, not a plain average",
    close(agg.SPY.score, 0.24, 1e-9), `got ${agg.SPY.score}`);
  ok("a plain average would have given a different (wrong) answer",
    Math.abs(agg.SPY.score - (0.4 + -0.4) / 2) > 1e-6);
  ok("article count is tracked", agg.SPY.articles === 2);
}
{
  const rows = [
    { ticker: "A", relevance: 1, score: 0.5 },   // bullish (> 0.15)
    { ticker: "A", relevance: 1, score: -0.5 },  // bearish (< -0.15)
    { ticker: "A", relevance: 1, score: 0.05 },  // neutral
  ];
  const agg = F.aggregateSentiment(rows);
  ok("bullish/bearish/neutral counts use the same threshold convention as Alpha Vantage",
    agg.A.bullish === 1 && agg.A.bearish === 1 && agg.A.articles === 3);
}
{
  const agg = F.aggregateSentiment([]);
  ok("aggregating zero rows produces an empty map, not an error", Object.keys(agg).length === 0);
}
{
  // A ticker with total relevance-weight of zero (e.g. every mention scored
  // 0 relevance) must not divide by zero.
  const rows = [{ ticker: "X", relevance: 0, score: 0.9 }];
  const agg = F.aggregateSentiment(rows);
  ok("zero total relevance weight falls back to 0 rather than NaN or Infinity",
    agg.X.score === 0);
}

// ---------------------------------------------------------------- output shape
section("Output JSON shape");
{
  const agg = { SPY: { score: 0.24, articles: 2, bullish: 1, bearish: 1 } };
  const out = F.buildOutputJSON(agg, ["SPY", "AGG"]);
  ok("every requested ticker appears in the output, even with no coverage",
    "SPY" in out.tickers && "AGG" in out.tickers);
  ok("an uncovered ticker gets a null score and zero counts, not an absent entry",
    out.tickers.AGG.score === null && out.tickers.AGG.articles === 0);
  ok("a covered ticker's aggregate carries through unchanged",
    out.tickers.SPY.score === 0.24 && out.tickers.SPY.articles === 2);
  ok("a generation timestamp is included", typeof out.generatedAt === "string" && out.generatedAt.length > 0);

  const parsed = JSON.parse(JSON.stringify(out));
  ok("the output round-trips through JSON.stringify/parse cleanly",
    parsed.tickers.SPY.score === 0.24);
}

// ---------------------------------------------------------------- CLI arguments
section("Command line parsing");
{
  const a = F.parseArgs(["node", "s.js", "--tickers=SPY,AGG", "--key=abc123", "--limit=25"]);
  ok("key=value arguments are parsed", a.tickers === "SPY,AGG" && a.key === "abc123");
  ok("numeric-looking values stay as strings for the caller to parse", a.limit === "25");
  ok("an absent flag is undefined", a.topics === undefined);
}

// ---------------------------------------------------------------- summary
console.log(`\n${"=".repeat(64)}`);
console.log(`${pass} passed, ${fail} failed`);
if (fail) { console.log("Failures:"); failures.forEach((f) => console.log(`  - ${f}`)); }
console.log("=".repeat(64));
console.log("\nNot covered: the HTTPS request itself, which needs network access to");
console.log("Alpha Vantage. Everything that interprets its responses is tested above.");
process.exit(fail ? 1 : 0);
