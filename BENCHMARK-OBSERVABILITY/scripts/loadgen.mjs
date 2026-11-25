#!/usr/bin/env node
// Drive the grid like a production client would: several users, each with
// their own concurrent stream of requests, all through the gateway.
//
//   node scripts/loadgen.mjs --users 4 --concurrency 3
//   node scripts/loadgen.mjs --users 1 --concurrency 8 --idempotency-demo
//   node scripts/loadgen.mjs --users 16 --concurrency 1-8 \
//        --prompt-tokens 100-2000 --max-tokens 64-2048 --seed 42
//
// Every request carries a unique nonce, so the gateway's exact-match cache
// never answers one — otherwise the engine is never touched and the run
// proves nothing (this is how the first soak was invalidated).
//
// Production traffic is not uniform: one user sends a hundred-token prompt,
// the next sends two thousand, and both want a long answer. A load that
// gives every request the same size cannot tell you which ceiling bound —
// so the sizes here are drawn per request, not fixed for the whole run:
//
//   --users N                how many users
//   --concurrency N|MIN-MAX  workers per user, drawn per user
//   --requests  N|MIN-MAX    requests per user, drawn per user
//                            (default: that user's own concurrency)
//   --prompt-tokens N|MIN-MAX  input tokens per request, drawn per request
//   --max-tokens   N|MIN-MAX  output cap per request, drawn per request
//   --max-context N           the engine's context window. A request longer
//                             than this is refused as `too_long`, which is a
//                             client bug rather than a scheduling result, so
//                             the client never sends one.
//   --concurrency-limit N      the per-user concurrency this client declares
//                             to the gateway (default: each user declares
//                             the concurrency it actually uses)
//   --seed N                  the same seed draws the same mix every run
//
// A request's size is a function of its (user, seq) and the seed, not of the
// order the requests happen to arrive in: an experiment that cannot be
// replayed is not an experiment.
//
// Reads the decision headers the grid sets on every response:
//   x-scheduler-decision   admit | hold | reject
//   x-scheduler-reason     fit | capacity | kv_capacity | physical_capacity
//                          | user_concurrency | user_reserved | queue_full
//                          | user_queue_full | too_long
//   x-scheduler-hold-time  seconds the request sat in the scheduler's queue
//   x-kv-blocks-reserved   KV blocks the ledger promised this request
//   x-kv-blocks-demand    KV blocks this request needs if it is admitted
//   x-gateway-cache        hit | miss
//
// `reserved` and `demand` are different numbers on purpose. `demand` is what
// the request needs; `reserved` is what the ledger gave it. A held request has
// demand > 0 and reserved 0 — it was refused room, not blocks. Reading only
// `reserved` makes a held request look like it needed nothing.

const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) =>
    a.startsWith("--") ? [[a.slice(2), all[i + 1]?.startsWith("--") ? true : all[i + 1]]] : []
  )
);

const URL = args.url ?? "http://localhost:9000";
const USERS = Number(args.users ?? 4);
const CONCURRENCY = parseRange(args.concurrency ?? 3, "concurrency");
const REQUESTS = args.requests === undefined ? null : parseRange(args.requests, "requests");
const PROMPT = parseRange(args["prompt-tokens"] ?? 64, "prompt-tokens");
const MAXOUT = parseRange(args["max-tokens"] ?? 64, "max-tokens");
const CONCURRENCY_LIMIT =
  args["concurrency-limit"] === undefined ? null : Number(args["concurrency-limit"]);
const MAX_CONTEXT = Number(args["max-context"] ?? 4096);
const IDEMPOTENCY_DEMO = Boolean(args["idempotency-demo"]);
const SEED = args.seed === undefined ? (Math.random() * 2 ** 32) >>> 0 : Number(args.seed) >>> 0;

// "64" is one size; "100-2000" is a range of sizes.
function parseRange(spec, what) {
  if (spec === true || spec === undefined || spec === null) {
    throw new Error(`--${what} needs a number or a MIN-MAX range`);
  }
  const raw = String(spec).trim();
  const m = /^(\d+)\s*-\s*(\d+)$/.exec(raw);
  if (m) {
    const lo = Number(m[1]);
    const hi = Number(m[2]);
    if (lo < 1 || hi < lo) throw new Error(`--${what}: bad range "${raw}"`);
    return { lo, hi };
  }
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 1) throw new Error(`--${what}: bad number "${raw}"`);
  return { lo: n, hi: n };
}

// A stable hash of (seed, key) -> [0, 1). Same seed, same mix, whatever order
// the requests arrived in.
function hash01(key) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  h ^= h >>> 16;
  h = Math.imul(h, 2246822507) >>> 0;
  h ^= h >>> 13;
  return (h >>> 0) / 4294967296;
}

const draw = (range, key) =>
  range.lo + Math.floor(hash01(`${SEED}/${key}`) * (range.hi - range.lo + 1));

// One "word " is one token for this tokenizer, so n repeats is an n-token
// prompt.
const pad = (n) => "word ".repeat(Math.ceil(n)).trim();
const t0 = Date.now();
const results = [];
let clamped = 0;

async function one(user, seq, idemKey, limit) {
  const wantOut = draw(MAXOUT, `${user}/${seq}/out`);
  const wantPrompt = draw(PROMPT, `${user}/${seq}/prompt`);
  // The output cap comes first: the prompt is what gets shortened to fit.
  const maxOut = Math.min(wantOut, MAX_CONTEXT - 1);
  const promptTokens = Math.max(1, Math.min(wantPrompt, MAX_CONTEXT - maxOut));
  if (maxOut !== wantOut || promptTokens !== wantPrompt) clamped++;

  const prompt = `${pad(promptTokens)} [nonce ${user}-${seq}-${Math.random().toString(36).slice(2)}]`;
  const body = {
    model: "local",
    prompt,
    max_tokens: maxOut,
    temperature: 0,
    concurrency_limit: limit,
  };
  const headers = { "content-type": "application/json", "x-user-id": user };
  if (idemKey) headers["idempotency-key"] = idemKey;

  const started = Date.now();
  try {
    const res = await fetch(`${URL}/v1/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    const text = await res.text();
    const decision = res.headers.get("x-scheduler-decision") ?? "-";
    const hold = Number(res.headers.get("x-scheduler-hold-time") ?? 0);
    // Null, not zero, when the header is absent: the request never reached
    // the scheduler (a cache hit, a gateway error), so its demand was never
    // measured. Zero would claim it needed no blocks.
    const numOrNull = (v) => (v === null ? null : Number(v));
    const blocks = numOrNull(res.headers.get("x-kv-blocks-reserved"));
    const demand = numOrNull(res.headers.get("x-kv-blocks-demand"));
    const cache = res.headers.get("x-gateway-cache") ?? "miss";
    const replay = res.headers.get("x-idempotent-replay") ?? "-";
    const ms = Date.now() - started;
    let tokens = 0;
    let usedPrompt = 0;
    let usedOutput = 0;
    let bodyReason = "";
    try {
      const j = JSON.parse(text);
      tokens = j.usage?.total_tokens ?? 0;
      usedPrompt = j.usage?.prompt_tokens ?? 0;
      usedOutput = j.usage?.completion_tokens ?? 0;
      bodyReason = j.reason ?? "";
    } catch {}
    // The scheduler says why it decided; a request the gateway refused never
    // reached it, so the body is the only place that reason appears.
    const reason = res.headers.get("x-scheduler-reason") || bodyReason;
    results.push({
      user,
      seq,
      status: res.status,
      decision,
      hold,
      ms,
      tokens,
      usedPrompt,
      usedOutput,
      promptTokens,
      maxOut,
      blocks,
      demand,
      cache,
      replay,
      reason,
    });
    console.log(
      `${String(user).padEnd(8)} #${String(seq).padEnd(3)} ${res.status} ` +
        `${decision.padEnd(6)} ${String(ms).padStart(6)}ms hold=${hold.toFixed(2)}s ` +
        `p=${String(promptTokens).padStart(4)} o=${String(maxOut).padStart(4)} ` +
        `tok=${String(tokens).padStart(5)} ` +
        `blk=${String(blocks ?? "-").padStart(4)} dem=${String(demand ?? "-").padStart(4)} ` +
        `cache=${cache}${reason ? ` reason=${reason}` : ""}` +
        `${replay === "true" ? " IDEMPOTENT-REPLAY" : ""}`
    );
    return { res, text };
  } catch (err) {
    results.push({ user, seq, status: 0, decision: "err", hold: 0, ms: Date.now() - started, tokens: 0 });
    console.log(`${String(user).padEnd(8)} #${String(seq).padEnd(3)} ERR ${err.message}`);
  }
}

const fmtRange = (r) => (r.lo === r.hi ? String(r.lo) : `${r.lo}-${r.hi}`);

// The per-user concurrency this client declares to the gateway. Declaring
// less than the concurrency you actually use is a client bug: the gateway
// refuses the excess with `user_concurrency`, which then shows up in the
// results as a 429 and hides the scheduler's own ceilings. So by default
// each user declares its own.
const limitFor = (conc) => CONCURRENCY_LIMIT ?? conc;

// The plan, before it runs: what each user was dealt.
const plan = Array.from({ length: USERS }, (_, u) => {
  const user = `user-${u}`;
  const conc = draw(CONCURRENCY, `${user}/concurrency`);
  const reqs = REQUESTS ? draw(REQUESTS, `${user}/requests`) : conc;
  return { user, conc, reqs };
});

async function main() {
  console.log(
    `→ ${URL}  users=${USERS} concurrency/user=${fmtRange(CONCURRENCY)} ` +
      `req/user=${REQUESTS ? fmtRange(REQUESTS) : "=concurrency"} ` +
      `limit/user=${CONCURRENCY_LIMIT ?? "=concurrency"} ` +
      `prompt=${fmtRange(PROMPT)}tok max_out=${fmtRange(MAXOUT)}tok ` +
      `context=${MAX_CONTEXT} seed=${SEED}`
  );

  // Per user: CONCURRENCY workers draining that user's REQUESTS queue, so
  // one user's burst cannot silently become the whole batch.
  const perUser = plan.map(({ user, conc, reqs }) => {
    let next = 0;
    const worker = async () => {
      while (next < reqs) await one(user, next++, undefined, limitFor(conc));
    };
    return Array.from({ length: Math.min(conc, reqs) }, () => worker);
  });
  await Promise.all(perUser.flat().map((w) => w()));

  if (IDEMPOTENCY_DEMO) {
    console.log("\n-- idempotency: same user + same key, twice, concurrently --");
    const key = `idem-${Date.now()}`;
    // Concurrent, not sequential. The registry holds a key only while its
    // request is in flight (design §9.3: the entry is removed when the
    // request finishes), so a retry *after* the first completes is a new
    // request, not a duplicate. The 409 is for a client that resubmits
    // while the first is still running.
    await Promise.all([one("idem-user", 1, key, limitFor(2)), one("idem-user", 2, key, limitFor(2))]);
  }

  const by = (k) => results.reduce((acc, r) => ((acc[r[k]] = (acc[r[k]] ?? 0) + 1), acc), {});
  const holds = results.filter((r) => r.hold > 0).map((r) => r.hold);
  const stat = (xs) => {
    if (!xs.length) return "n/a";
    const s = [...xs].sort((a, b) => a - b);
    return `min ${s[0]}  p50 ${s[Math.floor(s.length / 2)]}  max ${s.at(-1)}`;
  };
  // A zero *token* count means "not measured". A zero *block* count is a
  // measurement: a held request reserved nothing. Dropping those rows hid
  // exactly the requests that were waiting, so blocks get their own filter.
  const pick = (k) => results.map((r) => r[k]).filter((v) => typeof v === "number" && v > 0);
  const pickBlocks = (k) => results.map((r) => r[k]).filter((v) => typeof v === "number");

  console.log("\n== summary ==");
  console.log(`requests   ${results.length}`);
  console.log(`statuses   ${JSON.stringify(by("status"))}`);
  console.log(`decisions  ${JSON.stringify(by("decision"))}`);
  console.log(`cache      ${JSON.stringify(by("cache"))}`);
  const reasons = results
    .filter((r) => r.reason)
    .reduce((acc, r) => ((acc[r.reason] = (acc[r.reason] ?? 0) + 1), acc), {});
  if (Object.keys(reasons).length) console.log(`reasons    ${JSON.stringify(reasons)}`);
  console.log(`elapsed    ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  if (holds.length) {
    holds.sort((a, b) => a - b);
    console.log(`hold p50   ${holds[Math.floor(holds.length / 2)].toFixed(2)}s   max ${holds.at(-1).toFixed(2)}s`);
  }
  console.log(`prompt tok asked   ${stat(pick("promptTokens"))}`);
  console.log(`            actual  ${stat(pick("usedPrompt"))}`);
  console.log(`output tok asked   ${stat(pick("maxOut"))}`);
  console.log(`            actual  ${stat(pick("usedOutput"))}`);
  const reserved = pickBlocks("blocks");
  const demands = pickBlocks("demand");
  console.log(`blocks reserved  ${stat(reserved)}   sum ${reserved.reduce((a, b) => a + b, 0)}   n=${reserved.length}`);
  console.log(`blocks demand    ${stat(demands)}   sum ${demands.reduce((a, b) => a + b, 0)}   n=${demands.length}`);
  // The gap between the two lines above: requests that needed blocks and got
  // none, because a ceiling said no before the ledger ever promised them.
  const unpromised = results.filter(
    (r) => typeof r.demand === "number" && r.demand > 0 && (r.blocks ?? 0) === 0
  );
  if (unpromised.length) {
    const why = unpromised.reduce(
      (acc, r) => ((acc[r.reason || "-"] = (acc[r.reason || "-"] ?? 0) + 1), acc),
      {}
    );
    console.log(
      `unpromised  ${unpromised.length} request(s) needed blocks but reserved 0   ${JSON.stringify(why)}`
    );
  }
  if (clamped) console.log(`clamped    ${clamped} request(s) shortened to fit ${MAX_CONTEXT} tokens`);
}

main();
