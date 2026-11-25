# BENCHMARK-OBSERVABILITY

A benchmarking and observability platform for a self-hosted vLLM deployment, on one
RTX 3050 Laptop (6 GB), driven from WSL2 — and, since the production-grid work, the
grid itself.

Four packages, one repo:

| Package | Directory | What it does |
|---|---|---|
| `inferbench` | `benchmarking/` | workload generator, engine modes, results store, report |
| `infergw` | `gateway/` | tier 1 — request router, admission control, exact-match cache, idempotency registry |
| `infersched` | `scheduler/` | tier 2 — KV-block ledger, FIT/HOLD/REJECT admission, hold queue, dispatch |
| `inferobs` | `observability/` | metric names, SLO evaluation, error-budget burn, tracing |

`inferobs` is the shared spine: it owns the metric names, so the code that emits a
metric and the dashboard that queries it cannot drift apart.

The production request path — persistent model on the GPU, concurrent per-user
requests, a scheduler that holds what does not fit — is designed in
[`PRODUCTION-DESIGN.md`](PRODUCTION-DESIGN.md) and **built**. This README is how you
run it.

---

## The grid

```
client
  │
  ▼
┌─────────────────────────────┐   :9000
│  tier 1   infergw           │   router · exact-match cache · per-user registry
│                             │   idempotency · x-user-id scoping
└──────────────┬──────────────┘
               │ POST /internal/v1/submit   → {admit | hold | reject}
               │ GET  /internal/v1/stream/{id}  (SSE)
               ▼
┌─────────────────────────────┐   :9001
│  tier 2   infersched        │   KV-block ledger · admission · hold queue
│  ┌───────────────────────┐  │   round-robin dispatch · reconciliation
│  │ tier 3  vLLM engine  │  │   ← in-process, not a service
│  │         (the GPU)    │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

**The engine is not a service of its own.** The scheduler loads it in-process, so the
KV-block pool is a readable Python object and the ledger cannot lie about it. That is
the whole reason for the tier split: a ledger that has to ask another process over
HTTP what its block count is, is a ledger that is eventually consistent with the one
thing it exists to describe.

Two processes, two ports, one GPU:

| | Port | Holds the GPU? | Knows about |
|---|---|---|---|
| `infergw` | 9000 | no | users, idempotency keys, cache |
| `infersched` | 9001 | **yes** | KV blocks, requests, users' block budgets |

---

## Why WSL2 and library-mode vLLM, not Docker

Docker gives you `vllm serve` behind an HTTP boundary. That is the right answer for
deploying vLLM and the wrong answer for this project, because every question worth
asking here is on the *inside* of that boundary:

- which requests did the scheduler admit this step, and which are waiting?
- how full is the KV cache, and did it have to preempt anything?
- did the prefix cache actually hit, and how often?

None of that crosses an HTTP boundary. So vLLM is installed as a **library** inside
WSL2 and the engine object lives in our own process. Everything else in this repo
follows from that one decision.

Consequence worth knowing up front: the metrics this platform exports
(`engine_*`) are read out of the in-process engine and exported by *our*
server, not by vLLM's own `/metrics`. Same numbers, but you can see where they
come from.

---

## Quickstart

Inside WSL2:

```bash
./scripts/bootstrap_wsl.sh          # uv + python 3.11 + vllm (library)
./scripts/dev.sh                    # scheduler :9001, gateway :9000
```

In a second terminal (the venv lives outside the repo, so point uv at it):

```bash
export UV_PROJECT_ENVIRONMENT=/home/anupam/.venv
uv run --no-sync inferbench bench --experiment smoke --base-url http://localhost:9000
uv run --no-sync inferbench report --experiment smoke
```

`smoke` is 4 requests of 32 in / 32 out. It exists to prove the wiring, not to
measure anything — if it does not pass, nothing downstream will. `bench` exits
non-zero when any request failed, so this test can actually fail.

**Benchmark through the gateway (`:9000`), not the scheduler (`:9001`).** The gateway
is the only place that knows the arrival rate, the admission decisions and the cache
hit. Hitting the scheduler directly measures tier 2 without tier 1, which is a
different experiment.

`dev.sh` refuses to start on a busy port: a leftover engine from an earlier session
answers `/health`, so a naive readiness wait would be satisfied by the *wrong*
process and the real one would die on bind.

### Driving the grid like a client

Point a client at the **gateway**, not the scheduler — the gateway is tier 1, and
the scheduler's port is an internal hop.

**`scripts/spike.py` is the synthesis file**: many users, real requests,
variable prompt and output sizes, and a verdict at the end.

```bash
# inside the VM that runs the gateway
/home/anupam/.venv/bin/python scripts/spike.py \
  --users 24 --concurrency 10 --requests 6 \
  --prompt-tokens 200-2000 --max-tokens 32-512 --seed 7
```

It finds the grid itself. The ports are not part of the run: they are a
property of the machine, and the machine may already have something on
`:9000` (the defaults) — in which case the grid runs on `:9100`. So the
script probes both and tells you which it found:

```
== gateway http://127.0.0.1:9100   scheduler http://127.0.0.1:9101 ==
```

Pass `--url` and `--scheduler-url` only to override that, or if you run the
grid somewhere else entirely.

It is standard-library Python, so it runs anywhere Python does — including the
WSL2 image, which has **no node** and therefore cannot run `loadgen.mjs`. That
matters: the load has to come from inside the VM, or the Windows→WSL2 port
relay saturates and the gateway goes dark in Prometheus
([below](#when-the-gateway-goes-dark-in-prometheus)).

`scripts/loadgen.mjs` is the node equivalent and stays for hosts that have
node:

```bash
node scripts/loadgen.mjs --url http://localhost:9000 --users 1  --concurrency 6
node scripts/loadgen.mjs --url http://localhost:9000 --users 10 --concurrency 4
```

Each user gets its own pool of concurrent workers, so one user's burst cannot
quietly become the whole batch. Every request carries a unique nonce, so the
gateway's cache never answers one — otherwise the engine is never touched and
the run proves nothing.

#### The two populations a run produces, and why they must not be mixed

A run has requests that became **work** and requests a ceiling turned away at
the **door**. They are not the same thing and the wall clock does not measure
them the same way:

```
served    96   p50 33.74s   p95 55.75s   max 58.37s
refused  144   p50    171ms max    332ms   (decided at the door: cost the server nothing)
held      65   wait p50 25.75s   max 45.84s   (all of them were released and served)
```

Conflating them is how a 960-request burst comes to "finish in 3 seconds" while
every served request actually took seconds: the 429s were decided in ~40 ms
each and never became work. `spike.py` prints them in separate rows and never
adds them together.

`blk=0` on a held request is **correct and must stay**: a hold reserves
nothing, because a request that has not been given a slot has no claim on the
pool. What it *wanted* is the other number — `dem=129` — and `spike.py` says so
in one line: *"65 held request(s) wanted 6491 blocks and got 0 — refused room,
not blocks."*

#### Mixed traffic: every request a different size

A load where every request is the same size cannot tell you which ceiling
bound, because the scheduler's decision depends on the *sizes* it is handed.
So the sizes are drawn per request, not fixed for the run. Every one of these
knobs takes a fixed number or a `MIN-MAX` range:

| flag | drawn for | fixed `N` means | `MIN-MAX` means |
|---|---|---|---|
| `--prompt-tokens` | each request | every prompt is `N` tokens | each prompt is drawn from the range |
| `--max-tokens` | each request | every output cap is `N` | each output cap is drawn from the range |
| `--concurrency` | each user | every user gets `N` workers | each user's worker count is drawn |
| `--requests` | each user | every user sends `N` requests | each user's count is drawn (default: their own concurrency) |

```bash
# 16 users, mixed: prompts of 100..2000 tokens, output caps of 64..2048
node scripts/loadgen.mjs --url http://localhost:9100 \
  --users 16 --concurrency 1-8 \
  --prompt-tokens 100-2000 --max-tokens 64-2048 --seed 42

# the same shape, but every request may use the whole 2048-token output budget
node scripts/loadgen.mjs --url http://localhost:9100 \
  --users 16 --concurrency 1-8 \
  --prompt-tokens 100-2000 --max-tokens 2048 --seed 42
```

`--seed` makes the mix reproducible: a request's size is a function of its
`(user, seq)` and the seed, not of the order the requests happened to arrive
in, so the same seed replays the same experiment.

`--max-context` (default 4096, the engine's `max_model_len`) clamps the pair
so `prompt + output` always fits. A request that does not fit is refused by
the scheduler as `too_long` — that is a client bug, not a scheduling result,
so the client never sends one. The summary counts how many were shortened.

The driver prints the decision headers the grid sets on every response:

| header | meaning |
|---|---|
| `x-scheduler-decision` | `admit` (ran now), `hold` (waited for a slot), `reject` |
| `x-scheduler-reason` | *which* ceiling bound decided it: `fit`, `capacity`, `kv_capacity`, `physical_capacity`, `user_reserved`, `user_concurrency`, `too_long` |
| `x-scheduler-hold-time` | seconds the request sat in the scheduler's queue |
| `x-kv-blocks-reserved` | KV blocks the ledger promised this request |
| `x-kv-blocks-demand` | KV blocks this request needs if it is admitted |
| `x-gateway-cache` | `hit` — served by the gateway, never reached the GPU |
| `x-idempotent-replay` | `true` — a duplicate submit, answered `409` |

`x-scheduler-reason` is the one that turns a run into an explanation. A held
request is otherwise just "held": `capacity` means the scheduler's global
in-flight ceiling bound, `kv_capacity` means the block pool is full,
`physical_capacity` means the *free* blocks are gone even though the ledger
still shows room — the reservation was a promise and the pool is the truth,
`user_reserved` means *this user's* budget is exhausted, `user_concurrency`
means the user already has its share of slots. The summary breaks them down.

`reserved` and `demand` are different numbers on purpose. `demand` is what
the request needs; `reserved` is what the ledger gave it. A held request has
demand > 0 and reserved 0 — it was refused room, not blocks. Reading only
`reserved` makes a held request look like it needed nothing.

One wrinkle: the gateway's *own* refusals — `global_limit` (more than 64
requests in flight at once) and `user_concurrency` (more than this user's
declared share) — never reach the scheduler, so they carry no
`x-scheduler-decision` or `x-scheduler-reason` header. Their reason is in the
response body, and the driver falls back to it, so the line reads
`429 -  ... reason=global_limit`: the `-` in the decision column is the tell
that this one was the gateway's ceiling, not the scheduler's.

The per-request line reads `p=` prompt tokens asked, `o=` output cap asked,
`tok=` tokens actually used, `blk=` KV blocks reserved, `dem=` KV blocks
demanded:

```
user-8   #0   200 admit    3511ms hold=0.00s p=1522 o=  85 tok= 1625 blk= 103 cache=miss reason=fit
user-13  #3   429 -         211ms hold=0.00s p=1707 o=1532 tok=    0 blk=   0 cache=miss reason=global_limit
user-10  #5   200 hold   118748ms hold=111.74s p=1124 o=  83 tok= 1222 blk=   0 cache=miss reason=kv_capacity
```

The three decisions read differently, and that is the point:

- **admit** — it fit right away: `blk` is the reservation it now holds.
- **429 `global_limit`** — the *gateway*'s own in-flight ceiling refused
  it before it ever reached the scheduler. That ceiling is
  `max_in_flight (32, learned from the engine) + max_queue_depth_global (256)`
  = **288**. `decision` is `-` because no scheduler was involved. Note
  `blk=0`: nothing was reserved, so nothing leaks.
- **hold** — the scheduler queued it. `blk=0` here too: the scheduler
  reserves blocks only at admit time, so a held request's demand is not yet
  a promise. `dem` is still non-zero — it wanted blocks and was refused
  room, not because it wanted none. The reason says which ceiling bound it.

and the summary reports the distributions the experiment is about — what was
asked, what was used, and what it reserved:

```
== summary ==
requests   73
statuses   {"200":64,"429":9}
decisions  {"-":9,"admit":31,"hold":33}
cache      {"miss":73}
reasons    {"global_limit":9,"fit":31,"kv_capacity":18,"user_reserved":15}
elapsed    155.4s
hold p50   34.91s   max 116.27s
prompt tok asked   min 102  p50 974   max 1992
            actual  min 119  p50 1029  max 2007
output tok asked   min 83   p50 855   max 2039
            actual  min 83   p50 866   max 2039
blocks      min 20   p50 118  max 236   sum 3660
```

Two ceilings bound the same run — `kv_capacity` (the block pool is full) and
`user_reserved` (*this* user's budget is spent) — and a third sits in front
of them at the gateway. A uniform load can only ever show you one of them at
a time, because with every request the same size only one constraint is ever
the tightest. That is the whole reason the sizes are drawn.

`actual` prompt tokens run slightly above `asked`: the nonce that busts the
gateway's cache is itself a few tokens. The summary reports both rather than
hiding it.

`blocks` is what one request needs: `ceil(prompt/16) + ceil(output_cap/16)`
under `reservation_mode: strict`. That is the request's own demand, and it is
what the ledger commits on its behalf — not a claim on the whole pool. When
output caps vary, the *output* term dominates the demand far more often than
the prompt does — a 100-token prompt with a 2048-token output cap needs 135
blocks against a 2000-token prompt with a 32-token cap's 129. Vary both if you
want to see both terms move.

`--idempotency-demo` sends the same `(x-user-id, Idempotency-Key)` twice
*concurrently* and shows the second refused with `409` before it can start a
second engine request. Concurrent, not sequential: the gateway drops a key when
its request finishes, so a retry after the first has completed is a new request,
not a duplicate (design §9.3).

Watch the ledger while it runs:

```bash
curl -s http://localhost:9001/internal/v1/state
```

`allocated_blocks` is what the engine actually granted, `reserved_blocks` is the
ledger's commitment (granted plus what the live requests may still add), and
`drift_blocks` is the difference between what the engine holds and what the
ledger accounts for — it must stay `0`. `per_user` shows each user's running
and queued counts, which is where round-robin fairness is visible.

**On a machine where `localhost:9000` is already taken** — a Docker container
published there will win the port and your client will talk to it instead of
the grid. Check first, then move the grid:

```bash
netstat -ano | findstr :9000
GATEWAY_PORT=9100 SCHEDULER_PORT=9101 ./scripts/dev.sh
```

`dev.sh` keeps the two tiers in step: it exports `INFERSCHED_PORT` and
`INFERGW_SCHEDULER_URL` from those two variables, so they cannot drift apart.

**Stopping must release the GPU.** The engine lives inside the scheduler
process, so `Ctrl-C` on `dev.sh` is the clean way to stop — it signals the
service child before the `uv` wrapper. If you kill the parent by hand, an
orphaned engine child can keep the card:

```bash
pkill -f infersched
nvidia-smi    # the ~5.3 GB must be gone
```

### The A/B baseline path

```bash
MODE=engine ./scripts/dev.sh        # engine :8000, gateway :9000, no scheduler
```

This is the comparison the queueing numbers are drawn against: the same gateway in
front of a bare engine, with no ledger and no hold queue. It costs a second config
path, not a second machine.

---

## How a request flows

```
POST /v1/completions          X-User-Id: alice          Idempotency-Key: k1
  │
  ├─ cache lookup ────────────── hit ──→ 200 (cached)
  ├─ idempotency lookup ──────── seen ─→ 409 + x-idempotent-replay: true
  │
  └─→ POST /internal/v1/submit ─→ tier 2
        │
        ├─ admit  → the engine starts it, the stream comes back
        ├─ hold   → the gateway *waits* on the SSE and relays it when it runs
        └─ reject → 429 (only when the request can never fit)
```

**A hold is not a rejection.** A request that does not fit *right now* waits; only a
request that can never fit is refused — too long for the context, bigger than its
user's whole block budget, or a queue that is full. Mapping a hold to a 429 would
turn ordinary backpressure into a visible error, and `gate_rejected_total` would
become a rate panel that fires during every peak.

**The ledger predicts; the engine disposes.** Before dispatching, the scheduler
reserves KV blocks for the request (prompt blocks + worst-case output blocks under
`strict` mode). The engine grants its own blocks. The gap between the two is
`sched_reservation_drift_blocks`, and it must stay 0 — if it does not, the ledger is
lying about capacity and every admission decision after it is wrong. It is reconciled
on every engine step, where the engine's truth actually changes.

**One user cannot take the whole GPU.** The scheduler caps how many of a user's
requests *run* at once (`max_in_flight_per_user: 4`); the rest wait in that user's
queue, and dispatch is round-robin across users. So two users cannot starve each
other, and one user cannot starve the other by flooding.

**A duplicate submit does not start a second engine request.** The gateway scopes
idempotency keys per user and remembers them for `idempotency_ttl_s`. A duplicate
gets a 409 with the original `x-request-id`, not a second run on the GPU.

---

## The three ways to run the engine

The point of owning the engine object is that you can drive it three different
ways from the same config.

### 1. Step through it — `inferbench inspect`

```bash
uv run --no-sync inferbench inspect --config baseline --workload smoke --max-tokens 64
```

Builds a V1 `LLMEngine`, feeds it a workload, and calls `step()` in a loop,
printing one line per iteration:

```
step 0007 | sched: 2 new + 5 running | tok 5 | kv 38% (612/1600 blk) | 41ms
↳ preempted 3 (kv 96% full)
```

`tok` is the number of *running* requests, which is a lower bound on the
tokens moved: decode emits one token per running sequence, but a prefill step
moves more. `prefix` only appears when the tap could measure it — with
prefix caching off, as in `baseline`, the field is absent rather than `0%`,
because "we could not measure this" is not the same as "this was zero".

This is the mode the whole project exists for. A preemption is printed as its
own line rather than folded into a counter, because "the scheduler evicted work
to make room" is the single most diagnostic event at 6 GB, and a counter
average hides it.

Runs the V1 engine with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, so the engine core
lives in this process and its scheduler is a readable Python object.
`inferbench.instrument` reads them defensively —
a missing attribute yields `None`, never `0`, so "we could not measure this"
never renders as "this was zero".

### 2. Serve it — `inferbench serve`

```bash
uv run --no-sync inferbench serve --config optimized --port 8000
```

Our own FastAPI OpenAI-compatible server (`/v1/completions`, streaming and
not) around an `AsyncLLMEngine`, with the instrumentation tap attached. Exposes
`/health`, `/v1/models`, `/metrics`.

Note this is *not* `vllm serve`. That distinction is the reason the tap can
read the scheduler at all. In grid mode this server is not used — the scheduler
runs the engine itself.

### 3. Offline batch — the runner

```bash
uv run --no-sync inferbench bench --experiment conc_scale --base-url http://localhost:9000
```

Drives an experiment matrix through HTTP with a bounded concurrency, and writes
every request to sqlite.

---

## What the gateway adds

`infergw` is not a load balancer for its own sake. It sits in front for four
specific reasons:

**1. It sees queue time.** `x-gateway-queue-time` is the interval between
accepting a request and forwarding it. The engine cannot report it — by the
time the engine sees the request, that time has already elapsed. Under
saturation this is where latency actually goes, so the benchmark asks the
harness to time the whole server separately from the engine.

**2. It makes overload visible.** Admission control refuses with a `429` instead
of queueing without bound. A gateway that absorbs infinite load reports
wonderful latency numbers right up until the engine dies. Rejections are
counted (`gate_rejected_total`) and are a signal to look at, not an error to
suppress.

**3. It caches, and reports what that saved.** Exact-match on the
decision-relevant fields (prompt, length caps, sampling params) —
deliberately excluding `stream`, so a streaming and non-streaming request for
the same thing share one entry. A hit records the upstream latency it avoided
(`gate_cache_saved_seconds`), which is the number that tells you whether the
cache is worth its complexity.

**4. It scopes the caller.** `X-User-Id` is required (`require_user_id: true`).
Tier 2 uses it for fairness; tier 1 uses it to scope idempotency keys. A request
without the header is a 400, because "anonymous" is not a fairness policy.

Two routing strategies:

- `least_outstanding` (default) — send to the backend with the fewest in-flight
  requests. Used when you are *not* trying to split traffic.
- `weighted` — a fixed proportion, served deterministically round-robin. Used
  for A/B. It is deterministic on purpose: a traffic split that wobbles between
  runs cannot be compared to the run before it.

> **On 6 GB:** `config/gateway.yaml` ships with **one healthy backend** and a
> second marked `healthy: false`. Two 1.5B fp16 engines do not fit on this
> card. The three ways to a true simultaneous A/B — a smaller model, more VRAM,
> or accepting a swap to disk — are documented in that file. The alternative
> was to configure two healthy backends, have it OOM, and let you discover that
> from a stack trace.

---

## What the scheduler adds

`infersched` owns the GPU and everything about capacity. It runs the engine's
step loop itself, so it sees the block pool directly.

| Endpoint | What it is for |
|---|---|
| `POST /internal/v1/submit` | admit / hold / reject, with the reserved-block count |
| `POST /internal/v1/abort` | release a request's reservation |
| `GET /internal/v1/stream/{request_id}` | SSE, one frame per event |
| `GET /internal/v1/state` | live ledger: free/total/reserved/allocated blocks, per-user state |
| `GET /internal/v1/requests/{request_id}` | one request's decision, hold time, outcome |
| `GET /health` | liveness — 200 while the weights are still loading |
| `GET /ready` | readiness — 503 until the engine is attached, then 200 |

`curl -s localhost:9001/internal/v1/state` is the fastest way to see what the
grid thinks it has:

```json
{"ready": true, "inflight": 12, "held": 3, "free_blocks": 3041,
 "total_blocks": 3662, "reserved_blocks": 621, "allocated_blocks": 621,
 "drift_blocks": 0, "per_user": {"alice": {"inflight": 2, "queued": 1}}}
```

`drift_blocks: 0` is the invariant. `allocated_blocks` is what the engine
actually granted, as a union so a shared prefix counts once. `reserved_blocks`
is the ledger's *commitment*: those granted blocks plus what the live requests
may still add. A request that is reserved but not yet dispatched is **not**
drift — it is a request in flight.

---

## Metrics: one source of truth

Every metric name and label key lives in
[`observability/src/inferobs/metric_names.py`](observability/src/inferobs/metric_names.py).

Five prefixes, by who can observe the thing:

| Prefix | Emitted by | Examples |
|---|---|---|
| `gate_*` | the gateway | queue time, cache hits, rejections, idempotent replays |
| `sched_*` | the scheduler | free/total/reserved/allocated blocks, drift, hold depth, dispatch latency |
| `engine_*` | the tap in `inferbench.instrument` | KV usage, prefix hit rate, preemptions |
| `bench_*` | a benchmark run | `bench_metric{...}` per run/config/concurrency |
| `nvidia_smi_*` | `dcgm-exporter` | GPU utilisation, power, temperature |

The benchmark's own results reach Prometheus through the **pushgateway** —
they are per-run facts with a finite lifetime, which is what a pushgateway is
for. They land as one `bench_metric` series with `metric` and `stat` labels
rather than dozens of named gauges, so adding a measurement is a data change
rather than a code change.

If you rename anything in `metric_names.py`, three places must move with it:
the dashboards in `observability/grafana/dashboards/`, the alert rules in
`observability/alerts/slo_rules.yml`, and the K8s `ServiceMonitor`.

### Alerts

`observability/alerts/slo_rules.yml` carries five scheduler alerts:

| Alert | Fires when |
|---|---|
| `ReservationDrift` | `sched_reservation_drift_blocks` is not 0 — the ledger has drifted from the engine |
| `UnexpectedPreemption` | `sched_preemption_total` increased in 5m — the engine evicted work, which strict mode exists to prevent |
| `SchedulerHoldQueueGrowing` | the hold queue is not draining |
| `SchedulerDispatchStalled` | a slot freed but nothing started |
| `SchedulerDown` | `sched_total_blocks` is absent or 0 |

The whole rule stack (eleven alerts and six recording rules) exists twice: once
locally, once in `deploy/k8s/prometheusrule.yaml`. `tests/test_alerts.py` fails
if the two stacks drift apart, and fails if any rule or dashboard panel queries
a metric nothing emits — a rule naming a metric that does not exist never
fires, and a panel querying one is a grey box that looks like an outage.

---

## Adding an experiment

Experiments are data, not code. `config/experiments.yaml`:

```yaml
- id: conc_scale
  workload: concurrent_chat
  engine: vllm
  axes:
    concurrency: [1, 4, 8, 16, 32]
    config: [baseline]
  repeats: 3
```

`axes` is a cartesian product; `cells()` expands it. Each cell runs `repeats`
times. **`repeats: 3` is not decoration** — a single run gives you a number,
three give you a spread, and without a spread you cannot tell a real difference
from noise on a laptop GPU.

Workload profiles live in `config/workloads.yaml`. The rule that matters:
`requests` must exceed the top concurrency, or there is no queue to measure and
the latency numbers describe an idle engine.

**A concurrency point is `concurrency` virtual users, not one user with
`concurrency` requests.** The scheduler caps one user's share of the GPU
(`max_in_flight_per_user: 4`), so a batch that arrives as a single user
measures that cap instead of the concurrency: every point above 4 looks the
same, and the curve goes flat for a reason that has nothing to do with the
engine. The harness hands each in-flight slot its own `x-user-id`
(`bench-u0`, `bench-u1`, …) and hands it back when the request finishes, so
the run is `concurrency` independent closed-loop clients. It is a queue of
slots rather than a fixed pool of workers, so a slow request cannot idle its
slot while another prompt waits for it.

**Each measured run gets its own prompts.** `PromptFactory(salt=prompt_epoch)`
makes the unique part of every prompt different for every run, so the second
repeat of a cell is not measured against a prefix cache the first repeat
warmed. The `prompt_epoch` is recorded in the run row, so a run says which
prompts it used. The *shared* prefix is never salted — that is the point of
it: the shared prefix is what makes prefix caching observable, and salting it
would measure the wrong thing.

Engine configurations live in `config/engine/`. `baseline` is the control
condition (no prefix caching, eager execution). `optimized` turns on prefix
caching and chunked prefill. Both keep `enforce_eager: true` — on 6 GB the
memory runs out before CUDA-graph replay would pay for itself, and honest step
timings beat fast-but-unexplainable ones.

The scheduler drives one of these configs: `config/scheduler.yaml` names it
(`engine: optimized`), and reads the block size from the engine's own vLLM
config at startup so the two cannot disagree about it.

---

## Results, reports, cost

Every run lands in sqlite (`results/bench.db`) with its full provenance:
model, dtype, `max_model_len`, sampling, engine flags, vLLM version, GPU, git
SHA. A run whose environment you cannot reconstruct is not a result.

```bash
uv run --no-sync inferbench report --experiment my-experiment --out results/report.md
```

```bash
uv run --no-sync inferbench report --experiment conc_scale --max-ttft-p95 0.5
```

Ranks configurations by a normalised throughput / latency / cost score. The
ceiling filter runs first, so a fast configuration that breaches the
interactive target is excluded rather than recommended — the fastest
configuration is worthless if it is too slow to use. Runs with failed requests
are excluded for the same reason: a configuration that served nothing has no
throughput, latency or cost to compare, and letting it into the ranking is how
a broken setup gets crowned as the winner.

Cost comes from `config/pricing.yaml`. The default basis is a rental-equivalent
hourly rate, stated explicitly in the file, because the honest alternative
(amortised hardware) gives a number roughly 10x smaller and the two are easy to
confuse. `cost_per_successful_request` divides by *successful* requests, so a
configuration that fails half its traffic cannot look cheap.

---

## Observability stack and SLOs

```bash
uv run --no-sync inferbench up      # docker compose -f observability/compose.yaml up -d
uv run --no-sync inferbench down
```

Prometheus, Grafana (two provisioned dashboards: live and benchmark
comparison), Tempo for OTLP traces, and a pushgateway for benchmark results.
SLO targets are in `config/slo_targets.yaml` and are **currently placeholders,
marked as such in the file**. They are deliberately loose. The intended
workflow is to run the benchmarks first and then set each target from the
measured distribution — a target guessed before you have data is either so
loose it never fires or so tight the team stops believing it.

```bash
uv run --no-sync inferbench slo --experiment my-experiment
```

An objective nobody measured renders as `[SKIP]`, never `[PASS]`. Rendering
"unknown" as "passing" is exactly how a broken pipeline looks healthy.

`inferobs.burn` implements error-budget burn rate and time-to-exhaustion, which
is what the alert rules in `observability/alerts/slo_rules.yml` fire on.

---

## Watching the grid during a traffic spike

The stack above tells you *that* the grid misbehaved. This section is how you
find out *which tier* misbehaved: each of the three tiers publishes its own
state, and a traffic spike leaves a signature you can read straight off the
dashboard.

### Point the stack at the grid that is running

```bash
GATEWAY_PORT=9000 SCHEDULER_PORT=9001 ./scripts/observability.sh
# Prometheus http://localhost:9090    Grafana http://localhost:3000
```

The two ports are substituted into `observability/prometheus/prometheus.local.yml`
because Prometheus cannot expand environment variables inside its config — the
stack and the grid cannot drift apart by accident.

`observability/prometheus/prometheus.yml` scrapes every **1 second**, not the
usual 15. A spike here is absorbed by admission control in about five seconds,
so a 5-second scrape catches one sample of the whole event and the per-tier
shape is invisible. The counters are exact either way; only the shape needs the
resolution.

### The 21 panels of "Live — vLLM + Gateway", by tier

| Tier | Panels |
| --- | --- |
| Gateway | in-flight requests; rejected (rate); requests by backend and status; cache hits vs misses |
| Scheduler | requests in flight; hold queue depth; reservation drift (must be 0); preemptions (must stay 0); KV blocks total/free/reserved/allocated; dispatch decisions; hold time p95; dispatch latency p95; request outcomes by status |
| Engine | running requests; waiting requests; KV cache usage / prefix cache hit rate; scheduled tokens by phase; preemptions; iteration time p95 |
| GPU | utilisation / power / temperature; memory used ratio |

### Make a spike

```bash
node scripts/loadgen.mjs --url http://localhost:9000 \
  --users 24 --concurrency 10 --requests 40 \
  --prompt-tokens 2000 --max-tokens 32 \
  --concurrency-limit 4
```

24 users × 10 concurrent requests each × 40 requests = 960 requests. Every
prompt carries a fresh nonce, so the gateway's exact-match cache never answers
one — a cached answer would never touch the engine and the run would prove
nothing.

`--concurrency-limit 4` is explicit because the spike deliberately declares a
*smaller* limit than it uses: each user opens 10 requests but tells the
gateway it only intends 4 at a time, so the gateway's per-user cap is the
thing under test. It is the client lying about its own concurrency, and the
`user_concurrency` rejections in the table below are that lie being caught.
Leave the flag off and each user declares the concurrency it actually uses,
which is what you want unless you are deliberately probing the per-user cap.

**Drive the spike from inside the VM that runs the gateway.** The WSL2 image
has no `node`, so `scripts/loadgen.mjs` cannot run there; the shipped
equivalent is `scripts/verify_cascade.py`, which is standard-library Python and
therefore runs in WSL2, in a container, or on a laptop:

```bash
# inside the VM (WSL2 here), against the grid on the same host
/home/anupam/.venv/bin/python scripts/verify_cascade.py \
  --gateway http://127.0.0.1:9000 --scheduler http://127.0.0.1:9001
# use :9100 / :9101 if Docker already owns :9000 / :9001 on this machine
```

It drives the same spike (24 users × 10 × 40, 2000-token prompts, 32 output
tokens, `concurrency_limit: 4`), samples both tiers' `/metrics` while the
burst runs, prints the table and trace below, and then checks **11 identities**
for balance. Exit code 0 means every identity held; a non-zero exit names the
one that did not. Every number quoted in the next two sections comes from a
run of this script.

Running the load from the Windows host instead is *not* equivalent, and it is
not a client-side problem — see
[When the gateway goes dark in Prometheus](#when-the-gateway-goes-dark-in-prometheus).

### What the cascade looks like

Measured on that exact 960-request spike (24 users, 2000-token prompts, 32
output tokens, `concurrency_limit: 4` per request), driven from inside the VM
with `scripts/verify_cascade.py`. The burst took 16.0 s. Prometheus's own
stored history for the same window agrees with the numbers below to within
one scrape.

| Tier | Metric | Peak | Ceiling it hit |
| --- | --- | --- | --- |
| Gateway | `gate_in_flight` (running) | **32** | engine `max_num_seqs: 32` |
| Gateway | `gate_waiting` (held) | **64** | per-user limit `4 + 64` |
| Gateway | `gate_rejected_total{reason="user_concurrency"}` | +604 | per-user cap (`concurrency_limit: 4`) |
| Gateway | `gate_rejected_total{reason="global_limit"}` | +0 | global cap 288 — never reached |
| Gateway | `gate_held_total{reason="capacity"}` | +315 | the engine's 32 slots are full |
| Gateway | `gate_held_total{reason="user_reserved"}` | +9 | `max_reserved_blocks_per_user: 512` |
| Gateway | `gate_held_total{reason="kv_capacity"}` | +0 | the pool was never short |
| Gateway | `gate_admitted_total` | +32 | |
| Scheduler | `sched_hold_depth` | **64** | |
| Scheduler | `sched_reserved_blocks` | 875 peak (≈245 in steady state) of 3662 | |
| Scheduler | `sched_allocated_blocks` | 252 | blocks actually granted (deduplicated) |
| Scheduler | `sched_free_blocks` (low) | 3410 | |
| Scheduler | `sched_reservation_drift_blocks` | 0 | must stay 0 |
| Scheduler | `sched_preemption_total` / `engine_preemptions_total` | 0 | must stay 0 |
| Scheduler | `sched_capacity_breach_total` | 0 | must stay 0 |
| Engine | `engine_running_requests` | **32** | `max_num_seqs: 32` |
| Engine | `engine_waiting_requests` | **0** | vLLM's own queue never formed |
| Engine | `engine_kv_cache_usage_ratio` | 0.069 | `> 0.95` is the `KVCacheNearFull` alert |

Two of those ceilings are worth spelling out, because they are the difference
between "the gateway refused my traffic" and "the gateway refused my traffic
because it was told to":

- The **global** cap is `max_in_flight (32, learned from the engine's
  `max_num_seqs`) + max_queue_depth_global (256)` = **288**. The gateway takes
  it from the scheduler at startup (`_adopt_scheduler_limits`), so the two
  tiers cannot drift apart. 24 users × 4 = 96 can never reach 288, so
  `global_limit` stays at 0 and every 429 in this run is a *per-user* 429.
- The **per-user** cap is `max_in_flight_per_user (4) + max_queue_depth_per_user
  (64)` = **68** — but each request declares `concurrency_limit: 4`, so the
  effective cap is 4. `gate_waiting` peaking at 64 is the per-user queue
  behind that cap, not a global one.

**The hold reason is the load-bearing signal.** 315 of the 324 holds say
`capacity` — the engine's 32 slots are full — and not one says
`kv_capacity`. That is the difference between "the GPU is busy" and "the KV
pool is exhausted", and a dashboard that cannot tell them apart will send you
to the wrong tier. The pool was never short: 3410 of 3662 blocks were free
throughout, and `sched_reserved_blocks` settled at 245 against 252 actually
granted.

That last part deserves its own note, because the number looks wrong until
you know what the workload is. The prompts in this burst are all the same
2000 words with a different nonce at the end, so the engine's prefix cache
serves all 32 of them from one set of blocks. The pool is barely touched
(6.9 %) and the *engine's* limit is what binds. `scripts/verify_cascade.py
--distinct-prompts` breaks the shared prefix as well, which is the case that
actually exercises the pool:

| Workload | KV usage | hold reason | preemptions |
| --- | --- | --- | --- |
| shared prefix (default) | 0.069 | `capacity` (315) | 0 |
| `--distinct-prompts` | **0.955** | `kv_capacity` (337) | 0 |

Same grid, same burst, same 960 requests. The distinct run drives the pool to
95.5 % with 163 blocks free, the hold reason flips to `kv_capacity`, and the
scheduler admits 32 instead of 25 — and **still** nothing is preempted,
because the scheduler holds the request that would not fit instead of letting
the engine start it and take it back. The GPU sits at 96–100 % utilisation
throughout. That is the whole design working: the pool runs nearly full, the
GPU runs at full tilt, and no request is ever killed.

The one-second shape of that spike — this is the panel row to screenshot when a
break happens (t = seconds after the burst starts):

```
t  gate_in gate_wait hold resv alloc free eng_run eng_wait kv
 0       0         1    0    0     0 3662        0        0 0.0000
 1      32        64   64  232   200 3462       32        0 0.0546
 2      32        64   64  244   212 3450       32        0 0.0579
 3      32        64   64  246   214 3448       32        0 0.0584
 4      32        56   55  625   180 3482       29        0 0.0492
 5      32        50   47  623   178 3484       29        0 0.0486
 6      32        50   50  245   245 3417       32        0 0.0669
 7      32        50   50  242   236 3426       32        0 0.0644
 8      32        50   50  252   220 3442       32        0 0.0601
 9      32        50   50  252   252 3410       32        0 0.0688
10      32        50   50  248   220 3442       32        0 0.0601
11      32        41   41  243   211 3451       32        0 0.0576
12      32        31   28  622   177 3485       29        0 0.0483
13      32        31   31  244   244 3418       32        0 0.0666
14      32         4    4  235   235 3427       32        0 0.0642
15       4         0    0  141   141 3521        4        0 0.0385
16       0         0    0    0     0 3662        0        0 0.0000
```

The `resv` column is the ledger's **commitment**, not a sum of worst cases:
the blocks the engine has granted (as a union, so the shared prefix counts
once) plus what the live requests may still add. That is why it reads 252
against `alloc` 252 while the pool holds 3410 free blocks — 32 requests
sharing one 2000-token prefix cost 125 blocks between them, not 32 × 129.

Latency over the same window, from Prometheus's histograms:
`gate_request_duration_seconds` p50 3.56 s / p95 4.93 s / p99 8.02 s,
`sched_hold_seconds` p50 1.94 s / p95 4.47 s, and
`sched_dispatch_latency_seconds` p50 0.50 ms / p95 0.95 ms. The
`GatewaySaturated` alert fired; `KVCacheNearFull` did not.

`gate_ttft_seconds` and `gate_tpot_seconds` are **empty** for this run, and
that is not a gap in the instrumentation: TTFT and TPOT are observed only on
the streaming path, and this load generator posts non-streaming JSON. The
per-request client-side lines carry the same information:

```
status=200 decision=admit reason=fit          hold=0.00s reserved=129 demand=129 1802ms
status=200 decision=hold  reason=user_reserved hold=1.74s reserved=0   demand=129 3076ms
status=429 decision=-     reason=-             hold=0.00s reserved=-   demand=-      6ms
```

`reserved=129 demand=129` on the admitted line is the ledger's promise
matching what the engine actually needed. `reserved=0 demand=129` on the held
line is the same request *before* it was granted: the demand was computed and
carried, but nothing was reserved yet. A refused request never reaches the
scheduler, so it has neither — and note it also has no
`gate_request_duration_seconds` sample, because the gateway answers a 429
before it ever reaches the timer. The refusal is decided instantly and costs
the server nothing; what it costs is the client, which waited 6 ms to be told
no.

**The story in one line: the gateway sheds, the scheduler queues, the engine
never queues.** 604 of 960 requests were refused at the door (429), 324 were
parked in the scheduler's hold queue and released within ~4.5 s, and 32 ran on
the GPU concurrently with vLLM's internal queue empty — which is the entire
point of having a scheduler tier in front of the engine.

### Throughput, measured properly

The burst above answers *"how many requests got through the door"*, which
mixes admission with throughput and is not a steady-state number — the offered
load decays as each user's 40 requests run out. To measure the engine, hold a
fixed load open and count what completes: 24 users × 4 workers (the most the
gateway will hold per user at `concurrency_limit: 4`) for 25 s.

| Ledger | Engine concurrency | Throughput | Latency p50 |
| --- | --- | --- | --- |
| promise = sum of worst cases | 28 of 32 slots | 25.12 req/s (804 tok/s) | 4.21 s |
| promise = granted + outstanding | **32 of 32 slots** | **26.68 / 27.04 req/s (854–865 tok/s)** | 4.17 / 3.95 s |

Two consecutive runs each, repeatable to three decimals, on the same grid.
The second row is ~6.5 % more throughput at a slightly *lower* p50 — the
engine's 32 slots are all doing work instead of four of them idling behind a
promise the pool could have honoured.

### The ledger: identities that must balance

These hold at every instant, and they are how you localise a break to one tier.
Counters are read from the gateway's own `http://localhost:9000/metrics`, not
from Prometheus, when you need them exact — a Prometheus snapshot can be one
scrape (one second) stale.

```
gate_in_flight                       ≈ sched_user_inflight  ≈ engine_running_requests
gate_waiting                         = sched_hold_depth
gate_cache_misses_total              = rejected + ok
gate_cache_hits_total                = 0   (only when every prompt is unique)
gate_requests_total{status="ok"}     = admitted + Σ held
gate_requests_total{status="rejected"} = Σ gate_rejected_total{reason}
sched_dispatch_total{decision="admit"} = gate_admitted_total
sched_dispatch_total{decision="hold"}  = Σ gate_held_total{reason}
```

The three gauge rows are `≈`, not `=`, and the reason is worth knowing. The
gateway is a single asyncio event loop, and under a 240-way fan-in it is
also answering hundreds of 429s while the scheduler's completion frames queue
up unread behind them. `gate_in_flight` can therefore read a few *above*
`sched_user_inflight` for a moment — the scheduler has already finished those
requests, the gateway has not yet processed their frames. Both gauges return
to zero when the traffic stops, so a permanent gap is a bug and a momentary
one is not. The *counter* identities below are exact, because a counter does
not care when you read it.

`gate_in_flight` and `gate_waiting` are **separate** gauges: the first counts
requests the engine is *running*, the second counts requests the scheduler is
*holding*. Before that split, one gauge carried both and
`gate_in_flight = sched_hold_depth + engine_running_requests` held. It does
not any more, and a dashboard still asserting the old identity will show a
permanent "missing" bar.

On the measured spike:

```
960 requests = 604 rejected + 356 ok
356 ok       =  32 admitted + 324 held
324 held     = 315 capacity + 9 user_reserved
gate_in_flight 32 = sched_user_inflight 32 = engine_running_requests 32
gate_waiting   64 = sched_hold_depth 64
```

Three of those look wrong until you read the code, and all three are correct:

- `gate_requests_total{status="ok"}` (356) is **larger** than
  `gate_admitted_total` (32). A held request is counted as held *and*, when it
  finally completes, as an `ok` request — the gateway only ever sees a
  finished response. `ok = admitted + held`, always. A cache replay counts
  here too, under `backend="cache"` rather than `backend="scheduler"`; the
  `gate_cache_hits_total` guard above is what tells the two apart.
- `gate_cache_misses_total` (960) equals `ok + rejected`, not
  `admitted + held`. A miss is counted before admission, so a request that is
  later refused is still a miss.
- `sched_reserved_blocks` and `sched_allocated_blocks` are two different
  questions. `allocated` is the *union* of the block ids the engine actually
  granted — a prefix-cache hit that shares blocks counts once. `reserved` is
  the ledger's **commitment**: those same granted blocks, plus what the live
  requests may still add. So `reserved ≥ allocated` always, and the gap is
  the outstanding promise. It used to be a sum of worst cases — 32 requests
  × 129 blocks = 4128 — which is why the old dashboard read "98 % of the pool
  promised" while 3410 blocks sat free and four engine slots idled. The
  self-check is `sched_reservation_drift_blocks` — what the engine holds minus
  what the ledger accounts for — and it stays at 0.

`scripts/verify_cascade.py` asserts all eleven of these against a live grid
and exits non-zero if any one of them fails.

### Symptom → metric

| What you see | Read this | It means |
| --- | --- | --- |
| Clients get 429s | `gate_rejected_total{reason="user_concurrency"}` | one user hit its own ceiling (`concurrency_limit`, capped by `max_in_flight_per_user`) |
| Clients get 429s | `gate_rejected_total{reason="global_limit"}` | the gateway's global cap is full — `max_in_flight` (32, learned from the engine) + `max_queue_depth_global` (256) = **288** — add capacity or shed deliberately |
| Requests are slow, not refused | `sched_hold_depth`, then `gate_held_total{reason}` | `capacity` = scheduler in-flight cap; `kv_capacity` = not enough free KV blocks; `physical_capacity` = the pool is empty even though the ledger shows room; `user_reserved` = that user's reservation cap |
| Everything is slow but nothing is queued | `gate_ttft_seconds` p95, `gate_tpot_seconds` p95 | the engine itself got slower — these are measured live, accept to first token and per-token after it, so they separate "waiting" from "generating" |
| The queue is deep | `gate_waiting` (held) vs `gate_in_flight` (running) | `gate_in_flight` is only what the engine is *running*; `gate_waiting` is what the scheduler is holding. Together they are the old combined in-flight number |
| Holds grow and never drain | `sched_hold_seconds` p95 rising, `sched_dispatch_total{decision="hold"}` flat | the engine is not releasing slots — check `engine_running_requests` against `sched_user_inflight` |
| Engine-side latency | `sched_dispatch_latency_seconds` | slot-free → engine-start overhead; this is the scheduler's own cost, normally sub-millisecond |
| vLLM is queueing | `engine_waiting_requests > 0` | the engine has requests it cannot start yet. Two different causes, and the KV ratio tells them apart: **KV high** (near 0.95) = the pool is the constraint, which is correct behaviour; **KV low** = the engine's own token budget (`max_num_batched_tokens`) is smaller than a prompt, so it can only prefill one at a time — see [The engine's own queue](#the-engines-own-queue) |
| GPU is hot or slow | `engine_kv_cache_usage_ratio`, `engine_iteration_seconds` p95 | KV pressure; `KVCacheNearFull` fires at 0.95 |
| Blocks leak | `sched_reservation_drift_blocks > 0` | the engine holds blocks the ledger never accounted for — a reservation is not being released |

### The engine's own queue

The scheduler tier exists so that **vLLM never has to queue** — the tier's
queue is the one queue, and it is a visible one with a reason code on every
request. But vLLM keeps a queue of its own, and it is worth knowing when it
forms, because two different things put requests in it and they need different
fixes.

`engine_waiting_requests` is vLLM's `num_waiting_reqs`: requests the engine
has accepted but not yet scheduled. A sample from the same grid, same burst,
same code — only the prompt size differs:

| workload | KV usage | `engine_waiting_requests` peak | cause |
| --- | --- | --- | --- |
| prompts 100–300 words | 0.235 | **0** | the pool was never the constraint |
| prompts 200–2000 words | 0.907 | **18** | the pool was nearly full |

At 90 % KV the engine cannot start more requests because there are no blocks
for them — and that is the *scheduler tier working*: the requests it could not
place are the ones it held (`kv_capacity`, `capacity`), and the ones it did
place are waiting for blocks that are about to be freed. The KV ratio
distinguishes the two cases, and that is why the row in the table above reads
"KV high = correct behaviour".

The other cause is the engine's own **token budget**. vLLM's default
`max_num_batched_tokens` is 2048, which is smaller than one long prompt: a
1300-token prompt nearly fills it, so only one request can prefill per step
and the rest of the batch waits *inside* the engine, invisible to the tier.
`config/engine/optimized.yaml` now sets it to 8192 under `extra:`, and the
engine's queue stopped forming on the same workload.

The consequence of getting this wrong is exactly what the tier exists to
prevent: **two queues, and the engine's is the one that decides the order** —
so round-robin fairness across users and the per-user hold reason codes stop
meaning anything.

| The config says one thing and the engine does another | the key is missing from the engine's kwargs | a key the config model does not name is dropped by pydantic; it must be nested under `extra:` |
### Preemption

Preemption is vLLM evicting a running request to make room for another: it
frees the victim's KV blocks, resets the victim's progress to zero, and puts
it back in the waiting queue. The client sees nothing but latency — the
tokens are recomputed from scratch — so it is the worst outcome available,
and the whole scheduler tier exists to make it unnecessary.

**It is a backstop, and in every run so far it has not fired.** Measured
across the shared-prefix burst, the `--distinct-prompts` burst that drives
the pool to 95.5 %, and a `reservation_mode: prompt_only` run where the
promise deliberately under-reserves:

| Run | KV usage | `engine_preemptions_total` | `sched_preemption_total` | `sched_capacity_breach_total` |
| --- | --- | --- | --- | --- |
| shared prefix | 0.069 | 0 | 0 | 0 |
| `--distinct-prompts` | 0.955 | 0 | 0 | 0 |
| `prompt_only` | 0.884 | 0 | 0 | 0 |

That is not luck, and it is not the engine being gentle. Two things prevent
it, in order:

1. **The reservation.** Under `strict` the ledger promises each request its
   prompt blocks *plus* its worst-case output, so the sum of the promises is
   known to fit in the pool before anything starts. The engine is never asked
   to run more than the cache can hold.
2. **The physical backstop.** Even when the promise is wrong — a prefix-cache
   hit, a `prompt_only` promise, a request that grew — admission checks the
   *actual* free blocks against what the running requests still need, and
   holds instead of admitting. That is the `physical_capacity` hold reason,
   and it is the check that stops the engine's `allocate_slots` from ever
   returning `None`.

When both are in place the engine has no reason to preempt, and it does not.

**If it does happen, it is turned into a visible, retryable failure.** The
scheduler watches the engine's own preemption count, picks the running
request that matters least — the largest priority number, then the latest
arrival, the same order vLLM uses — aborts it, and counts a
`sched_capacity_breach_total`. The client gets a **503** with
`retry-after: 1` and a `capacity_breach` reason, not a truncated answer. The
pool gets the blocks back immediately instead of waiting for the victim's
tokens to finish. `tests/test_dispatcher.py::test_a_preemption_sheds_the_least_important_request`
and `tests/test_gateway_scheduler.py` cover that path end to end.

So the answer to "does the system preempt requests so the others can work?" is
**no** — and the two mechanisms above are the reason. The requests that do not
fit are *held*, they wait their turn, and they complete normally.


| The scheduler stopped answering | `up{job="scheduler"} == 0`, `SchedulerDown` | the tier is gone; `SchedulerDispatchStalled` covers the case where it is up but not dispatching |
| A tier goes dark in Prometheus while it is still serving clients | `up{job="gateway"} == 0` for ~15 s, `scrape_duration_seconds` pinned at exactly the scrape timeout | the scrape path is congested, not the tier — see [When the gateway goes dark in Prometheus](#when-the-gateway-goes-dark-in-prometheus) |
| Numbers do not add up | the identities above | the gap names the tier that broke |

### When the gateway goes dark in Prometheus

During the peak of a burst driven from the Windows host, the gateway tier
disappears from Prometheus for 14–16 s at a stretch: `up{job="gateway"}` = 0
for 8 consecutive two-second samples, and `scrape_duration_seconds` pinned at
exactly `1.00` — the scrape timeout, not the time the gateway took to answer.
The tier keeps serving clients the whole time. This reads like the gateway
crashed, and it is worth knowing that it did not.

**The gateway application is exonerated.** The `/metrics` handler is a trivial
async function; `admission.py` takes no locks; `registry.py`'s lock is never
held across an `await`; and the TCP handshake to the gateway completes in
0–3 ms. With 240 idle sockets open, a fresh `GET /metrics` still returns in
33 ms. Nothing in the request path blocks long enough to starve a scrape.

**The cause is the Windows → WSL2 boundary.** Windows reaches WSL2's ports
through a "localhost forwarding" relay, and a host-driven burst fans 240
concurrent connections at that relay. Bypassing the relay does not help:
driving the same burst from Windows at WSL2's vSwitch IP
(`172.24.216.39:9100`) still produced 15 s of `up=0`. The differentiator is
*where the load client runs*, not the IP it addresses. The scheduler goes dark
in the same windows where its own port is hammered (12 down-windows against
the gateway's 13), which names the shared Windows↔WSL2 path rather than either
tier's code.

**The remedy is to drive the load from inside the VM.** The identical
960-request, 240-connection burst driven from inside WSL2 leaves
`up{job="gateway"}` = 1 at every scrape, with `scrape_duration_seconds`
≤ 0.15 s. Prometheus's own stored history for 21:06:41–21:12:41 UTC shows the
gateway up at every sample and the scheduler up at every sample. Since WSL2
has no `node`, that is what `scripts/verify_cascade.py` is for: it is
standard-library Python, so it runs in the VM where `scripts/loadgen.mjs`
cannot.

**The shipped Prometheus config is deliberately unchanged.** A 14–16 s hole
exceeds any timeout that could be configured: `promtool` rejects a
`scrape_timeout` greater than its job's `scrape_interval`, and the shipped
config scrapes every **1 second** globally, so no legal timeout covers the
hole. Widening the interval to 5 s would close the window at the cost of the
1 s resolution the per-tier shape depends on. The gateway answers `/metrics`
in ≤ 150 ms whenever the transport is not congested — the only thing to fix is
where the load comes from.

---

## Kubernetes

`deploy/k8s/` holds a kustomize overlay: namespace, scheduler deployment,
gateway deployment, vLLM deployment, `ServiceMonitor`, `PrometheusRule` and the
model-cache PVC. It targets a real cluster with a real GPU node — this laptop
cannot run it, and it says so in its `README.md`. It is here because the
manifests are the artefact a production deployment needs, and they are written
against the same metric names as everything else.

Two rules are load-bearing there:

- **The scheduler owns the GPU, so `replicas: 1` and `strategy: Recreate`.** Two
  replicas would be two processes each holding 6 GB of weights on one card, and
  a rolling update would briefly be both of them. The scheduler has no HPA.
  The gateway has one, pinned to `maxReplicas: 1` — its cache *and* its
  idempotency registry are both in-process, so a second pod would let the same
  `Idempotency-Key` start a second engine request.
- **Readiness is `/ready`, liveness is `/health`.** The scheduler process is alive
  long before the weights have loaded; a readiness probe that passed on
  liveness would route traffic to a scheduler that can only reject it.

---

## Tests

```bash
uv run --no-sync pytest             # 247 tests, no GPU and no engine needed
uv run --no-sync ruff check .
```

The suite runs on CPU with a 1-token-per-character fixture tokenizer, so it
needs neither a GPU nor vLLM installed. What it covers:

- `test_metrics.py` — TTFT/ITL/TPOT/e2e derivation, checked against
  `numpy.percentile` for exactness, including the multi-token-chunk case where
  one SSE chunk carries several tokens and the interval has to be split.
- `test_gateway.py` — routing, admission, cache key correctness and LRU/TTL,
  and both proxy paths end to end over `httpx.ASGITransport` with a mocked
  upstream. Includes the streaming admission release, because Starlette drains
  a `StreamingResponse` generator *after* the handler returns and a release
  written in the wrong place leaks the budget silently.
- `test_ledger.py`, `test_admission.py`, `test_dispatcher.py` — the reservation
  ledger, the FIT/HOLD/REJECT decision, and the round-robin dispatch. The
  ledger's own test is the one that matters: a request that has been handed
  its blocks stops promising them, so a prefix-shared pool is not counted
  once per request. The single-writer invariant gets its own test: a fake
  engine whose free-block count is authoritative, two concurrent dispatch
  attempts, and an assertion that the second one is refused.
- `test_registry.py`, `test_scheduler_api.py`, `test_gateway_scheduler.py` —
  the per-user registry, the internal API, and the gateway↔scheduler contract:
  forward, propagate the decision headers, **wait on a hold and 429 only on a
  reject**, abort on client disconnect, and suppress idempotent duplicates.
- `test_alerts.py` — the two rule stacks carry identical alerts and recording
  rules, and no rule or dashboard panel queries a metric nothing emits.
- `test_store.py`, `test_report.py`, `test_slo.py` — the record of truth, the
  views built on it, and the SLO arithmetic.

The GPU path (`inspect`, `serve`, `bench`) is the part tests cannot cover; it
needs the real card. `bootstrap_wsl.sh` ends with a check that prints your
actual device, VRAM and compute capability, which is the fastest way to find
out that you are on a different GPU than you assumed.

---

## Deliberate simplifications

Each of these is a real ceiling with a known upgrade path, not an oversight:

- **A typo'd config key is now a startup failure, not a silent default.**
  Pydantic drops unknown fields by default, so `max_num_batched_tokens: 8192`
  at the top level of an engine config was dropped without a word and the
  experiment ran with vLLM's default 2048 instead. Every config model now
  sets `extra="forbid"`, so the key must go under `extra:` — which is the
  escape hatch that is merged into the engine's kwargs verbatim. The
  alternative was a config file that lies about what the engine is doing.
- **The scheduler is a single point of failure and owns the GPU.** One GPU, one
  engine — there is nothing to shard yet. Multiple GPUs → one scheduler replica
  per GPU, gateway routes by user affinity.
- **The gateway's cache and idempotency registry are in-process, per replica.**
  A second pod would have its own copy: the same prompt could miss on one pod
  and hit on another (cache-hit metrics become meaningless), and the same
  `Idempotency-Key` could reach two pods and start two engine requests. The
  HPA is pinned to one replica until both move to Redis — nothing else here
  needs it.
- **Per-user state is not durable across a gateway restart.** Idempotency keys
  make resubmission safe; the scheduler is the source of truth for capacity.
  Move the registry to Redis when it must survive restarts.
- **SLO targets are placeholders.** Replace them with numbers derived from your
  own measured runs.
- **The K8s overlay assumes a GPU node.** Untested here by necessity.
- **`inspect` needs the engine core in-process.** It forces
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`, which costs throughput against V1's
  default child-process core. That trade is the price of reading the scheduler.
