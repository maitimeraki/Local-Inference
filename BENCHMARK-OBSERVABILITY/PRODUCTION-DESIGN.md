# Production Request-Handling Design

**Status:** design only — no code written yet.
**Scope:** the persistent, multi-tenant request path that replaces the current one-shot "paste → Enter → model loads → prompt runs → terminal dies" loop.
**Builds on:** this repo, unchanged in intent. Roughly 80% of the hard parts already exist (the in-process tap, the metric vocabulary, the gateway's cache and admission, the SSE client). What is missing is the **scheduler tier**: the component that owns KV-block budget, holds requests that do not fit, and recycles batch slots.

---

## 1. The change

Today the engine is a **process lifetime**: `inferbench serve` loads the model, serves, and dies with the terminal. Every model load costs ~30–60 s and the GPU is idle whenever the process is gone. Nothing survives, so nothing can be queued.

Production is the inverse:

| | Today | Production |
|---|---|---|
| Model | loaded per run, then discarded | **resident on the GPU** |
| Requests | one at a time | many, concurrently, from many users |
| Identity | none | every request carries `user_id` + `request_id` |
| Overload | the engine dies | the queue holds, and the shed is visible |
| Backpressure | none | KV-block admission with reservation |

The user's stated shape: *gateway → scheduler → engine*. The gateway sees arrivals and users; the scheduler owns capacity and holds what does not fit; the engine executes. When a request finishes, its batch slot frees and the next held request takes it.

---

## 2. Goals and non-goals

**Goals**

1. The model stays resident on the GPU; the scheduler never unloads it.
2. Many concurrent requests, tracked per user, with no user able to monopolise the batch.
3. A request whose KV demand does not fit is **held** on the scheduler side, not rejected and not silently queued inside the engine where it cannot be observed.
4. When a request finishes, its slot is handed to the next held request promptly and without polling latency.
5. Every admission decision is observable: admit / hold / reject, with a reason, and the time spent held.
6. All of the above is verifiable with the existing GPU-free test approach.

**Non-goals**

- Multi-node / multi-GPU. One node, one GPU, one scheduler. (§15 lists what changes if that moves.)
- Semantic caching. The exact-match cache in the gateway is enough; a semantic cache is a different project.
- Model serving of anything other than completions.
- Changing vLLM's internal batching. The engine keeps doing what it does well; the scheduler governs **what reaches it**.

---

## 3. Topology

```
                    ┌──────────────────────────────────────────────────┐
   client           │  GATEWAY — infergw (stateless, 1 replica)       │
   ───────────────▶ │                                                  │
   X-User-Id        │  • identity + idempotency                        │
   X-Request-Id     │  • exact-match cache (hit ⇒ never reaches tier 2)│
   Idempotency-Key  │  • per-user registry (in-flight, queue depth)    │
                    │  • global admission (max_in_flight)              │
                    │  • queue-time measurement                        │
                    └───────────────────────┬──────────────────────────┘
                                            │ POST /internal/v1/submit
                                            │   (streamed response back)
                                            ▼
                    ┌──────────────────────────────────────────────────┐
                    │  SCHEDULER — infersched (single replica)        │
                    │                                                  │
                    │  • owns the vLLM engine **in-process**           │
                    │  • KV reservation ledger (the authority)          │
                    │  • FIT / HOLD / REJECT admission                 │
                    │  • per-user FIFO + round-robin dispatch          │
                    │  • single dispatcher coroutine (the only writer) │
                    │  • reconcile ledger against engine truth         │
                    └───────────────────────┬──────────────────────────┘
                                            │ in-process (no HTTP)
                                            ▼
                    ┌──────────────────────────────────────────────────┐
                    │  ENGINE — vLLM V1, engine core in-process         │
                    │  • continuous batching, prefix cache, preemption │
                    │  • scheduler readable as a plain Python object   │
                    └──────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────┐
                    │  OBSERVABILITY — Prometheus / Grafana / Tempo     │
                    │  • gate_* (gateway)  sched_* (scheduler)          │
                    │  • engine_* (tap)    bench_* (benchmark runs)     │
                    └──────────────────────────────────────────────────┘
```

### The tier grid

| Tier | Package | Replicas | Owns | Scaled by | Stateless? |
|---|---|---|---|---|---|
| 1. Gateway | `infergw` | 1 (see §12) | arrivals, users, cache, admission | traffic | mostly — cache is per-process |
| 2. Scheduler | `infersched` *(new)* | 1 | KV ledger, hold queue, dispatch | **nothing — owns the GPU** | no |
| 3. Engine | vLLM V1 | 1 (inside tier 2) | KV blocks, batching, weights | GPU memory | no |
| 4. Observability | compose stack | — | metrics, dashboards, alerts | — | yes |

**Why the scheduler owns the engine rather than talking to it over HTTP.** The scheduler's decisions must be based on the engine's *real* free-block count. If the scheduler were a separate process it would either duplicate vLLM's block accounting (and drift, and cause preemption anyway) or have to poll the engine for the number (and add staleness to every admission decision). Hosting the engine in-process makes the scheduler's view exact by construction — the same reasoning that already produced `inferbench.inspect`.

This is why tier 2 is a single replica and cannot be scaled horizontally on one node: it *is* the GPU host.

**The engine core must run in-process, and that rules out the async serving path.** Verified against the installed vLLM 0.31.x source:

- `v1/engine/core_client.py:95-133` — `EngineCoreClient.make_client` raises `NotImplementedError("Running EngineCore in asyncio without multiprocessing is not currently supported.")` for the async-without-multiprocessing combination. There is **no** async in-process engine core in 0.31.x.
- `v1/engine/async_llm.py:180` — `AsyncLLM` calls `EngineCoreClient.make_async_mp_client(...)` unconditionally. `AsyncLLMEngine` therefore *always* runs the core in a child process, whatever `VLLM_ENABLE_V1_MULTIPROCESSING` says.
- `v1/engine/llm_engine.py:155-195` — `LLMEngine.from_engine_args` honours `VLLM_ENABLE_V1_MULTIPROCESSING=0`, giving an `InprocClient` whose `.engine_core` is the `EngineCore` that owns the scheduler.

So **the scheduler tier drives the sync `LLMEngine`, not `AsyncLLMEngine`**, and runs its own step loop. That is not a preference: it is the only configuration in which `block_pool.get_num_free_blocks()` and `req_to_blocks` are reachable at all.

What the async path would have cost, since it is the obvious choice to reach for:

| Needed by the ledger | Available in-process? | Available across the process boundary? |
|---|---|---|
| free block count | `block_pool.get_num_free_blocks()` | no — `SchedulerStats` carries only `kv_cache_usage` (a ratio) |
| total block count | `block_pool.num_gpu_blocks` | no — `cache_config.num_gpu_blocks` is set in the child |
| per-request blocks | `coordinator.*.req_to_blocks[request_id]` | no |
| preemption count | `SchedulerStats.kv_cache_eviction_events` | yes |

`collective_rpc` is not a way around it: `EngineCore.collective_rpc` forwards to `model_executor.collective_rpc`, which reaches the **executor workers**, not the scheduler.

The step loop is straightforward, because V1's `EngineCore.step()` returns `({}, False)` immediately when there is nothing to do rather than spinning — `InprocClient.get_output()` calls `step_fn`, and `core.py:236` assigns `step_fn = step` when there is no batch queue. The scheduler owns a background thread that steps the engine and drains a work queue; HTTP handlers submit into that queue and await a future.


---

## 4. Request lifecycle

```
   client
     │
     ▼
 ┌──────────┐   hit     ┌────────────────────────────┐
 │ GATEWAY  │──────────▶│ serve from cache           │──▶ 200
 └────┬─────┘           └────────────────────────────┘
      │ miss
      ▼
 ┌──────────┐  per-user limit / global limit / queue full
 │ GATEWAY  │──────────────────────────────────────────▶ 429 (Retry-After)
 └────┬─────┘
      │ admit to scheduler
      ▼
 ┌──────────┐   doesn't fit KV
 │SCHEDULER │──────────────────────────────┐
 │          │                              │  HOLD (queued, observable)
 │          │◀─────────────────────────────┘
 │          │   slot freed
 │  FIT ✓   │
 └────┬─────┘
      │ dispatch
      ▼
 ┌──────────┐
 │  ENGINE  │──▶ streaming tokens ──▶ gateway ──▶ client
 └────┬─────┘
      │ finish / abort / disconnect
      ▼
 ┌─────────────────────────────────────────────────────────┐
 │ release reservation · decrement in-flight · signal the  │
 │ dispatcher → next held request is dispatched            │
 └─────────────────────────────────────────────────────────┘
```

### State machine

| State | Owner | Entered when | Exits to |
|---|---|---|---|
| `RECEIVED` | gateway | request accepted by HTTP | `CACHED` / `REJECTED` / `ADMITTED` |
| `CACHED` | gateway | exact-match hit | terminal (`200`) |
| `REJECTED` | gateway | per-user or global limit hit | terminal (`429`) |
| `ADMITTED` | scheduler | block reservation succeeded | `RUNNING` |
| `HELD` | scheduler | reservation failed | `ADMITTED` |
| `RUNNING` | engine | dispatched to `engine.generate()` | `DONE` / `FAILED` / `ABORTED` |
| `DONE` / `FAILED` / `ABORTED` | scheduler | generator ended, error, or client disconnect | terminal |

Every transition is recorded once, as a **request record**, and carries the reason. The reason set is closed and finite — it becomes metric label values, so it must be small: `cache_hit`, `admitted`, `kv_insufficient`, `user_concurrency`, `user_queue_full`, `global_limit`, `global_queue_full`, `engine_error`, `client_abort`, `timeout`.

**Three vocabularies, one mapping.** The state names above, the decision returned by `decide()` (§6.4) and the token in the `X-Scheduler-Decision` header (§12.1) are different words for the same thing and must not drift apart:

| State (§4) | Decision (§6.4) | Header / label (§12.1, §11.2) |
|---|---|---|
| `CACHED` | — | `cache` |
| `ADMITTED` | `FIT` | `admit` |
| `HELD` | `HOLD` | `hold` |
| `REJECTED` | `REJECT` | `reject` |

`FIT`/`HOLD`/`REJECT` are the decision function's return values; `admit`/`hold`/`reject` are the wire and metric tokens. The header and the metric label must use the lowercase wire token — a metric label that changes case between runs breaks every query written against it.

---

## 5. Data model

### 5.1 Request envelope

Every tier passes one envelope. It is the unit of accounting, of tracing, and of the metrics.

```
RequestEnvelope
├── request_id      uuid4, unique. Generated at the gateway if absent.
├── user_id         str, from X-User-Id. Required.
├── idempotency_key str|None, from Idempotency-Key.
├── prompt          str
├── params          SamplingParams-equivalent (max_tokens, temperature, seed, …)
├── max_concurrency int|None, requested per-user concurrency (see §8.2)
├── submitted_ts    monotonic, at the gateway
├── deadline_ts     submitted_ts + client timeout
└── trace_id        str, propagated
```

`request_id` doubles as the vLLM `request_id` passed to `engine.generate(prompt, params, request_id)`. That is what makes the engine's request traceable back to a user, and it is the key the scheduler uses to look up the reservation.

### 5.2 User registry (tier 1)

```
UserRegistry
└── user_id → UserState
      ├── inflight: set[request_id]        # admitted, not yet finished
      ├── queued:   deque[request_id]      # rejected at the gateway, waiting for a per-user slot
      ├── inflight_limit: int              # from the user's entitlement, clamped by the server ceiling
      └── limit_updated_ts: float
```

The registry holds **only** what tier 1 can enforce without tier 2's help: how many of this user's requests are outstanding, and whether another one may start. It does **not** know about KV blocks — that is tier 2's job, and duplicating it in two tiers is how the two drift.

### 5.3 Reservation ledger (tier 2)

The ledger is the scheduler's authoritative view of KV capacity. It is the thing that must never drift.

```
ReservationLedger
├── total_blocks: int                    # from block_pool.num_gpu_blocks
├── free_blocks: int                     # from block_pool.get_num_free_blocks()
├── reservations: dict[request_id → Reservation]
├── reserved_blocks: int                 # sum of all live reservations
├── by_user: dict[user_id → int]         # reserved blocks per user
└── block_size: int                      # vllm_config.cache_config.block_size
```

```
Reservation
├── request_id, user_id
├── reserved_blocks: int                 # blocks we hold for this request
├── actual_blocks: int|None              # last observed from the engine
├── admitted_ts: float
└── state: RESERVED | RUNNING | RELEASED
```

**Two counters, not one.** `free_blocks` is the engine's truth, read fresh. `reserved_blocks` is our ledger's truth. The gap between them is the drift we must detect. The invariant is:

```
free_blocks (engine) + reserved_blocks (ledger) <= total_blocks
```

When that fails, the ledger has over-promised and the engine will preempt. Reconciliation (§9.3) is what restores it.

---

## 6. Admission: FIT / HOLD / REJECT

This is the core algorithm. It is a **pure function** of `(request, ledger, registry)` so it can be unit-tested with no engine and no GPU — the same trick `inferbench.inspect.format_step` uses.

```
def decide(req, ledger, registry, cfg) -> Decision
```

### 6.1 Block demand

```
block_size      = engine.vllm_config.cache_config.block_size   # read it, do not hard-code
prompt_blocks   = ceil(prompt_tokens / block_size)
output_blocks   = ceil(max_tokens   / block_size)
```

`block_size` is read from the engine's live config. vLLM V1's default is 16 (`CacheConfig.DEFAULT_BLOCK_SIZE`, verified against the installed 0.31.x), but the repo's existing code deliberately probes attributes defensively because they move between releases; hard-coding 16 would make the ledger wrong the day that changes, and the failure mode is silent preemption. Note that `CacheConfig.block_size` is declared as `default=None` and resolved by a model validator, so a config object read *before* validation has `block_size is None` — read it from the built engine's `vllm_config`, never from the raw YAML.

### 6.2 What vLLM already does — and what it cannot do

This matters more than anything else in this document, because it decides whether the scheduler tier is worth building at all.

**vLLM V1 already reserves KV blocks and already parks requests that do not fit.** Verified in the installed source:

- `v1/core/sched/scheduler.py:1194-1219` computes `reserved_blocks = self._inflight_prefill_reserved_blocks()` and passes it into allocation.
- `v1/core/kv_cache_manager.py:572-575` — the admission check is `available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks`; if the request needs more than that, `allocate_slots` returns `None` and **the request stays in the scheduler's `waiting` queue**.
- `v1/core/kv_cache_manager.py:535` — a second, harsher check: if the *entire* sequence will not fit, the request is not scheduled at all.
- A `watermark` (`scheduler_config.watermark`) holds back extra headroom for waiting and preempted admissions.

So the literal mechanism the user described — *"if the scheduler's cache is insufficient, the request should be held on the scheduler side; when one request finishes, its batch slot becomes available for the next request"* — **already exists inside the engine**, and it is exactly the `waiting` queue visible in `inferbench inspect` output as `sched: N new + M running`.

What vLLM's mechanism does **not** do, and what the external scheduler tier is actually for:

| vLLM's internal scheduler | The external scheduler tier |
|---|---|
| Holds requests in an unbounded, engine-internal `waiting` queue | Bounded, observable hold queue with a real `429` at the boundary |
| Has no concept of a user — requests are anonymous to it | Per-user FIFO, per-user concurrency caps, per-user reservation caps |
| First-come, first-served within a step | Round-robin fairness across users |
| Reservation is **per-step and incremental** — it never reserves `max_tokens` up front | Reservation is **worst-case, at admission** — a request that fits is guaranteed never to be preempted |
| Preemption is the pressure-relief valve | Admission control prevents the pressure from building |
| Its waiting queue is invisible to the gateway, the client, and Prometheus | Hold time, hold depth and decision reason are first-class metrics |

**The distinction that makes the tier worth building is the third row of that table.** vLLM reserves what a request needs *right now*; it reserves nothing for what the request *will* need. A request admitted with 3 free blocks that goes on to generate 900 tokens will be preempted mid-generation when the pool fills. The external scheduler's `strict` mode reserves the worst case at admission, so the request it admits is one the engine can finish without evicting. That is the difference between "it was scheduled" and "it will complete".

**This also means the external ledger is advisory, not authoritative.** vLLM does its own admission check regardless of what the external scheduler believed. If the external scheduler over-promises (reserves less than the engine will need), the engine does not fail — it parks the request in its own `waiting` queue and the external ledger's view drifts from reality. §10.7 and §7.3 exist to detect exactly that. The external ledger is a **predictor** of engine behaviour, and §11.2's `sched_reservation_drift_blocks` is the metric that says whether the predictor is any good.

### 6.3 Reservation modes

Reserving the *worst case* (`prompt + max_tokens`) for every request is safe but collapses effective concurrency: a 3072-token prompt with `max_tokens=1024` reserves 256 blocks to produce maybe 40. Three strategies, selectable in config:

| Mode | Blocks reserved at admission | Preemption possible? | Use when |
|---|---|---|---|
| `strict` | `prompt_blocks + output_blocks` | no — this is stricter than vLLM's own per-step reservation (§6.2) | interactive traffic with a latency SLO — the default for this project |
| `prompt_only` | `prompt_blocks` | yes, for long generations — same exposure vLLM already has | batch work where throughput matters more than tail |
| `adaptive` | `prompt_blocks + ceil(min(p95_output, max_tokens) / block_size)` — the measured p95 of what this user actually produced, capped by the request's own `max_tokens` | rarely | once real traffic exists |

`adaptive` promises what the workload actually costs, so the ledger stops
over-promising by the gap between "the most it could say" and "the most it
usually says". The p95 is per user and rolling (last 256 completed
requests, minimum 20 samples): one user's workload is not another's, and a
long-lived process must not grow without bound. With no measurement yet it
falls back to `strict` — an unmeasured promise is no promise at all.

`adaptive` is only available here because this repo already pins output lengths in `config/workloads.yaml` and records per-request output-token counts in sqlite — the measurement exists, it just has to be plumbed in.

**Default: `strict`.** It matches the user's stated requirement that a request which does not fit is *held* rather than admitted and preempted later. It also makes the external ledger agree with the engine's actual needs, which is what keeps `sched_reservation_drift_blocks` near zero.

### 6.4 The decision

```
demand   = reserved blocks for this request under the active mode
headroom = total_blocks - reserved_blocks          # what is genuinely available now

REJECT  if  registry.inflight(user) >= registry.inflight_limit(user)
      or  registry.queued(user)   >= cfg.max_queue_depth_per_user
      or  cfg.max_queue_depth_global reached
      or  prompt_tokens + max_tokens > engine.max_model_len
      → 429 with Retry-After

FIT     if  headroom >= demand
      and scheduler_inflight < cfg.max_in_flight
      and free_blocks - demand >= blocks_the_running_requests_still_need
      → ADMIT: reserve `demand` blocks, dispatch immediately

HOLD    otherwise
      → enqueue in the user's FIFO; the dispatcher re-evaluates on every
        slot release
```

**The last clause is the physical backstop, and it is what makes the smaller
promise safe.** `headroom` is the ledger's view: the promise fits inside the
pool's *total*. The backstop is the engine's view: the pool's *free* blocks
must still cover what the requests already running need for the output they
have left. A request that passes the ledger check but fails this one is
held with reason `physical_capacity`, and the reason it is not admitted is
the one that matters — admitting it is exactly what makes vLLM preempt
mid-generation, where the client waits and the tokens are recomputed. Holding
costs latency; admitting costs preemption. So the promise (1b) may shrink,
but the pool may never be spent on a request the running set needs the room
for.

The reason vocabulary that comes out of this is closed: `fit`, `capacity`
(scheduler in-flight ceiling), `kv_capacity` (the pool's total is spoken
for), `physical_capacity` (the pool's free blocks are gone), `user_reserved`
(this user's budget), `user_concurrency` (this user's slots), `too_long`
(the request does not fit `max_model_len` at all).

**`headroom` counts `total_blocks`, not `free_blocks`.** `reserved_blocks` is
the ledger's *total* outstanding promise — it includes the blocks of requests
that are already running and therefore already allocated. `free_blocks` is the
engine's free count, which already excludes those allocated blocks, so
`free_blocks - reserved_blocks` subtracts the running requests' reservations a
second time and under-counts the headroom by however much the running set
holds. `total_blocks - reserved_blocks` is the only subtraction that is not
double-counting. (This is the shipped formula in `infersched/admission.py`:
`reserved_blocks + demand > total_blocks` → HOLD.)

Two things to notice:

- **The queue depth is bounded.** An unbounded hold queue converts overload into latency instead of into a visible rejection, which is exactly what the gateway's 429 exists to prevent. The bound is per-user *and* global.
- **`max_in_flight` in the scheduler is a second ceiling**, distinct from the gateway's. The gateway's ceiling is about protecting the gateway; the scheduler's is about protecting the engine from a batch it cannot hold.

### 6.5 Rejection is a first-class outcome

A rejection carries a reason, a `Retry-After`, and the client's queue position if known. It is a *measurement*, not an error: `gate_rejected_total{reason}` is a rate panel, and `GatewaySaturated` in `observability/alerts/slo_rules.yml` already fires on it.

---

## 7. Dispatch and slot recycling

### 7.1 The single-writer rule

**Exactly one coroutine may dispatch requests to the engine.** If two dispatchers both read "3 blocks free" and both admit a 3-block request, the engine will preempt one of them — the exact outcome the reservation scheme exists to prevent.

The rule is enforced structurally: the ledger has a single owning task, and `decide()` and `reserve()` are only callable from it.

### 7.2 Event-driven, not polled

Slot release must be *pushed*, not discovered:

```
engine.generate() async generator ends (or raises, or the client disconnects)
  → on_completion(request_id)
  → ledger.release(request_id)          # returns blocks to the pool
  → registry.mark_finished(request_id)  # frees the user's concurrency slot
  → dispatcher_event.set()              # wakes the dispatcher
```

The dispatcher is a single coroutine:

```
while True:
    await dispatcher_event.wait()
    dispatcher_event.clear()
    while (req := next_eligible()):
        if decide(req) is FIT:
            reserve(req); dispatch(req)
        else:
            break
```

Cost when idle: one `await` on an `asyncio.Event`. Cost when hot: the drain loop is pure Python with no I/O.

### 7.3 Reconciliation safety net

The event is the fast path; a slow timer is the safety net. Every `cfg.dispatch_interval_s` (default 1 s) the scheduler re-reads the engine's free-block count and each running request's actual block count, and fixes the ledger:

```
for request_id, reservation in ledger.reservations.items():
    actual = engine_actual_blocks(request_id)     # None if the request is gone
    if actual is None:
        # The request ended without the callback firing (crash, abort, timeout).
        ledger.release(request_id)
    elif actual != reservation.actual_blocks:
        reservation.actual_blocks = actual
```

Without this, a single missed completion callback permanently shrinks capacity, and the system degrades to a slow leak rather than failing loudly. **A leak is the failure mode you cannot see from the outside** — the queue just gets slower — so the ledger must be reconciled, and the drift is a metric (§11).

### 7.4 Reading the engine's per-request blocks

There is no clean public accessor for a single request's block count at the `KVCacheManager` level. Verified in the installed source, the mapping lives one layer down: `SingleTypeKVCacheManager.req_to_blocks` is a `defaultdict[str, list[KVCacheBlock]]` (`v1/core/single_type_kv_cache_manager.py:128`), and the coordinator holds a manager per cache type. The path from the scheduler is therefore:

```
coordinator = scheduler.kv_cache_manager.coordinator
for manager in coordinator.managers:            # one per attention type
    blocks = manager.req_to_blocks.get(request_id, ())
    actual_blocks(request_id) = len(blocks)
```

Sum across managers for the total. This is three levels of private attribute access and it is exactly the kind of path that moves between releases — which is why the repo already probes for these names defensively in `inferbench.inspect`, and why the scheduler must reuse that style rather than assume them:

```
pool = getattr(getattr(sched, "kv_cache_manager", None), "block_pool", None)
total = getattr(pool, "num_gpu_blocks", None)
free  = getattr(pool, "get_num_free_blocks", None)
```

Treat every name above as version-dependent, exactly as `inspect._get()` already does. If the per-request path cannot be found, reconciliation degrades to the free-block count alone — which is still enough to detect gross drift, just not per-request attribution. **Per-request attribution is a nice-to-have; free-block drift detection is the requirement.** Build the reconciler around the free-block count first and add per-request attribution only if the free-block view proves insufficient.

---

## 8. Per-user fairness

### 8.1 The rule

The user's requirement is that **one user can send multiple requests**, and that those requests do not starve other users.

Two mechanisms, both needed:

1. **Per-user FIFO.** A user's queued requests are dispatched in the order they arrived. One user never reorders against itself, and one user's request cannot jump ahead of that user's earlier request.
2. **Round-robin across users.** The dispatcher takes **one** request from each eligible user in turn, then comes back around. Draining one user's queue in full would let a user with 50 queued requests starve everyone else for the entire batch.

```
eligible_users = [u for u in registry.users if u.queued and u.inflight < u.inflight_limit]
for user in rotate(eligible_users):
    req = user.queued.popleft()
    if decide(req).is_fit: dispatch(req)
    else: user.queued.appendleft(req)   # put it back; try the next user
```

**Starvation-freedom argument.** Each round-robin pass serves at least one request from every user that has one, so no user waits more than one full pass of the other users' queues. The per-user `inflight_limit` bounds how much of the batch any single user can occupy, so one user's queue can never grow without bound relative to others.

### 8.2 User-specified concurrency

"The number of request tokens and the concurrency level should be specified by the user."

- **`max_tokens`** — per request, in the request body. Already standard.
- **`concurrency_limit`** — per request, optional. The user may state how many concurrent requests they intend to run. The server **clamps it** to `min(requested, cfg.max_in_flight_per_user)`. A client cannot raise its own ceiling by asking.

The registry then enforces: `inflight(user) + 1 <= effective_limit(user)`. Requests over the limit are queued at the gateway (they are not rejected while the user's queue has room), because the user's own concurrency pattern is intentional.

**Tokens are reserved, not just counted.** A user with 4 concurrent requests each claiming `max_tokens=1024` is holding 4× the output-block reservation. Per-user reservation caps (§5.3) prevent one user's large `max_tokens` from consuming the whole pool — a user that needs 300 reserved blocks when the pool has 320 free is a self-inflicted stall.

---

## 9. Idempotency and duplicate in-flight requests

"Some requests will be new, while others will correspond to already-processing user requests."

### 9.1 The duplicate case

A client retries, a client double-submits, a network hiccup causes a resend. Without handling, the gateway starts a *second* engine request for work already running — burning KV blocks and doubling the user's cost.

### 9.2 Rule

The gateway keeps, per user, a map of live idempotency keys:

```
inflight_keys: dict[(user_id, idempotency_key) → request_id]
```

On submit:

- key absent → create a new `request_id`, record it.
- key present → the request is already running. **Do not start another.** Return `409 Conflict` with `X-Request-Id` set to the original and `X-Idempotent-Replay: true`, and the client can poll or stream from the original. (A "join the existing stream" implementation is possible — the gateway can fan the original SSE stream to all waiters — but that is a second feature, not this one. Start with `409`.)

The key is scoped per user, so two different users may reuse the same key string.

### 9.3 The completion hazard

The registry entry must be removed when the request finishes — **including when it aborts**. A leaked idempotency key makes that user's next legitimate request return `409` forever. The abort path (§10.1) must clean up: idempotency entry, in-flight set, queue position, and reservation.

---

## 10. Failure modes and recovery

Every row here is a production requirement, not a nice-to-have.

| # | Failure | Detection | Response | Residual risk |
|---|---|---|---|---|
| 1 | **Client disconnects mid-stream** | gateway sees the response stream break | gateway aborts upstream → scheduler releases the reservation and the in-flight slot | This is the #1 capacity leak. The abort must be wired end to end or the ledger bleeds until the pool is empty. |
| 2 | **Client stalls, no disconnect** | per-request deadline | scheduler aborts the engine request at `deadline_ts`, releases the reservation | A slow client can still occupy a slot; the deadline bounds it |
| 3 | **Engine raises mid-generation** | exception in `engine.generate()` | reservation released, request marked `FAILED`, client gets `502`; **the reservation is released even though the engine may still hold blocks until it reclaims them** — the reconciler cleans up | If the engine is wedged, the reconciler and the engine disagree; the scheduler must be able to declare itself unhealthy |
| 4 | **Scheduler process dies** | process exit | everything in flight is lost. The engine dies with it (same process). Gateway returns `503` until the scheduler is ready again | Model load takes 30–60 s on restart. The gateway must not report healthy during that window (§10.2) |
| 5 | **Scheduler restarts with a running engine** | n/a | not possible in this topology — the engine is in the scheduler's process, so a scheduler restart is always a cold start | This is a direct consequence of the single-process decision in §3 |
| 6 | **Gateway process dies** | process exit | in-flight requests continue at the scheduler and complete; the gateway loses only its registry | Clients see a dropped connection and must retry — safe only if idempotency keys are used (§9) |
| 7 | **Reservation drift** (ledger ≠ engine) | reconciler (§7.3) | ledger corrected from engine truth | Silent until it is not: alert on `sched_reservation_drift_blocks` |
| 8 | **Held queue grows without bound** | queue-depth gauge | bounded by `max_queue_depth_global`; overflow becomes `429` | Sustained overflow means the GPU is the limit. That is a finding, not a bug |
| 9 | **One user floods the gateway** | per-user counters | per-user queue and in-flight caps; over the limit → `429` | A user with many identities is out of scope — that is auth, not scheduling |
| 10 | **Scheduler is alive but the engine is wedged** | no slot release, no progress | reconcile finds `free_blocks` frozen and in-flight non-empty → mark unhealthy, fail readiness, shed traffic | Needs a liveness signal: *"the free-block count has not changed and no request finished in N seconds"* |

### 10.1 The abort path (failure #1, in detail)

```
client disconnects
  → gateway's response generator is closed
  → gateway sends POST /internal/v1/abort {request_id}
  → scheduler: engine.abort(request_id)   # vLLM's own abort
               ledger.release(request_id)
               registry.mark_finished(request_id)
               dispatcher_event.set()
```

In vLLM, aborting a request that has already been scheduled may not immediately return its blocks to the pool — the engine reclaims them as it processes the abort. The reconciler (§7.3) is what makes the ledger eventually agree. **This is why the reconciler exists and why it runs even when nothing else is happening.**

### 10.2 Readiness

The gateway's `/health` must not say "ok" while the scheduler cannot serve. The gateway's health check should be:

- scheduler reachable **and** its `/internal/v1/state` reports `ready: true`
- `ready` means: engine loaded, model warmed, scheduler dispatcher running, ledger reconciled once

A gateway that reports healthy while the GPU is cold sends traffic into a timeout. During a scheduler restart the gateway must return `503` on `/v1/completions`, not accept-and-hang.

---

## 11. Observability

### 11.1 Metric namespaces

The existing convention in `observability/src/inferobs/metric_names.py` is one vocabulary owned in one place. The new tier adds a third prefix — `sched_*` — alongside `gate_*` and `engine_*`.

```
gate_*    tier 1 — arrivals, users, cache, admission
sched_*   tier 2 — KV budget, hold queue, dispatch decisions
engine_*  tier 3 — the existing tap (running, waiting, KV usage, preemptions)
bench_*   benchmark runs, via pushgateway
```

### 11.2 New metrics

**Tier 1 — gateway**

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `gate_admitted_total` | Counter | — | requests forwarded to the scheduler |
| `gate_held_total` | Counter | `reason` | requests queued at the gateway waiting for a per-user slot |
| `gate_rejected_total` | Counter | `reason` | *existing* — reasons extended to the closed set in §4 |
| `gate_idempotent_replays_total` | Counter | — | duplicates suppressed (§9) |
| `gate_hold_seconds` | Histogram | — | time a request sat in a gateway-side queue |
| `gate_user_queue_depth` | Histogram | — | depth of a user's gateway-side queue at admission time |
| `gate_in_flight` | Gauge | — | requests **running** on the engine |
| `gate_waiting` | Gauge | — | requests **held** in the scheduler's queue |
| `gate_ttft_seconds` | Histogram | — | accept → first token, measured live |
| `gate_tpot_seconds` | Histogram | — | time per output token after the first — vLLM's definition |
| `gate_request_duration_seconds` | Histogram | `backend`, `status` | *existing* |
| `gate_cache_hits_total` / `gate_cache_misses_total` | Counter | — | *existing* — see §11.4 |

**`gate_in_flight` counts running only; `gate_waiting` counts held.** The
original gauge counted both, which made one number answer two different
questions: "is the engine busy?" and "is the queue deep?". An alert on a
combined number fires when the queue is deep even though the engine is
idle, and a panel of it cannot show where the time goes. The two gauges
still sum to the old value, so a combined query is still one expression.

**`gate_ttft_seconds` and `gate_tpot_seconds` are live, not batch.** They are
observed on the gateway's own streaming relay, so they include the scheduler's
hold time in the TTFT and exclude prefill from the TPOT — the same
definitions vLLM and the benchmark harness use, measured where the client
actually is. A benchmark run's numbers arrive hours later through the
pushgateway; these arrive while the traffic is happening, which is the
difference between a metric and a post-mortem.

**Tier 2 — scheduler**

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `sched_free_blocks` | Gauge | — | engine's free KV blocks |
| `sched_total_blocks` | Gauge | — | engine's total KV blocks |
| `sched_reserved_blocks` | Gauge | — | blocks held by the ledger |
| `sched_reservation_drift_blocks` | Gauge | — | `(total_blocks - free_blocks) - allocated_blocks`; **non-zero is a bug** (§10.7) |
| `sched_dispatch_total` | Counter | `decision` | `admit` \| `hold` \| `reject` |
| `sched_hold_depth` | Gauge | — | requests waiting for capacity |
| `sched_hold_seconds` | Histogram | — | how long a request was held before dispatch |
| `sched_dispatch_latency_seconds` | Histogram | — | a slot becoming free → the engine starting the request (should be ~0; non-zero means the event path is broken) |
| `sched_preemption_total` | Counter | — | engine preemptions observed; **should be 0 under `strict` reservation** |
| `sched_capacity_breach_total` | Counter | — | requests the scheduler shed after the engine ran out of blocks |
| `sched_user_inflight` | Gauge | — | total in-flight requests |

**A preemption is a bug report, so it is not allowed to stay silent.** vLLM
preempting means the pool ran out while a request was being generated: that
request is recomputed from scratch and the client sees nothing but latency.
The scheduler watches the engine's own preemption counter, and each
preemption it sees makes it shed the least important of its own requests —
a `503` with `Retry-After` the client can act on, instead of a silent
stall. `sched_capacity_breach_total` counts those sheds;
`sched_preemption_total` counts the engine's preemptions that caused them.

**Drift is the difference between two views of the same pool.**
`total_blocks - free_blocks` is what the *engine* says it has handed out;
`allocated_blocks` is the union of block ids the ledger's tracked requests
actually hold. Drift is the gap between them. It is not `free + reserved -
total` — that expression conflates the ledger's *promise* with the engine's
*grant*, and a reservation that has been made but not yet granted is not
drift, it is a request in flight.

**Hold time and dispatch latency are disjoint by construction.** Hold time is
submit → admission; dispatch latency is slot-free → engine start. They were
originally one number, which meant a p95 that was "the cache is busy" looked
identical to a p95 that was "the event path is broken" — two different
failures with two different fixes. `sched_hold_seconds` carries the first,
`sched_dispatch_latency_seconds` the second.

### 11.3 Cardinality rules

`gate_user_inflight{user_id}` and anything else with a raw user label is **forbidden** in Prometheus. User IDs are unbounded; a thousand users is 1000 series per metric, and a scraper bug can mint millions. Per-user state is exposed through:

- the scheduler's `/internal/v1/state` endpoint (on demand, not scraped), and
- **aggregated** histograms — `gate_user_queue_depth` is a histogram over requests, not a gauge per user.

If per-user dashboards are genuinely needed later, the aggregation has to happen in the collector (a recording rule that buckets by a hashed user into a bounded number of buckets). Do not label by raw user id.

### 11.4 One metric bug to fix while we are here

`gate_cache_misses_total` is declared in `gateway/src/infergw/metrics.py` and queried by `observability/grafana/dashboards/live.json`, but **nothing ever incremented it** — only `cache_hits` was incremented (`gateway/src/infergw/app.py`). The Live dashboard's "Cache hits vs misses" panel therefore plotted a permanently flat zero. The `ExactCache` tracks its own internal `self.misses` counter, which reaches `/gateway/cache` as JSON but never Prometheus. Fixing this is a two-line change and belongs with this work, because the cache hit rate is now a first-class production signal.

**Fixed.** `gateway/src/infergw/app.py:152` now increments `cache_misses` on the miss path, next to the existing `cache_hits` increment on the hit path. The counter is incremented *before* the idempotency check, so a duplicate submit counts as a cache miss and then as a replay — the two are different facts and the panel shows both.

### 11.5 Alerts

Existing alerts in `observability/alerts/slo_rules.yml` cover TTFT burn, error rate, gateway saturation and KV-cache pressure. The scheduler tier's own group, `scheduler_alerts`, adds five:

| Alert | Condition | Why |
|---|---|---|
| `ReservationDrift` | `sched_reservation_drift_blocks != 0` for 2m | A ledger bug — capacity is being lost silently |
| `UnexpectedPreemption` | `increase(sched_preemption_total[5m]) > 0` | Under `strict` reservation this must never fire; if it does, the reservation model is wrong |
| `SchedulerHoldQueueGrowing` | `sched_hold_depth > 0` for 5m | The GPU is the bottleneck; someone needs to know before users do |
| `SchedulerDispatchStalled` | `sched_dispatch_latency_seconds` p95 > 0.5 for 5m | The event path is broken; held requests are waiting for a poll |
| `SchedulerDown` | `absent(sched_total_blocks)` or `sched_total_blocks == 0` for 2m | The tier that owns the GPU is gone, or up but has never attached an engine |

`UserFairnessViolation` (p95 hold time for one bucket ≫ median) is **not
shipped**. It needs the bucketed aggregation from §11.3, and until per-user
state is aggregated into a bounded number of buckets there is nothing honest
for it to fire on. It stays optional.

The same eleven rules are written twice: `observability/alerts/slo_rules.yml`
for the local stack and `deploy/k8s/prometheusrule.yaml` for the cluster.
`tests/test_alerts.py` fails if the two drift apart, and fails if any rule or
dashboard panel queries a metric nothing emits — a rule that names a metric
that does not exist never fires, and a panel that queries one is a grey box
that looks like an outage.

---

## 12. API contracts

### 12.1 Public — client → gateway

```
POST /v1/completions
  Headers:
    X-User-Id:        u-12345          (required)
    X-Request-Id:     <uuid>           (optional; gateway generates if absent)
    Idempotency-Key:  <opaque>         (optional)
  Body:  OpenAI-compatible {model, prompt, max_tokens, stream, temperature, …}
         + "concurrency_limit": 4       (optional, clamped server-side)

  200  application/json | text/event-stream
  400  invalid body / missing X-User-Id
  409  idempotency replay (headers: X-Request-Id, X-Idempotent-Replay: true)
  429  over a limit; body {error, reason, in_flight}; Retry-After: 1
  503  scheduler not ready, or the scheduler shed this request
       (body {error: "capacity_breach", reason, request_id}; Retry-After: 1)

  Response headers:
    X-Request-Id, X-User-Id, X-Trace-Id
    X-Gateway-Queue-Time: 0.004213        # seconds at the gateway
    X-Scheduler-Decision: admit|hold|reject|cache
    X-Scheduler-Reason: fit|capacity|kv_capacity|physical_capacity|
                        user_reserved|user_concurrency|too_long
    X-Scheduler-Hold-Time: 0.183          # seconds held in tier 2
    X-Kv-Blocks-Reserved: 96             # what the ledger promised
    X-Kv-Blocks-Demand: 96               # what the request asked for
```

`reserved` and `demand` are different numbers on purpose. `demand` is what
the request needs; `reserved` is what the ledger gave it. A held request has
demand > 0 and reserved 0 — it was refused room, not blocks. Reading only
`reserved` makes a held request look like it needed nothing.

**A scheduler-initiated abort is a `503`, not a silent truncation.** When the
engine preempts and the scheduler sheds a request to protect the pool, the
client must be able to tell that apart from a finished answer: a `503` with
`Retry-After: 1` says *try again*, and a partial body says *here is what I
had*. In streaming mode the status line is already sent, so the error
arrives as an SSE error frame — but it is still an error frame, not an
empty completion.

The gateway passes the upstream body through untouched. It must not buffer a streaming response — the existing `infergw` code already gets this right, and the reason is stated in its own docstring: *"Buffering here would silently destroy TTFT, which is the metric half this platform exists to measure."*

### 12.2 Internal — gateway → scheduler

```
POST /internal/v1/submit
  {request_id, user_id, prompt, max_tokens, temperature, top_p, seed,
   idempotency_key, deadline_ts, trace_id}
  → 200 {decision: "admit"|"hold"|"reject", reason, reserved_blocks,
         demand, hold_seq}

POST /internal/v1/abort
  {request_id, reason}
  → 200 {released_blocks}

GET  /internal/v1/stream/{request_id}
  → text/event-stream, one SSE frame per event

GET  /internal/v1/state
  → {ready, inflight, held, free_blocks, total_blocks, reserved_blocks,
     allocated_blocks, drift_blocks,
     per_user: {user_id: {inflight, queued, reserved_blocks}}}

GET  /internal/v1/requests/{request_id}
  → {request_id, user_id, state, decision, reason, reserved_blocks,
     admitted_ts, hold_seconds, completion_tokens, error}

GET  /health   → 200 always — liveness; the process is alive while the
                 weights are still loading
GET  /ready    → 503 until the engine is attached, then 200 — readiness
```

The submit body is **flattened**: the sampling parameters are top-level fields,
not a nested `params` object. There is no `prompt_tokens` field either — the
scheduler counts the prompt itself with the engine's own tokenizer
(`dispatcher.engine.count_tokens`), because a token count from another
tokenizer is a guess and the block demand is computed from it.

`/health` and `/ready` are two different questions and both are needed.
`/health` stays 200 while the weights load, because the process *is* alive and
that is what a liveness probe asks; a liveness probe that fails during a
two-minute model load restarts the pod forever. `/ready` returns 503 until
there is an engine for the ledger to account for, so a load balancer does not
send traffic to a scheduler that can only reject it.

`/internal/v1/state` is what the gateway's readiness check and the per-user dashboards read. It must not be exposed publicly — the per-user map is unbounded (§11.3).

---

## 13. Configuration

New file `config/scheduler.yaml`, alongside the existing `config/gateway.yaml` and `config/engine/*.yaml`:

```yaml
# Which engine config from config/engine/ this scheduler drives.
engine: optimized

# strict     = reserve prompt blocks + worst-case output blocks (no preemption)
# prompt_only = reserve prompt blocks only (preemption possible, more throughput)
# adaptive   = reserve prompt blocks + the measured p95 output, so the promise
#              follows what the model actually produces. Falls back to strict
#              until there is enough measurement to trust the p95.
reservation_mode: strict

# Capacity.
# max_in_flight is derived from the engine's max_num_seqs at startup, because
# promising more concurrent requests than the engine can run is how a queue
# turns into preemption. It is not set here: a hand-copied number drifts
# from the engine's the moment either side changes. Uncomment to lower it.
# max_in_flight: 32           # scheduler-side ceiling on concurrent engine requests
max_in_flight_per_user: 4   # one user cannot hold the whole GPU; the rest of
                            # that user's requests wait in that user's queue
max_queue_depth_global: 256 # held requests, total
max_queue_depth_per_user: 64  # a user's waiting requests, before one is refused
max_reserved_blocks_per_user: 512   # stops one user monopolising the KV pool

# Where this server listens.
host: 0.0.0.0
port: 9001
```

**`max_in_flight` is not in the config because the engine already knows it.**
The scheduler reads `max_num_seqs` off the engine's own vLLM config when the
engine attaches and uses that as its ceiling; a configured value may only
*lower* it. This is the same rule as `block_size` below, and for the same
reason: a ceiling that exists in two places is a ceiling that will disagree
with itself, and the smaller number wins without anybody noticing which file
it came from.

**No `block_size` key.** The block size is not configured here: it is read from
the engine's own `vllm_config.cache_config.block_size` at startup, so the
scheduler and the engine cannot disagree about it. A config key would be a
second copy of a value the engine already knows, and the day they disagree
the ledger is wrong in a way that only shows up as preemption.

**No `dispatch_interval_s`, no `reconcile_on_step`.** The dispatcher is
event-driven (§7.2) and reconciliation happens on every engine step, which is
where the engine's truth actually changes. A poll interval would be a knob
whose only job is to be tuned wrongly.

Gateway additions to `config/gateway.yaml`:

```yaml
scheduler_url: http://localhost:9001     # tier 2
scheduler_timeout_s: 300.0
require_user_id: true                     # reject requests with no X-User-Id
idempotency_ttl_s: 900.0                 # how long a key is remembered
max_in_flight_per_user: 32                # the gateway's own ceiling
```

The gateway's `max_in_flight_per_user` and the scheduler's are **not** the same
number and must not be. The gateway's is a ceiling on how many of a user's
requests it will *hold open*; the scheduler's is a ceiling on how many of a
user's requests *run at once*. The gateway's must be at or above the
scheduler's, or the gateway refuses a request the scheduler was willing to
hold. 32 ≥ 4 satisfies that. See §18.1.

---

## 14. Repo mapping

### 14.1 New package — `scheduler/`

```
scheduler/
├── pyproject.toml                    # inferbench-style: hatchling, deps inferobs/fastapi/httpx
└── src/infersched/
    ├── __init__.py
    ├── config.py          # SchedulerConfig, ReservationMode, load_scheduler()
    ├── ledger.py          # ReservationLedger — reservations, drift, release
    ├── registry.py        # UserRegistry — per-user in-flight, queues, caps
    ├── admission.py       # decide() — the pure FIT/HOLD/REJECT function
    ├── dispatcher.py      # the single dispatcher coroutine
    ├── state.py           # reconciliation: engine truth → ledger
    ├── server.py          # FastAPI: /internal/v1/* + the engine-facing stream
    ├── metrics.py         # sched_* Prometheus instruments
    └── tests/             # (see §15) — or extend the root tests/ dir
```

The repo's existing layout puts tests in a single root `tests/` directory rather than per-package, so the new tests should go in `tests/` too, matching `test_gateway.py`, `test_metrics.py`, etc.

**The engine host lives in `inferbench`, not `infersched`.** `infersched` needs the engine config, the engine builder and the instrumentation tap, all of which `inferbench` already owns; putting the host in `infersched` would make `inferbench` depend on `infersched` for the server, and `infersched` depend on `inferbench` for the config — a cycle. The dependency runs one way: `infersched` → `inferbench`.

```
benchmarking/src/inferbench/
├── engine.py         # unchanged — builds the engine
├── instrument.py     # unchanged — the stat-logger tap
├── engine_host.py    # NEW (Phase 0) — owns AsyncLLMEngine + the tap;
│                     #       the engine-truth readers, in one place
└── step_engine.py    # NEW (Phase 1) — the *sync* in-process engine plus the
                      #       step-loop thread the scheduler tier drives
                      #       (§3: the async serving path cannot put the
                      #        engine core in this process)
```

Two engine hosts, deliberately. `engine_host.EngineHost` wraps `AsyncLLMEngine` and serves `inferbench serve` — the single-user benchmark-facing server, which wants async streaming and does not care about block accounting. `step_engine.StepEngine` wraps the sync `LLMEngine` and is what the scheduler tier drives, because it is the only one whose scheduler is a Python object in this process. They share `engine_host`'s readers (`free_blocks`, `total_blocks`, `request_blocks`, `block_size`), so there is one definition of "what the engine thinks".

### 14.2 New/changed files

| File | Change |
|---|---|
| `scheduler/pyproject.toml` | new — package `infersched` |
| `scheduler/src/infersched/*.py` | new — see above |
| `config/scheduler.yaml` | new — §13 |
| `config/gateway.yaml` | add `scheduler_url`, `require_user_id`, `idempotency_ttl_s` |
| `observability/src/inferobs/metric_names.py` | add the `sched_*` constants and the new `gate_*` ones |
| `gateway/src/infergw/app.py` | forward to the scheduler; per-user registry; idempotency; abort on disconnect; increment `cache_misses` |
| `gateway/src/infergw/config.py` | `SchedulerConfig` fields on `GatewayConfig` |
| `gateway/src/infergw/metrics.py` | new counters/histograms from §11.2 |
| `gateway/src/infergw/registry.py` | new — the per-user registry |
| `benchmarking/src/inferbench/engine_host.py` | new (Phase 0) — the engine host and the engine-truth readers, extracted from `inferbench.server` and `inferbench.inspect` |
| `benchmarking/src/inferbench/server.py` | the engine host moves out to `inferbench.engine_host`; `inferbench.server` stays as the single-user / benchmark-facing server |
| `benchmarking/src/inferbench/inspect.py` | unchanged behaviour — its private readers now come from `inferbench.engine_host` |
| `observability/grafana/dashboards/live.json` | new panels: hold queue depth, reservation drift, dispatch latency, preemption counter |
| `observability/alerts/slo_rules.yml` | the 5 new alerts from §11.5 |
| `deploy/k8s/scheduler-deployment.yaml` | new — 1 replica, 1 GPU, `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| `deploy/k8s/gateway-deployment.yaml` | point at the scheduler; **keep `maxReplicas: 1`** until the cache moves out of the process (§16) |
| `scripts/dev.sh` | start three tiers; the scheduler on :9001 |
| `README.md` | document the tier topology and the new flow |

### 14.3 What does **not** change

This matters as much as what does:

- `inferbench.inspect`, `inferbench.instrument`, `inferbench.metrics`, `inferbench.store`, `inferbench.report`, `inferbench.cost` — untouched. The benchmark harness is a *client* of the same engine and stays exactly as it is.
- `inferbench.runner` — untouched; it already speaks OpenAI-compatible HTTP and can point at the gateway unchanged. The new tier is transparent to it.
- `inferbench.client` — **one line changed.** The client now sends `x-user-id` on every request (`benchmarking/src/inferbench/client.py:95`), because the gateway requires it: with `require_user_id: true`, a request without the header is a 400. The client defaults to `"bench"`, so a single benchmark run is one user with one budget — which is the honest thing for it to be, and is why the A/B comparison in §16 points the harness at the gateway rather than at the engine.
- `inferobs.slo`, `inferobs.burn` — untouched; they consume the store, not the live path.
- The gateway's existing order of operations (cache → admission → route → proxy) — the scheduler slot is inserted between admission and route.

---

## 15. Test plan

GPU-free, matching the existing suite's approach: pure functions plus fakes, no engine required.

| Test file | Covers |
|---|---|
| `test_admission.py` | the FIT/HOLD/REJECT truth table. Reservation modes produce the expected demand. A request that does not fit is HELD, never admitted. Over-limit is REJECTED with the right reason. `max_model_len` overflow is rejected. |
| `test_ledger.py` | reserve/release accounting; the drift invariant `free + reserved <= total`; release is idempotent; releasing an unknown request is a no-op, not an error; a leak is detected by reconciliation. |
| `test_registry.py` | per-user in-flight cap; per-user queue cap; FIFO within a user; the user's concurrency limit is honoured; a request cannot raise its own ceiling. |
| `test_dispatcher.py` | slot release dispatches the next held request; the dispatcher is the only writer (a second concurrent dispatch cannot over-commit); round-robin fairness — with two users holding requests, neither starves; reconciliation recovers a missed completion callback. |
| `test_scheduler_api.py` | the internal API contract over `httpx.ASGITransport` + `MockTransport`; `/internal/v1/state` reports consistent ledger and registry numbers; `/internal/v1/abort` releases exactly the right blocks. |
| `test_gateway_scheduler.py` | gateway → fake scheduler: forward, propagate the decision headers, **wait on a hold and 429 only on a reject**, abort on client disconnect, suppress idempotent duplicates, and (from §11.4) actually increment the cache-miss counter. |
| `test_alerts.py` | the two rule stacks (local YAML and the K8s `PrometheusRule`) carry identical alerts and recording rules, and no rule or dashboard panel queries a metric nothing emits. |

**A hold is not a rejection.** The gateway *waits* on a held request: it
streams the scheduler's SSE and the caller sees one continuous response,
however long the hold lasted. A `429` means the request will never run —
it is too long for the context, it is bigger than its user's whole block
budget, or a queue is full — and those are the only cases that get one. The
original plan mapped a hold to a 429; that would have turned ordinary
backpressure into a visible error, and `gate_rejected_total` would have
become a rate panel that fires during every peak.

The single most valuable test is the **single-writer invariant**: it needs a fake engine whose free-block count is authoritative, two concurrent dispatch attempts, and an assertion that the second one is refused. That is the bug that turns "reserve blocks" into "reserve blocks and hope".

The GPU path (`serve`, `bench`, `inspect`) still needs the real card — unchanged, and unchanged in its coverage.

---

## 16. Migration path

Each step is separately shippable and separately verifiable.

| Phase | Change | Gate | Status |
|---|---|---|---|
| 0 | Extract `inferbench.engine_host` from `inferbench.server`; no behaviour change | `inferbench serve` still works; existing tests still pass | done |
| 1 | Add `ledger` + `admission`; the scheduler admits everything (no HOLD path yet) | Ledger and engine free-block counts agree after every request | done |
| 2 | Enable HOLD; add `registry` and round-robin dispatch | A request that does not fit is held, then dispatched when a slot frees; no preemption occurs | done |
| 3 | Gateway forwards to the scheduler; per-user registry; idempotency | Two users cannot starve each other; a duplicate submit does not start a second engine request | done |
| 4 | Reconciliation, alerts, dashboards, K8s manifests | `sched_reservation_drift_blocks` stays 0 over a long run; `UnexpectedPreemption` never fires | done — gate met on the live GPU |

Phase 2 is the one that changes behaviour for users, and it is the one worth running against the benchmark harness first: point `inferbench bench --base-url http://localhost:9000` at the new path and compare TTFT p95 against the recorded baseline in `results/report.md`. That comparison is the real acceptance test, and the harness to run it already exists.

### The Phase 4 gate, measured

A 16.6-minute soak against the live GPU drove 2,218 requests through the gateway: 6 users × 5 concurrent clients for 4 minutes, then 8 × 7 for 12 minutes, with prompts long enough to fill the KV pool (2,200 tokens in phase B, 400 in phase A). Every request carried a unique nonce, so the gateway's exact-match cache answered **none** of them — `cache_hits: 0`, `reached_scheduler: 2218`. A soak that never reaches tier 2 proves nothing, and the first attempt at one did exactly that.

What the ledger did under it:

| | |
|---|---|
| Samples | 496 (one every 2 s) |
| `sched_reservation_drift_blocks` | 0 at every sample; `drift_samples_nonzero: 0` |
| `sched_preemption_total` | 0 throughout; `preempt_samples_nonzero: 0` |
| KV pool | down to **191 free blocks of 3,662** — 95% full |
| Hold queue | peaked at 32 deep |
| Reserved / allocated at peak | 3,552 / 3,471 |
| Responses | 2,218 × `200`; 0 errors, 0 rejects |

The interesting part is *where* it held. The pool sat at 95% utilisation with 32 requests waiting for capacity that was not yet free, which is exactly the condition under which a reservation model is wrong: the ledger promised blocks for requests the engine could not start, and the engine still never had to take one back. Under `strict` mode the reservation is the thing being tested, and a preemption would have meant the estimate was too small. It was not.

The soak also settles a question the design left open: a **hold is not a failure**. 2,135 of the 2,218 requests came back with `x-scheduler-decision: hold` — queued behind capacity — and every one of them finished with a `200`. A hold that surfaced as an error would have been a `429` on 96% of the traffic.


---

## 17. Risks and deliberate ceilings

| Risk / ceiling | Why it is acceptable now | Upgrade path |
|---|---|---|
| The scheduler is a single point of failure and owns the GPU | One GPU, one engine — there is nothing to shard yet | Multiple GPUs → one scheduler replica per GPU, gateway routes by user affinity (§3) |
| The gateway runs at 1 replica because the exact-match cache is per-process | The existing `deploy/k8s/gateway-deployment.yaml` already says this | Move the cache to Redis, then let the HPA scale the gateway |
| `strict` reservation over-reserves and lowers effective concurrency | The alternative is preemption, which is worse for tail latency | `adaptive` mode once real output-length distributions are measured |
| **The external ledger duplicates an internal one** — vLLM reserves blocks and parks requests itself (§6.2) | The two answer different questions: vLLM asks *"does this fit right now?"*, the scheduler asks *"will this fit for its whole life, and is that fair to the other users?"* | If the drift metric shows the predictor is not earning its keep, collapse the tier back into the gateway and keep only the per-user registry and the fairness policy |
| The ledger can drift from the engine | Reconciled on every engine step — where the engine's truth actually changes — and drift is a metric *and* an alert | The reconciliation already runs at step granularity; there is nothing left to upgrade |
| A held request waits behind a large reservation | Per-user reserved-block caps bound this | Priority classes, if real traffic shows a need |
| Per-user state is not durable across a gateway restart | Idempotency keys make resubmission safe; the scheduler is the source of truth for capacity | Move the registry to Redis when it must survive restarts |
| The scheduler's own HTTP hop adds latency | It is a local hop on loopback and it buys the queue-time measurement the platform exists for | Keep the scheduler and engine co-located; measure before optimising |
| **The scheduler tier cannot use `AsyncLLMEngine`** — vLLM 0.31.x raises `NotImplementedError` for an async in-process engine core, and `AsyncLLM` forks unconditionally (§3) | The tier does not need async: it drives the sync `LLMEngine` from its own step-loop thread and hands results back through futures. The benchmark-facing `inferbench serve` keeps its async engine | A future vLLM that allows an async in-process core would let the scheduler drop its step thread — but it buys concurrency we do not need, since the GPU is the bottleneck |

---

## 18. Open questions

1. **Does the gateway or the scheduler enforce the per-user concurrency limit?** ~~Recommendation: one number, `max_in_flight_per_user`, read from the same config file by both tiers.~~ **Amended — the two numbers mean different things and must not be merged.** The gateway's `max_in_flight_per_user` (32) is a ceiling on how many of a user's requests the gateway will *hold open*; those requests are all forwarded to the scheduler, which queues them. The scheduler's `max_in_flight_per_user` (4) is how many of a user's requests may *run at once*; the rest wait in that user's queue. One shared number would either let a user run four requests and hold thirty-two sockets, or hold four sockets and starve the GPU of users. The invariant that does matter is directional, not equality: **the gateway's number must be at or above the scheduler's**, or the gateway refuses a request the scheduler was willing to hold — which is the confusing 429 this question was worried about, and it is now a configuration error with a test.
2. **Should the hold queue be visible to the client?** ~~Recommendation: keep `429` for v1 and add `X-Hold-Position`.~~ **Amended — the client waits on the hold instead.** The gateway keeps the request open and relays the scheduler's SSE, so a held request is one continuous response to the client and the wait is invisible from the outside except as latency. The `202`-plus-position design is only needed when the client *cannot* hold a connection open — a mobile client, or a request that must not tie up a server-side connection for minutes. The clients here are long-lived and the load is server-side, so waiting is the better answer: it keeps the connection accounting honest and it avoids a client-side retry loop that would re-send the prompt and re-consume the prefill. The queue depth and hold time are still observable — `sched_hold_depth`, `sched_hold_seconds`, and the per-user map on `/internal/v1/state`.
3. **What happens to a held request when the client times out?** Recommendation: drop it from the queue and release nothing (it never reserved). The client already got a timeout; keeping it would waste a slot later.
4. **Is `adaptive` reservation mode worth the complexity before there is real traffic?** Recommendation: no. Build `strict` and `prompt_only`, revisit when `results/bench.db` has real output-length distributions.
5. **Should the scheduler be in the same process as the gateway after all?** The answer becomes "no" as soon as the gateway needs more than one replica, and the moment there is real traffic that will happen. Keeping them separate now costs one local HTTP hop.
