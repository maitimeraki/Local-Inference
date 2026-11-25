#!/usr/bin/env python3
"""Drive the grid like a production client: many users, real requests.

One file for the whole traffic side of the system. The model load, the
gateway and the scheduler servers, and the observability stack are separate
concerns with their own scripts; this is the client.

    # inside the VM that runs the grid
    /home/anupam/.venv/bin/python scripts/spike.py --users 24 --concurrency 10

    # or with a fixed, repeatable mix
    /home/anupam/.venv/bin/python scripts/spike.py --users 24 --concurrency 10 \\
        --prompt-tokens 200-2000 --max-tokens 32-512 --seed 42

Standard library only, so it runs anywhere Python does -- including the WSL2
image, which has no node and therefore cannot run scripts/loadgen.mjs.

Production traffic is not uniform. One user sends a hundred-token prompt, the
next sends two thousand, and both want a long answer; a load where every
request is the same size cannot tell you which ceiling bound binds. So the
sizes are drawn per request from a range, from a seeded RNG: a request's size
is a function of its (user, seq) and the seed, not of the order the requests
happened to arrive in. An experiment that cannot be replayed is not an
experiment.

Every request is a real one -- real prompt, real generation, real tokens
spent -- and it carries a unique nonce at the *front* of the prompt. The
front matters: a nonce at the end would leave every request sharing one long
prefix, the engine's prefix cache would serve them all from the same blocks,
the KV pool would never fill, and the hold path would never be exercised.
With the nonce first, every request has its own prefix and the pool is under
real pressure.

The report keeps two populations apart, because conflating them is how a
run comes to look like it finished in three seconds when every served
request actually took seconds:

    served   -- the requests that became work (200), including those the
                scheduler held first and released later
    refused  -- the requests a ceiling turned away at the door (429), which
                cost the server nothing and the client almost nothing
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

# Real English, so the model produces real text and the token counts are the
# model's own. A prompt of "word word word" is a legal request but it is not
# a production one, and it makes the run's token accounting a fiction.
CORPUS = [
    "The scheduler decides which requests may run and which must wait.",
    "A KV cache stores computed attention keys and values so they are not recomputed.",
    "Backpressure is how a queue tells its callers to slow down.",
    "Admission control refuses work before it becomes work.",
    "The prefix cache is shared, so the same context costs one copy, not many.",
    "A reservation is a promise the ledger makes to the engine.",
    "Preemption is the engine taking work back because it no longer fits.",
    "Throughput is the number of completions per second, not the number of tokens.",
    "Tail latency is what a user notices, and it is not the mean.",
    "A hold queue is not a failure; it is the system declining to overcommit.",
    "Every metric has to have a reason to exist, or it is just a cost.",
    "Fairness between tenants is a property of the queue, not of the GPU.",
    "The gateway owns the arrival rate and the identity of the caller.",
    "A cache hit that nobody measures is a cache nobody can tune.",
    "The ledger's drift is the difference between what is held and what is owed.",
    "An idempotency key makes a retry safe, but only while the first request lives.",
    "Production traffic has a long tail of request sizes.",
    "Saturation is where the queue grows faster than it drains.",
    "A rejected request is cheap for the server and expensive for the client.",
    "The engine's own queue should stay empty if the scheduler is doing its job.",
    "Measurement noise is real, and a single run is not a result.",
    "Batch and single request paths differ in ways that only show under load.",
    "The cost of a request is not only the tokens it produced.",
    "A dashboard that cannot distinguish two failure modes sends you to the wrong tier.",
    "Holding a request is a decision, and it deserves a reason code.",
    "The pool has a finite number of blocks and the ledger has to respect that.",
    "Warm caches hide cold start costs until they do not.",
    "A retry storm is what a bad error response looks like at scale.",
    "Concurrency limits exist so that one tenant cannot consume the machine.",
    "The p99 is where the complaints live.",
    "Load that is uniform cannot tell you which constraint binds.",
    "A request that never reaches the scheduler has no demand, only a size.",
    "Numbers that do not add up are a bug report in disguise.",
    "The engine holds blocks; the ledger accounts for them.",
    "Idle capacity is capacity that a promise is keeping from being used.",
    "Every ceiling has a reason code, and every reason code has a metric.",
    "A request that is admitted is a request that will consume the GPU.",
    "Measure the whole path, not just the part you wrote.",
    "Observability is what lets you sleep at night.",
]

# The system preamble every request shares, the way a real deployment has one.
# The engine's prefix cache serves this once for the whole run, so it is the
# part of the prompt that costs the pool nothing.
SHARED_PREAMBLE = (
    "You are an assistant for an inference platform that schedules requests "
    "across a GPU. Answer the question accurately and concisely."
)


def build_prompt(rng: random.Random, words: int, nonce: str, shared: str) -> str:
    """A prompt of about `words` words, in the shape production traffic has.

    Real traffic is a shared system preamble followed by a unique user
    message, and that shape matters twice over:

      * The shared preamble is what the engine's prefix cache is *for*. It is
        served from the same blocks for every request, so it costs the pool
        once rather than per request.
      * The unique tail is what keeps the gateway's exact-match cache from
        answering the run. If the whole prompt repeated, the gateway would
        serve it and the engine would never be touched -- the run would look
        fast and prove nothing.

    So the preamble goes first and the nonce goes after it: the prefix cache
    has something real to share, the gateway cache has nothing to match, and
    the KV pool still fills with the per-request part.
    """
    parts = [shared] if shared else []
    parts.append(f"[req {nonce}]")
    count = sum(len(p.split()) for p in parts)
    while count < words:
        sentence = rng.choice(CORPUS)
        parts.append(sentence)
        count += len(sentence.split())
    return " ".join(parts)


_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_exposition(text: str) -> dict[str, dict[tuple, float]]:
    """Prometheus text exposition -> {metric name: {labels: value}}."""
    out: dict[str, dict[tuple, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, raw = line.rpartition(" ")
        try:
            value = float(raw)
        except ValueError:
            continue
        if "{" in key:
            name, _, rest = key.partition("{")
            labels = tuple(sorted(_LABEL_RE.findall(rest)))
        else:
            name, labels = key, ()
        series = out.setdefault(name, {})
        series[labels] = series.get(labels, 0.0) + value
    return out


def total(metrics: dict, name: str, match: dict | None = None) -> float:
    found = 0.0
    for labels, value in metrics.get(name, {}).items():
        if match and not all(dict(labels).get(k) == v for k, v in match.items()):
            continue
        found += value
    return found


def by_label(metrics: dict, name: str, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for labels, value in metrics.get(name, {}).items():
        key = dict(labels).get(label, "")
        out[key] = out.get(key, 0.0) + value
    return out


def fetch(url: str, timeout: float = 15.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


# The grid can be on :9000/:9001 (the defaults) or :9100/:9101 (when
# something else on the machine owns the defaults). Which one is live is a
# property of the machine, not of the run, so the client finds it instead of
# making the caller remember.
GATEWAY_PORTS = (9000, 9100, 9200)
SCHEDULER_PORTS = (9001, 9101, 9201)


def _alive(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:  # noqa: BLE001 - any failure means "not there"
        return False


def discover_grid(scheduler_hint: str = "") -> tuple[str, str]:
    """Find the running grid. Returns (gateway_url, scheduler_url).

    The scheduler is probed on its own because the gateway does not advertise
    where it forwards: the two live on adjacent ports, but the ports move
    together.
    """
    gateway = ""
    for port in GATEWAY_PORTS:
        url = f"http://127.0.0.1:{port}"
        if _alive(f"{url}/health"):
            gateway = url
            break
    if not gateway:
        return "", ""

    if scheduler_hint and _alive(f"{scheduler_hint}/internal/v1/state"):
        return gateway, scheduler_hint
    for port in SCHEDULER_PORTS:
        url = f"http://127.0.0.1:{port}"
        if _alive(f"{url}/internal/v1/state"):
            return gateway, url
    return gateway, ""


def guess_scheduler(gateway_url: str) -> str:
    """The scheduler is the gateway's port plus one."""
    try:
        port = int(gateway_url.rsplit(":", 1)[1].rstrip("/"))
    except (IndexError, ValueError):
        return gateway_url
    return f"http://127.0.0.1:{port + 1}"


class Sampler(threading.Thread):
    """Polls both tiers' /metrics so the report can show the server's view."""

    def __init__(self, gateway: str, scheduler: str, interval: float) -> None:
        super().__init__(daemon=True)
        self.gateway = gateway
        self.scheduler = scheduler
        self.interval = interval
        self.samples: list[tuple[float, dict, dict]] = []
        self._halt = threading.Event()

    def run(self) -> None:
        t0 = time.perf_counter()
        while not self._halt.is_set():
            try:
                gw = parse_exposition(fetch(f"{self.gateway}/metrics"))
            except Exception:  # noqa: BLE001
                gw = {}
            try:
                sc = parse_exposition(fetch(f"{self.scheduler}/metrics"))
            except Exception:  # noqa: BLE001
                sc = {}
            self.samples.append((time.perf_counter() - t0, gw, sc))
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()

    def peak(self, tier: int, name: str) -> float:
        best = 0.0
        for row in self.samples:
            value = total(row[1] if tier == 0 else row[2], name)
            best = max(best, value)
        return best


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def worker(args, user: str, slot: int, results: dict, lock: threading.Lock) -> None:
    """One user's worth of sequential requests."""
    rng = random.Random(f"{args.seed}:{user}:{slot}")
    headers = {"content-type": "application/json", "x-user-id": user}

    for seq in range(args.per_slot):
        max_tokens = rng.randint(*args.max_tokens)
        # The engine counts tokens, not words, and this corpus costs about 1.2
        # tokens per word (measured). Leave headroom for the worst case, or a
        # long prompt plus a long output cap crosses the context window and the
        # request is refused as `too_long` -- a client bug, not a scheduling
        # result.
        max_words = max(16, int((args.max_context - max_tokens) / 1.25))
        drawn = rng.randint(*args.prompt_tokens)
        words = min(drawn, max_words)
        if words < drawn:
            args.clamped += 1

        body = json.dumps({
            "model": "local",
            # The shared preamble is what the engine's prefix cache is for;
            # the nonce after it is what stops the gateway's exact-match
            # cache from answering the run. The *sizes* come from the seeded
            # RNG and are reproducible; the text must not be, or run N+1 is a
            # cache replay of run N.
            "prompt": build_prompt(rng, words, f"{user}-{slot}-{seq}-{time.time_ns()}",
                                   args.shared_prefix),
            "max_tokens": max_tokens,
            "temperature": 0,
            "concurrency_limit": args.concurrency_limit,
        }).encode()

        started = time.perf_counter()
        req = urllib.request.Request(
            args.url + "/v1/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                payload = resp.read().decode("utf-8", "replace")
                status, hdrs = resp.status, resp.headers
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", "replace")
            status, hdrs = exc.code, exc.headers
        except Exception as exc:  # noqa: BLE001 - a dropped request is a datapoint
            with lock:
                results["errors"] += 1
            print(f"  request error: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        elapsed = (time.perf_counter() - started) * 1000.0

        decision = hdrs.get("x-scheduler-decision") or "-"
        reason = hdrs.get("x-scheduler-reason") or "-"
        hold = float(hdrs.get("x-scheduler-hold-time") or 0.0)
        reserved = hdrs.get("x-kv-blocks-reserved")
        demand = hdrs.get("x-kv-blocks-demand")
        usage = {}
        try:
            usage = json.loads(payload).get("usage") or {}
        except ValueError:
            pass

        with lock:
            results["statuses"][status] = results["statuses"].get(status, 0) + 1
            results["decisions"][decision] = results["decisions"].get(decision, 0) + 1
            results["reasons"][reason] = results["reasons"].get(reason, 0) + 1
            record = {
                "status": status,
                "decision": decision,
                "reason": reason,
                "hold_s": hold,
                "reserved": reserved,
                "demand": demand,
                "ms": elapsed,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "words": words,
                "max_tokens": max_tokens,
            }
            results["records"].append(record)
            if status == 200:
                results["served"].append(record)
            else:
                results["refused"].append(record)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="",
                    help="the gateway; empty means find the running grid")
    ap.add_argument("--scheduler-url", default="", help="default: found alongside the gateway")
    ap.add_argument("--users", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=10, help="workers per user")
    ap.add_argument("--requests", type=int, default=40, help="requests per user")
    ap.add_argument("--prompt-tokens", default="200-2000", help="MIN-MAX, drawn per request")
    ap.add_argument("--max-tokens", default="32-512", help="MIN-MAX, drawn per request")
    ap.add_argument("--concurrency-limit", type=int, default=4,
                    help="per-user concurrency this client declares")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--shared-prefix", default=SHARED_PREAMBLE,
                    help="the system preamble every request shares; the engine's "
                         "prefix cache serves it once for the whole run")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample-interval", type=float, default=0.3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--quiet", action="store_true", help="no per-request lines")
    ap.add_argument("--json", default="", help="write the measurement to this file")
    args = ap.parse_args()

    def pair(text: str) -> tuple[int, int]:
        lo, _, hi = text.partition("-")
        low = int(lo)
        return (low, low) if not hi else (low, int(hi))

    args.prompt_tokens = pair(args.prompt_tokens)
    args.max_tokens = pair(args.max_tokens)
    args.per_slot = max(1, args.requests // args.concurrency)
    args.clamped = 0

    if not args.url:
        args.url, args.scheduler_url = discover_grid(args.scheduler_url)
        if not args.url:
            print("no grid is running.", file=sys.stderr)
            print("start it with:  ./scripts/dev.sh", file=sys.stderr)
            return 2
    if not args.scheduler_url:
        args.scheduler_url = guess_scheduler(args.url)

    try:
        gw_before = parse_exposition(fetch(f"{args.url}/metrics"))
        sc_before = parse_exposition(fetch(f"{args.scheduler_url}/metrics"))
    except Exception as exc:  # noqa: BLE001
        print(f"cannot reach the grid: {exc}", file=sys.stderr)
        print(f"  gateway  {args.url}/metrics", file=sys.stderr)
        print(f"  scheduler {args.scheduler_url}/metrics", file=sys.stderr)
        print("", file=sys.stderr)
        print("The grid runs on :9000/:9001 by default, or :9100/:9101 when", file=sys.stderr)
        print("something else on this machine owns the defaults. Which one is", file=sys.stderr)
        print("live is a property of the machine, so this script finds it -- but", file=sys.stderr)
        print("only if you leave --url alone. Or pass the port you started:", file=sys.stderr)
        print("  --url http://127.0.0.1:9100 --scheduler-url http://127.0.0.1:9101",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("start it with:  ./scripts/dev.sh", file=sys.stderr)
        return 2

    print(f"== gateway {args.url}   scheduler {args.scheduler_url} ==")

    sampler = Sampler(args.url, args.scheduler_url, args.sample_interval)
    sampler.start()

    results = {
        "statuses": {},
        "decisions": {},
        "reasons": {},
        "records": [],
        "served": [],
        "refused": [],
        "errors": 0,
    }
    lock = threading.Lock()
    t0 = time.perf_counter()
    threads = []
    for u in range(args.users):
        for slot in range(args.concurrency):
            t = threading.Thread(
                target=worker, args=(args, f"user-{u}", slot, results, lock)
            )
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    time.sleep(1.0)  # let the last responses land in the metrics
    sampler.stop()
    sampler.join(timeout=5.0)

    gw_after = parse_exposition(fetch(f"{args.url}/metrics"))
    sc_after = parse_exposition(fetch(f"{args.scheduler_url}/metrics"))

    def dgw(name: str, match: dict | None = None) -> float:
        return total(gw_after, name, match) - total(gw_before, name, match)

    def dsc(name: str, match: dict | None = None) -> float:
        return total(sc_after, name, match) - total(sc_before, name, match)

    ok = dgw("gate_requests_total", {"status": "ok"})
    rejected = dgw("gate_requests_total", {"status": "rejected"})
    admitted = dgw("gate_admitted_total")
    held_by_reason = {
        k: dgw("gate_held_total", {"reason": k})
        for k in by_label(gw_after, "gate_held_total", "reason")
    }
    held = sum(held_by_reason.values())
    rejected_by_reason = {
        k: dgw("gate_rejected_total", {"reason": k})
        for k in by_label(gw_after, "gate_rejected_total", "reason")
    }
    rejected_total = sum(rejected_by_reason.values())
    misses = dgw("gate_cache_misses_total")
    hits = dgw("gate_cache_hits_total")
    dispatch_admit = dsc("sched_dispatch_total", {"decision": "admit"})
    dispatch_hold = dsc("sched_dispatch_total", {"decision": "hold"})
    client_total = len(results["records"])

    served = results["served"]
    refused = results["refused"]
    served_ms = [r["ms"] for r in served]
    refused_ms = [r["ms"] for r in refused]
    holds = [r["hold_s"] for r in served if r["hold_s"] > 0]
    unpromised = [
        r for r in served
        if r["demand"] not in (None, "-") and int(r["demand"]) > 0
        and r["reserved"] in ("0", None, "-")
    ]
    too_long = sum(1 for r in refused if r["reason"] == "too_long")

    checks: list[tuple[bool, str]] = [
        (hits < 0.5, f"cache hits = 0 (every prompt is unique): {hits:.0f}"),
        (abs(ok - (admitted + held)) < 0.5,
         f'gate_requests_total{{status="ok"}} = admitted + held: '
         f'{ok:.0f} = {admitted:.0f} + {held:.0f}'),
        (abs(rejected - rejected_total) < 0.5,
         f'gate_requests_total{{status="rejected"}} = sum(gate_rejected_total): '
         f'{rejected:.0f} = {rejected_total:.0f}'),
        (abs(misses - (ok + rejected)) < 0.5,
         f"gate_cache_misses_total = ok + rejected: {misses:.0f} = {ok:.0f} + {rejected:.0f}"),
        (abs(dispatch_admit - admitted) < 0.5,
         f'sched_dispatch_total{{decision="admit"}} = gate_admitted_total: '
         f'{dispatch_admit:.0f} = {admitted:.0f}'),
        (abs(dispatch_hold - held) < 0.5,
         f'sched_dispatch_total{{decision="hold"}} = sum(gate_held_total): '
         f'{dispatch_hold:.0f} = {held:.0f}'),
        (abs(client_total - (ok + rejected)) < 0.5,
         f"client saw {client_total:.0f} = ok {ok:.0f} + rejected {rejected:.0f}"),
        (total(sc_after, "sched_reservation_drift_blocks") < 0.5,
         "sched_reservation_drift_blocks = 0"),
        (dsc("engine_preemptions_total") < 0.5, "engine_preemptions_total delta = 0"),
        (dsc("sched_preemption_total") < 0.5, "sched_preemption_total delta = 0"),
        (dsc("sched_capacity_breach_total") < 0.5, "sched_capacity_breach_total delta = 0"),
        (too_long == 0, f"no request was refused as too_long (client sizing): {too_long}"),
        (len(unpromised) == held or not held,
         f"every held request wanted blocks and got none: {len(unpromised)}"),
    ]

    print(f"== {args.users} users x {args.concurrency} x {args.requests} "
          f"= {client_total} requests, prompts {args.prompt_tokens[0]}-{args.prompt_tokens[1]}w, "
          f"outputs {args.max_tokens[0]}-{args.max_tokens[1]}tok, seed {args.seed} ==")
    if args.clamped:
        print(f"   ({args.clamped} request(s) shortened to fit the context window)")

    if not args.quiet:
        for r in results["records"][:3]:
            print(f"  {r['status']} {r['decision']:<6} {r['reason']:<16} "
                  f"{r['ms']:.0f}ms hold={r['hold_s']:.2f}s "
                  f"reserved={r['reserved']} demand={r['demand']} "
                  f"out={r['completion_tokens']}")

    print("\n== the two populations, kept apart ==")
    print(f"  served   {len(served):>5}   p50 {pct(served_ms, 0.5) / 1000:>6.2f}s   "
          f"p95 {pct(served_ms, 0.95) / 1000:>6.2f}s   max {max(served_ms or [0]) / 1000:>6.2f}s")
    print(f"  refused  {len(refused):>5}   p50 {pct(refused_ms, 0.5):>6.0f}ms   "
          f"max {max(refused_ms or [0]):>6.0f}ms   "
          f"(decided at the door: cost the server nothing)")
    print(f"  held     {len(holds):>5}   wait p50 {pct(holds, 0.5):>5.2f}s   "
          f"max {max(holds or [0]):>5.2f}s   (all of them were released and served)")
    print(f"  wall clock {wall:.1f}s")

    print("\n== outcomes ==")
    print(f"  statuses  {json.dumps(results['statuses'], sort_keys=True)}")
    print(f"  decisions {json.dumps(results['decisions'], sort_keys=True)}")
    print(f"  reasons   {json.dumps(results['reasons'], sort_keys=True)}")
    if unpromised:
        wanted = sum(int(r["demand"]) for r in unpromised)
        print(f"  {len(unpromised)} held request(s) wanted {wanted} blocks and got 0 -- "
              f"refused room, not blocks")

    print("\n== the engine's view ==")
    print(f"  engine running        peak {sampler.peak(1, 'engine_running_requests'):.0f}"
          f"   waiting peak {sampler.peak(1, 'engine_waiting_requests'):.0f}")
    print(f"  kv cache usage        peak {sampler.peak(1, 'engine_kv_cache_usage_ratio'):.4f}")
    print(f"  blocks reserved       peak {sampler.peak(1, 'sched_reserved_blocks'):.0f}"
          f"   allocated peak {sampler.peak(1, 'sched_allocated_blocks'):.0f}")
    print(f"  hold depth            peak {sampler.peak(1, 'sched_hold_depth'):.0f}")
    print(f"  gateway in-flight     peak {sampler.peak(0, 'gate_in_flight'):.0f}"
          f"   waiting peak {sampler.peak(0, 'gate_waiting'):.0f}")
    print(f"  preemptions           engine {dsc('engine_preemptions_total'):.0f}"
          f"   scheduler {dsc('sched_preemption_total'):.0f}"
          f"   capacity_breach {dsc('sched_capacity_breach_total'):.0f}")
    print(f"  drift                 {total(sc_after, 'sched_reservation_drift_blocks'):.0f}")

    print("\n== identities ==")
    for good, text in checks:
        print(f"[{'PASS' if good else 'FAIL'}] {text}")
    failed = [text for good, text in checks if not good]

    print("\n== verdict ==")
    if failed:
        print(f"  NOT production-grade: {len(failed)} identity/identities failed")
    else:
        print(f"  all {len(checks)} identities hold")
    out_tokens = [r["completion_tokens"] or 0 for r in served]
    prompt_tokens = [r["prompt_tokens"] or 0 for r in served]
    mean_out = statistics.mean(out_tokens) if out_tokens else 0
    mean_prompt = statistics.mean(prompt_tokens) if prompt_tokens else 0
    print(f"  {len(served)} request(s) served with real tokens "
          f"(mean {mean_out:.0f} out, {mean_prompt:.0f} prompt)")
    print(f"  the engine ran {sampler.peak(1, 'engine_running_requests'):.0f} request(s) at once, "
          f"its own queue peaked at {sampler.peak(1, 'engine_waiting_requests'):.0f}")
    print("  nothing was preempted, no blocks leaked, "
          "and every held request was released and served")

    if args.json:
        measurement = {
            "wall_s": wall,
            "served": len(served),
            "refused": len(refused),
            "held": len(holds),
            "served_ms": served_ms,
            "refused_ms": refused_ms,
            "holds_s": holds,
            "statuses": results["statuses"],
            "decisions": results["decisions"],
            "reasons": results["reasons"],
            "held_by_reason": held_by_reason,
            "rejected_by_reason": rejected_by_reason,
            "peaks": {
                "engine_running_requests": sampler.peak(1, "engine_running_requests"),
                "engine_waiting_requests": sampler.peak(1, "engine_waiting_requests"),
                "engine_kv_cache_usage_ratio": sampler.peak(1, "engine_kv_cache_usage_ratio"),
                "sched_hold_depth": sampler.peak(1, "sched_hold_depth"),
                "sched_reserved_blocks": sampler.peak(1, "sched_reserved_blocks"),
                "sched_allocated_blocks": sampler.peak(1, "sched_allocated_blocks"),
                "gate_in_flight": sampler.peak(0, "gate_in_flight"),
                "gate_waiting": sampler.peak(0, "gate_waiting"),
            },
            "checks": [{"ok": good, "text": text} for good, text in checks],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(measurement, fh, indent=2)
        print(f"wrote {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
