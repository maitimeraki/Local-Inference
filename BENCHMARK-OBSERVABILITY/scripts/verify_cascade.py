#!/usr/bin/env python3
"""Check the cascade identities the README quotes, against a live grid.

Drives the same spike the README describes (24 users x 10 concurrent x 40
requests, 2000-token prompts, 32 output tokens, `concurrency_limit: 4`),
samples both tiers' /metrics while the burst runs, and then checks that the
counters balance. Exit code 0 means every identity held.

    # Windows
    uv run --no-sync python scripts/verify_cascade.py \
        --gateway http://localhost:9000 --scheduler http://localhost:9001

    # inside the VM that runs the gateway -- the documented way to keep
    # Prometheus's scrape of the gateway up during a burst
    /home/anupam/.venv/bin/python scripts/verify_cascade.py

Standard library only, so it runs anywhere Python does -- including the WSL2
image, which has no node and therefore cannot run scripts/loadgen.mjs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request

_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def fetch(url: str, timeout: float = 15.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def parse(text: str) -> dict[str, dict[tuple, float]]:
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
    """Sum every series of `name` whose labels contain `match`."""
    found = 0.0
    for labels, value in metrics.get(name, {}).items():
        if match and not all(dict(labels).get(k) == v for k, v in match.items()):
            continue
        found += value
    return found


def by_label(metrics: dict, name: str, label: str) -> dict[str, float]:
    """{label value: summed metric} for every series of `name`."""
    out: dict[str, float] = {}
    for labels, value in metrics.get(name, {}).items():
        key = dict(labels).get(label, "")
        out[key] = out.get(key, 0.0) + value
    return out


class Sampler(threading.Thread):
    """Polls both tiers' /metrics and keeps the peaks."""

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
                gw = parse(fetch(f"{self.gateway}/metrics"))
            except Exception:
                gw = {}
            try:
                sc = parse(fetch(f"{self.scheduler}/metrics"))
            except Exception:
                sc = {}
            self.samples.append((time.perf_counter() - t0, gw, sc))
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()


def peak(samples, tier: int, name: str) -> float:
    """Highest value a metric reached across the burst (0.0 if never seen)."""
    best = 0.0
    for row in samples:
        metrics = row[1] if tier == 0 else row[2]
        value = total(metrics, name)
        if value > best:
            best = value
    return best


def trough(samples, tier: int, name: str) -> float:
    """Lowest value a metric reached across the burst."""
    seen = [total(row[1] if tier == 0 else row[2], name) for row in samples]
    return min(seen) if seen else 0.0


def burst(args, user: str, results: dict, lock: threading.Lock) -> None:
    """One user's worth of sequential requests on a worker thread."""
    headers = {"content-type": "application/json", "x-user-id": user}
    for seq in range(args.per_slot):
        # A fresh nonce per request: the gateway's exact-match cache must never
        # answer one, or the engine is never touched and the run proves nothing.
        #
        # The nonce is the *end* of the prompt, so the body still shares one
        # long prefix and the engine's prefix cache serves them all from the
        # same blocks. That is the default because it is the realistic case:
        # a fleet of users asking similar things. `--distinct-prompts` breaks
        # the prefix too, which is what actually exercises the KV pool.
        body = {
            "model": "local",
            "prompt": ("word " * args.prompt_tokens).strip()
            + f" [nonce {user}-{seq}-{time.time_ns()}]",
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "concurrency_limit": args.concurrency_limit,
        }
        if args.distinct_prompts:
            body["prompt"] = f"seed {time.time_ns()} " + body["prompt"]
        payload = json.dumps(body).encode()
        started = time.perf_counter()
        req = urllib.request.Request(
            args.gateway + "/v1/completions", data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                resp.read()
                status, hdrs = resp.status, resp.headers
        except urllib.error.HTTPError as exc:
            exc.read()
            status, hdrs = exc.code, exc.headers
        except Exception as exc:  # noqa: BLE001 - the summary must not lose a request
            with lock:
                results["errors"] += 1
            print(f"  request error: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        elapsed = (time.perf_counter() - started) * 1000.0
        with lock:
            results["statuses"][status] = results["statuses"].get(status, 0) + 1
            decision = hdrs.get("x-scheduler-decision") or "-"
            reason = hdrs.get("x-scheduler-reason") or "-"
            results["decisions"][decision] = results["decisions"].get(decision, 0) + 1
            results["reasons"][reason] = results["reasons"].get(reason, 0) + 1
            results["samples"].append(
                {
                    "status": status,
                    "decision": decision,
                    "reason": reason,
                    "hold_s": float(hdrs.get("x-scheduler-hold-time") or 0.0),
                    "reserved": hdrs.get("x-kv-blocks-reserved") or "-",
                    "demand": hdrs.get("x-kv-blocks-demand") or "-",
                    "ms": elapsed,
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:9000")
    ap.add_argument("--scheduler", default="http://localhost:9001")
    ap.add_argument("--users", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--requests", type=int, default=40, help="requests per user")
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--concurrency-limit", type=int, default=4)
    ap.add_argument("--sample-interval", type=float, default=0.3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--json", default="", help="write the measurement to this file")
    ap.add_argument(
        "--distinct-prompts",
        action="store_true",
        help="make every prompt unique in bulk, not just by nonce. The default "
             "prompts share a long prefix, so the engine's prefix cache serves "
             "them from one set of blocks and the KV pool is never exercised. "
             "This flag measures the pool under no sharing at all.",
    )
    args = ap.parse_args()

    args.per_slot = max(1, args.requests // args.concurrency)

    try:
        gw_before = parse(fetch(f"{args.gateway}/metrics"))
        sc_before = parse(fetch(f"{args.scheduler}/metrics"))
    except Exception as exc:  # noqa: BLE001
        print(f"cannot reach the grid: {exc}", file=sys.stderr)
        print(f"is the gateway up at {args.gateway}?", file=sys.stderr)
        return 2

    sampler = Sampler(args.gateway, args.scheduler, args.sample_interval)
    sampler.start()

    results = {
        "statuses": {},
        "decisions": {},
        "reasons": {},
        "samples": [],
        "errors": 0,
    }
    lock = threading.Lock()
    t0 = time.perf_counter()
    threads = []
    for u in range(args.users):
        for _ in range(args.concurrency):
            t = threading.Thread(target=burst, args=(args, f"user-{u}", results, lock))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    time.sleep(1.0)  # let the last responses land in the metrics
    sampler.stop()
    sampler.join(timeout=5.0)

    gw_after = parse(fetch(f"{args.gateway}/metrics"))
    sc_after = parse(fetch(f"{args.scheduler}/metrics"))

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

    client_total = sum(results["statuses"].values())

    checks: list[tuple[bool, str]] = [
        (hits < 0.5,
         f"gate_cache_hits_total delta = 0 (every prompt is unique): {hits:.0f}"),
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
    ]

    print(f"== burst: {args.users} users x {args.concurrency} x {args.requests} "
          f"= {client_total} requests in {elapsed:.1f}s ==")
    print("statuses ", json.dumps(results["statuses"], sort_keys=True))
    print("decisions", json.dumps(results["decisions"], sort_keys=True))
    print("reasons  ", json.dumps(results["reasons"], sort_keys=True))

    print("\n== counters (delta over the burst) ==")
    print(f"gate_admitted_total                            {admitted:.0f}")
    for reason, value in sorted(held_by_reason.items()):
        if value:
            pad = " " * max(1, 24 - len(reason))
            print(f'gate_held_total{{reason="{reason}"}}{pad} {value:.0f}')
    for reason, value in sorted(rejected_by_reason.items()):
        if value:
            pad = " " * max(1, 21 - len(reason))
            print(f'gate_rejected_total{{reason="{reason}"}}{pad} {value:.0f}')
    print(f'gate_requests_total{{status="ok"}}                {ok:.0f}')
    print(f'gate_requests_total{{status="rejected"}}          {rejected:.0f}')
    print(f"gate_cache_misses_total                         {misses:.0f}")
    print(f"gate_cache_hits_total                           {hits:.0f}")

    print("\n== identities ==")
    for good, text in checks:
        print(f"[{'PASS' if good else 'FAIL'}] {text}")

    print("\n== peaks ==")
    peaks = [
        ("gate_in_flight", peak(sampler.samples, 0, "gate_in_flight")),
        ("gate_waiting", peak(sampler.samples, 0, "gate_waiting")),
        ("sched_user_inflight", peak(sampler.samples, 1, "sched_user_inflight")),
        ("sched_hold_depth", peak(sampler.samples, 1, "sched_hold_depth")),
        ("sched_reserved_blocks", peak(sampler.samples, 1, "sched_reserved_blocks")),
        ("sched_allocated_blocks", peak(sampler.samples, 1, "sched_allocated_blocks")),
        ("engine_running_requests", peak(sampler.samples, 1, "engine_running_requests")),
        ("engine_waiting_requests", peak(sampler.samples, 1, "engine_waiting_requests")),
        ("engine_kv_cache_usage_ratio", peak(sampler.samples, 1, "engine_kv_cache_usage_ratio")),
    ]
    for name, value in peaks:
        print(f"{name:<32} {value:.4f}")
    print(f'{"sched_free_blocks (min)":<32} {trough(sampler.samples, 1, "sched_free_blocks"):.0f}')
    print(f'{"sched_total_blocks":<32} {total(sc_after, "sched_total_blocks"):.0f}')

    print("\n== one-second trace ==")
    print("t  gate_in gate_wait hold resv alloc free eng_run eng_wait kv")
    buckets: dict[int, list] = {}
    for t, gw, sc in sampler.samples:
        buckets.setdefault(int(t), []).append((gw, sc))
    for second in sorted(buckets):
        gw, sc = buckets[second][-1]
        print(
            f"{second:>2} "
            f"{total(gw, 'gate_in_flight'):>7.0f} "
            f"{total(gw, 'gate_waiting'):>9.0f} "
            f"{total(sc, 'sched_hold_depth'):>4.0f} "
            f"{total(sc, 'sched_reserved_blocks'):>4.0f} "
            f"{total(sc, 'sched_allocated_blocks'):>5.0f} "
            f"{total(sc, 'sched_free_blocks'):>4.0f} "
            f"{total(sc, 'engine_running_requests'):>8.0f} "
            f"{total(sc, 'engine_waiting_requests'):>8.0f} "
            f"{total(sc, 'engine_kv_cache_usage_ratio'):>6.4f}"
        )

    print("\n== sample request lines ==")
    for wanted in ("admit", "hold", "reject"):
        for s in results["samples"]:
            if (wanted == "reject" and s["status"] == 429) or (
                wanted != "reject" and s["decision"] == wanted
            ):
                print(
                    f"  status={s['status']} decision={s['decision']} reason={s['reason']} "
                    f"hold={s['hold_s']:.2f}s reserved={s['reserved']} demand={s['demand']} "
                    f"{s['ms']:.0f}ms"
                )
                break

    failed = [text for good, text in checks if not good]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} identities hold")

    if args.json:
        measurement = {
            "counters": {
                "admitted": admitted,
                "held": held,
                "held_by_reason": held_by_reason,
                "rejected": rejected,
                "rejected_by_reason": rejected_by_reason,
                "ok": ok,
                "cache_misses": misses,
            },
            "client": {
                "statuses": results["statuses"],
                "decisions": results["decisions"],
                "reasons": results["reasons"],
                "elapsed_s": elapsed,
            },
            "peaks": {name: value for name, value in peaks},
            "checks": [{"ok": good, "text": text} for good, text in checks],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(measurement, fh, indent=2)
        print(f"wrote {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
