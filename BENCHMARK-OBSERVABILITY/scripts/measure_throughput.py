#!/usr/bin/env python3
"""Measure the engine's steady-state throughput under a fixed offered load.

The burst in `verify_cascade.py` answers "how many requests got through the
door", which mixes admission with throughput and is not a steady-state
number: the offered load decays as each user's requests run out. This holds
the load open instead and counts what completes.

    # inside the VM that runs the gateway
    /home/anupam/.venv/bin/python scripts/measure_throughput.py --seconds 25

Offered load is 24 users x 4 workers, because 4 is the most the gateway will
hold for one user at `concurrency_limit: 4`; a worker that wants more is
refused with a 429, which measures the gateway rather than the engine.

Standard library only, so it runs anywhere Python does -- including the WSL2
image, which has no node and therefore cannot run scripts/loadgen.mjs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:9000")
    ap.add_argument("--users", type=int, default=24)
    ap.add_argument("--workers-per-user", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--concurrency-limit", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=3.0)
    args = ap.parse_args()

    stop = threading.Event()
    lock = threading.Lock()
    lat: list[float] = []
    counts = {"200": 0, "429": 0, "err": 0}
    token_box = [0]

    def worker(user: str, slot: int) -> None:
        headers = {"content-type": "application/json", "x-user-id": user}
        local: list[float] = []
        seq = 0
        while not stop.is_set():
            seq += 1
            # A fresh nonce per request: the gateway's exact-match cache must
            # never answer one, or the engine is never touched and the run
            # measures nothing.
            body = json.dumps({
                "model": "local",
                "prompt": ("word " * args.prompt_tokens).strip()
                          + f" [nonce {user}-{slot}-{seq}-{time.time_ns()}]",
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "concurrency_limit": args.concurrency_limit,
            }).encode()
            req = urllib.request.Request(
                args.gateway + "/v1/completions", data=body, headers=headers, method="POST"
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp.read()
                    status = resp.status
            except urllib.error.HTTPError as exc:
                exc.read()
                status = exc.code
            except Exception:  # noqa: BLE001 - a dropped request is a datapoint
                with lock:
                    counts["err"] += 1
                continue
            dt = time.perf_counter() - t0
            with lock:
                counts[str(status)] = counts.get(str(status), 0) + 1
                if status == 200:
                    local.append(dt)
                    token_box[0] += args.max_tokens
        with lock:
            lat.extend(local)

    threads = []
    for u in range(args.users):
        for s in range(args.workers_per_user):
            t = threading.Thread(target=worker, args=(f"user-{u}", s), daemon=True)
            t.start()
            threads.append(t)

    # Let the engine reach steady state before the clock starts.
    time.sleep(args.warmup)
    with lock:
        counts.update({"200": 0, "429": 0, "err": 0})
        lat.clear()
        token_box[0] = 0

    t0 = time.perf_counter()
    time.sleep(args.seconds)
    elapsed = time.perf_counter() - t0
    stop.set()
    for t in threads:
        t.join(timeout=15)

    ok = counts["200"]
    print(f"offered        {args.users * args.workers_per_user} concurrent")
    print(f"completed      {ok} in {elapsed:.1f}s")
    print(f"throughput     {ok / elapsed:.2f} req/s   {token_box[0] / elapsed:.0f} tok/s")
    print(f"rejected 429   {counts.get('429', 0)}   errors {counts.get('err', 0)}")
    if lat:
        s = sorted(lat)
        print(f"latency p50    {s[len(s) // 2]:.3f}s")
        print(f"latency p95    {s[int(len(s) * 0.95)]:.3f}s")
        print(f"latency p99    {s[int(len(s) * 0.99)]:.3f}s")
        print(f"latency mean   {statistics.mean(s):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
