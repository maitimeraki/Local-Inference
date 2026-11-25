# vLLM Inference Platform — Plan

**What this is:** a benchmarking and observability platform for vLLM, run **white-box** — vLLM
imported as a library inside WSL2, not hidden behind a container. You should be able to watch the
scheduler hand out KV-cache blocks, iteration by iteration.

**Scope:** vLLM only. SGLang is a config field, never a build item.
**Status:** built. All nine phases in §5 shipped. This file is the plan of record —
[README.md](README.md) is how you run it, and [PRODUCTION-DESIGN.md](PRODUCTION-DESIGN.md)
covers the production grid (gateway → scheduler → in-process engine) layered on top.
§1's "current state" is a 2026-10-15 snapshot and is kept as it was written.

---

## 1. Current state (verified 2026-10-15)

| Item | Value | Consequence |
|---|---|---|
| GPU | RTX 3050 Laptop, **6 GB VRAM** | Models ~0.5–1.5B; concurrency is VRAM-bound |
| OS | Windows 11 + WSL2 Ubuntu (default v2) | vLLM runs here |
| Docker | 28.4.0, nvidia runtime registered | Observability stack only |
| Python | 3.10.10 (Windows), uv 0.10.9 | uv workspace for everything |
| vLLM | not installed | Phase 0 installs it |
| Repo conventions | FastAPI, Pydantic v2, structlog, pytest, ruff | This platform matches them |

Everything below is sized to these numbers, not to an ideal machine.

---

## 2. Decisions

1. **vLLM from imports in WSL2, not Docker.** Docker made it a black box: you could POST to it but
   never read `Scheduler.schedule()`.
2. **Observability stack runs in compose.** Prometheus, Grafana, Tempo, pushgateway, gpu-exporter.
   Hiding Grafana costs you nothing.
3. **Model: Qwen2.5-1.5B-Instruct.** Fits 6 GB with KV-cache headroom for real batching.
4. **Three run modes** — offline, step-through, server. The step-through mode is the point.
5. **A gateway** (router + admission control + exact-match cache) sits in front. It is where arrival
   rate, queueing and cache effects become measurable; two vLLM configs can be compared under one
   identical call pattern instead of sequentially.

---

## 3. Layout

```
LLM-INFERENCE/
└── BENCHMARK-OBSERVABILITY/
    ├── PLAN.md                      ← this file
    ├── README.md
    ├── pyproject.toml               ← uv workspace: inferbench, infergw, inferobs
    ├── config/
    │   ├── workloads.yaml           ← the 6 workload profiles
    │   ├── experiments.yaml         ← the 6 experiments as an axis matrix
    │   ├── engine/{baseline,optimized}.yaml   ← EngineArgs as data
    │   ├── gateway.yaml
    │   ├── pricing.yaml             ← cost model inputs
    │   └── slo_targets.yaml
    ├── benchmarking/                ← PROJECT 1 (the foundation)
    │   └── src/inferbench/
    │       ├── cli.py               bench | inspect | serve | report | slo
    │       ├── config.py            pydantic models for all yaml
    │       ├── engine.py       ★    build_engine(config) → LLM / AsyncLLMEngine
    │       ├── instrument.py   ★    custom StatLogger: scheduler + KV-cache tap
    │       ├── inspect.py      ★    step-by-step engine anatomist
    │       ├── server.py       ★    imported ASGI app under uvicorn
    │       ├── workloads.py         profiles → prompts at exact token lengths
    │       ├── client.py            httpx SSE client, per-request metrics
    │       ├── metrics.py           TTFT / ITL / TPOT math + percentiles
    │       ├── runner.py            experiment cells
    │       ├── collectors.py        pynvml GPU poll
    │       ├── store.py             sqlite3: runs + requests
    │       ├── cost.py              $/1M tokens, $/1k requests
    │       └── report.py            tables, plots, Grafana JSON
    ├── gateway/                     ← router + cache
    │   └── src/infergw/{app,router,admission,cache,metrics}.py
    ├── observability/               ← PROJECT 2
    │   ├── src/inferobs/{tracing,metric_names,slo,burn}.py   ← shared library
    │   ├── compose.yaml
    │   ├── prometheus/prometheus.yml
    │   ├── tempo/tempo.yaml
    │   ├── grafana/{provisioning,dashboards}/
    │   └── alerts/slo_rules.yml
    ├── deploy/k8s/                  ← kustomize + helm values (deployment path)
    ├── scripts/{bootstrap_wsl.sh,dev.sh}
    ├── results/                     ← gitignored
    └── tests/                       ← GPU-free, against fakes
```

Modules marked ★ are what make this white-box. Everything else is standard harness plumbing.

---

## 4. Run modes

| Mode | Import | What you see |
|---|---|---|
| **offline** | `from vllm import LLM` | batch `generate()`, per-request stat objects |
| **step-through** ★ | `LLMEngine` / `EngineCore` | every decoding iteration — scheduler output, prefill vs decode batch, KV blocks, preemptions |
| **server** | `vllm.entrypoints.openai.api_server.build_app` | the OpenAI API, as an ASGI app you own and wrap |

Step-through output looks like this, and this is the whole reason for the change:

```
step 0007 | sched: 2 new + 5 running | tok 4096 | kv 38% (612/1600 blk) | 41ms
step 0008 | sched: 0 new + 7 running | tok  512 | kv 41% (658/1600 blk) | 39ms
step 0009 | preempted 1  (kv 94% full)    ← memory pressure, made visible
```

What this exposes: scheduler decisions (scheduled tokens, prefill/decode split, preemptions), KV
cache (blocks used/free, utilisation, per-seq block count), prefix cache (hit rate, cached blocks),
per-request phases (queue, prefill, decode, first-token — read in-process, not scraped), engine
health (running vs waiting, iterations/sec, peak concurrency).

---

## 5. Steps (phases)

Each phase is a gate, not a milestone. Do not start the next until the gate passes.

| # | Step | Gate |
|---|---|---|
| 0 | `bootstrap_wsl.sh`: uv + torch(cu126) + vLLM into WSL2; `import vllm`; `nvidia-smi` | GPU visible in WSL2, one `LLM.generate()` returns text |
| 1 | `engine.py` + `instrument.py` + `inspect.py` | you can watch a preemption happen |
| 2 | `metrics.py` + `store.py` | our TTFT/ITL match `vllm bench serve` within noise |
| 3 | `server.py` (imported ASGI) + `client.py` | TTFT measured by our own client |
| 4 | `gateway/`: router, admission, cache | 429s appear under saturation; cache hits > 0 |
| 5 | collectors + Prometheus + Grafana + Tempo | one request shows a full phase trace |
| 6 | experiment matrix + `cost.py` + comparison dashboard | answers "which config wins, at what cost" |
| 7 | SLOs + burn-rate alerts | alert fires on an induced regression |
| 8 | `deploy/k8s/` | clean `kubectl apply` on kind |

Phases 0–3 are the foundation and are useful on their own. Phases 4–8 build on top without
changing them.

All nine shipped. The production grid — a gateway that forwards to a scheduler that owns
the KV-block ledger and the GPU — was designed afterwards and is tracked separately in
[PRODUCTION-DESIGN.md](PRODUCTION-DESIGN.md) §16.

---

## 6. Process conventions

- **uv workspace**, four members, one lockfile. `uv sync --all-packages --inexact` sets up
  everything (`--all-packages` is required, or the members and their `inferbench`/`infergw` scripts
  never install; add `--all-extras` for vllm and the OTel SDK).
- **Pydantic-settings** for config — no scattered `os.environ` reads.
- **structlog** JSON logs, matching the existing `LLM/` service.
- **ruff** lint + format. **pytest** with **GPU-free fakes** — the inspector is tested against a
  fake scheduler, the client against a canned SSE stream. CI stays green without a GPU.
- **Every run row records** model, dtype, `max_model_len`, sampling params, engine flags, vLLM
  version, GPU, git SHA — so a comparison is auditable months later.

### Reproducibility rules (correctness, not polish)

- Every workload pins `max_tokens = min_tokens = N` and `ignore_eos=True`. Without this, output
  lengths drift and run-to-run variance is just output-length noise.
- Held constant across all cells: model, dtype, `max_model_len`, temperature 0, fixed seed.
- `repeats: 3` on every cell, and report variance — a laptop GPU thermally throttles, and a single
  run will lie to you.

---

## 7. Constraints and risks

| Risk | Handling |
|---|---|
| Repo on Windows (`/mnt/c`) is slow; HF cache would land there | Code stays on `/mnt/c`; venv, `HF_HOME` and results live WSL2-native in `~/` |
| V1 runs `EngineCore` in a separate process — hides internals, breaks debugger | `VLLM_ENABLE_V1_MULTIPROCESSING=0` in every mode. Non-negotiable for white-box. |
| 6 GB VRAM | `--gpu-memory-utilization 0.80 --max-model-len 4096 --enforce-eager` |
| V1 may refuse to start on 6 GB | Lower `--gpu-memory-utilization` or `--max-model-len`. There is no V0 to fall back to — vLLM deleted it in 0.11.0. |
| Concurrency 32 will hit a wall on 6 GB | That wall is a finding, not a failure. Report the ceiling. |
| Thermal throttling over long runs | covered by `repeats: 3` + variance reporting |
| Wanting to read/modify vLLM itself | `bootstrap_wsl.sh --from-source` installs vLLM editable from a git clone |

---

## 8. Out of scope

SGLang (config field only). Semantic/embedding cache (exact-match first). Multi-region.
OpenTelemetry beyond traces.

---

## 9. Open items

1. **"Cogs"** — read as **caching** (gateway exact-match cache + prefix-cache measurement).
   Confirm, or say what was meant.
2. **Observability stack** — compose (default) or native WSL2 binaries?
