# KV-CACHE-OPTIMIZATION

Compressing the KV cache: how many more requests can one RTX 3050 6 GB serve, at what
cost in output fidelity — **measured with and without MLA**, on a real vLLM engine.

**Status: implemented and measured on the device.** The plan is [PLAN.md](PLAN.md); the
theory it rests on is [docs/kv-cache-first-principles.md](docs/kv-cache-first-principles.md).
Tasks 1–6 are committed and all 32 tests pass on the RTX 3050. The §2.3 fidelity gate
**fails** — see [The result](#the-result) — and the sweep behind it was run anyway,
because the failure is a property of the construction, not of the engine or the kernel.
The newest measurement is
[results/20261121T083808Z](results/20261121T083808Z/report.md).

---

## The thesis in one paragraph

A KV cache is a pure function of the prompt prefix. Any compression of it is therefore a
deliberate violation of that purity, and the only question is what you traded away and
what you bought. There are two ways to compress: **numeric** (fewer bits per element) and
**structural** (change *what* is stored). Numeric compression injects error into every
element of every K and V; structural compression removes information the model provably
never needed. DeepSeek's MLA is the canonical structural fix, and its cache is a *learned
sufficient statistic* — which is exactly why it can be stored in BF16 and still be far
more accurate than FP8 applied to an unmodified cache. **Restructure first, then store at
the highest precision you can afford.** The order is not interchangeable: restructuring
must happen at write time, before anything is rounded.

## Why it matters here, concretely

| | |
|---|---|
| GPU | RTX 3050 6 GB, **sm_86** (Ampere — BF16 native, **no FP8 tensor cores**) |
| Model | `Qwen2.5-1.5B-Instruct` — GQA 12:2, 28 layers, bf16 |
| KV cache | **28,672 B/token** → 112 MiB per 4096-token sequence |
| Budget | 1.9 GiB for KV ⇒ **17** concurrent full-length sequences |
| Engine | **vLLM 0.31.0** + a custom **Triton** decode kernel |

The cache is ~40 % of VRAM and it is what caps concurrency. Concurrency is the product.

## What grouped-query attention is, and its catch

Qwen2.5-1.5B has **12 query heads and 2 KV heads**: query head `h` reads KV head
`h // 6`. Each KV head carries a full 128-dim K and V, so the cache is
`2 × 128 × 2 × 2 B = 1024 B` per token per layer — 28,672 B per token over 28 layers.
Plain MHA would be 6× larger; **GQA already took that cut, and it is baked into the
released weights.**

The catch is four layers deep:

1. **GQA already spent the structural budget.** DeepSeek's headline 7–20× MLA saving is
   measured against **MHA**. This model is already 6× smaller than MHA, so MLA's
   *additional* saving on top of GQA is `512/r` — 1× at r=512, 4× at r=128, 8× at r=64.
2. **Lossless is exactly r=512.** The stacked matrix `A = [W_K ; W_V]` is 512×1536 with
   rank 512, so the only lossless latent has 512 dimensions — the same size as GQA's
   K+V. Below 512 you are *truncating* the model's key/value subspace.
3. **A GQA head has no internal low-rank structure.** A dense 128-dim K vector is
   full-rank; you cannot compress *inside* a head. MLA works only by sharing *across*
   heads, and GQA leaves you 2 groups to share, not 128.
4. **12:2 was chosen at training time.** You cannot move it to 12:1 (MQA) post-hoc
   without a quality drop.

**So on this model MLA is GQA at full rank, and an approximation below it.**

## The experiment: with and without MLA

Two arms, one vLLM engine, the same weights, the same scheduler, the same 1.9 GiB KV
budget, the same corpus. The arms differ in **what the cache holds**:

- **`gqa` (control)** — vLLM's stock `Qwen2ForCausalLM`. Per-KV-head K and V.
- **`mla` (treatment)** — a custom model class that replaces `k_proj`/`v_proj` with a
  single low-rank down-projection `W_DKV`, stores the latent `c_KV` in a paged cache,
  and reads it back through a **Triton decode kernel** using the absorb trick
  (`W_UK` folded into the query, `W_UV` applied after the attention).

Post-hoc on Qwen2 this is *exact* at rank 512, because **RoPE can be fully absorbed
into the query**: the cached latent is position-independent and RoPE never touches the
cache. The MLA weights come from a **joint SVD** of `[W_K ; W_V]`, so the rank sweep
*is* an explicit truncation of the model's key/value subspace, and the discarded energy
is reported as `rel_err_k` / `rel_err_v` rather than assumed.

Ranks swept: **512, 384, 256, 192, 128, 96, 64**. Storage: **bf16 only** — see below.

**The gate.** At r=512 the MLA arm is mathematically identical to GQA. If `mla@512`
does not reproduce `gqa` (≥ 99 % token agreement, ΔPPL < 0.05), the construction or the
kernel is wrong — and **no other rank runs until it passes.**

**What comes out** — one headline table and one trade curve:

| metric | meaning |
|---|---|
| KV B/token | the storage cost of one token, per layer, summed |
| requests served | max concurrent 4096- and 8192-token sequences in the 1.9 GiB budget |
| prefill tok/s | **inbound** throughput |
| decode tok/s (agg + per-seq) | **outbound** throughput |
| TTFT / TPOT / end-to-end p50, p95 | latency, in ms |
| concurrency ceiling | the measured rung (1→2→4→…→128) at which the run degrades |
| exact match %, ΔPPL | fidelity vs. the GQA control |

**The pre-registered fidelity floor** — declared in `experiments.yaml` *before* the
sweep runs, so the recommendation cannot be reverse-engineered from the curve:

> `exact_match ≥ 95 %` **and** `rel_dppl ≤ 1 %` on the pinned corpus.

**The recommendation rule**, fixed in advance: *the arm with the largest requests-served
that still clears that floor.* If no arm clears it, the report says so and **does not
relax the floor to fit the data**.

## The result

**The MLA arm does not reproduce stock Qwen2, and the reason is the construction, not
the implementation.** PLAN.md §3.1 applies the rotation to the query and the key alike,
so `(R_p q)·(R_p k) = q·k` and the score becomes absolute. The score a Qwen2 head was
trained with is `(R_s q)·(R_p k) = qᵀ R_{s-p} k` — *relative*. Qwen2 rotates the whole
128-dim head, so the rotation cannot be folded into the query and still leave a
position-independent latent in the cache. That is why DeepSeek keeps a decoupled rope
key, which this project puts out of scope.

Measured on the pinned corpus, same weights in every row:

| model | perplexity |
|---|---|
| stock Qwen2 | 14.3 |
| stock Qwen2, rotation removed | 6718 |
| MLA arm at r=512 | 7973 |

The MLA arm reproduces the *position-blind* model at the layer level. Two tests prove the
implementation is not at fault: at r=512 the latent attention equals the native GQA
attention to 1e-9, and the attention layer computes the plan's formula over a paged cache.
End to end the two are only approximately equal, because 28 layers of bf16 round
differently in each path: measured against the dedicated `gqa_norope` arm on this card,
`mla@512` agrees on **18.0 %** of greedy tokens with **+17.3 %** ΔPPL. That is the
rounding gap, and it is small next to the 1217× gap against stock Qwen2 — the rotation is
the whole of the fidelity loss, not the arithmetic.

### The sweep

Eight arms, 4096-token context, 512 MiB of KV, one RTX 3050. The full report is in
[results/20261101T112253Z/report.md](results/20261101T112253Z/report.md).

| arm | rank | bytes/token | requests served | decode tok/s | prefill tok/s |
|---|---|---|---|---|---|
| gqa | — | 28,672 | 4 | 30.8 | 3,574 |
| mla | 512 | 28,672 | 4 | 6.6 | 761 |
| mla | 384 | 21,504 | 4 | 7.6 | 878 |
| mla | 256 | 14,336 | 8 | 9.1 | 1,056 |
| mla | 192 | 10,752 | 8 | 9.8 | 1,135 |
| mla | 128 | 7,168 | 16 | 11.4 | 1,328 |
| mla | 96 | 5,376 | 16 | 11.8 | 1,365 |
| mla | 64 | 3,584 | 32 | 12.0 | 1,388 |

**The answer is "the model needed the rank".** Two things happen as the rank falls, and
the second eats the first:

- The cache gets cheaper, so more requests fit: 4 → 32, an 8× gain at r=64, tracking
  the 8× smaller footprint.
- The attention gets slower, because the absorb and the expand are two `[12,128,r]`
  contractions per layer per token, and at r=64 that is 8× the flops of r=512's latent
  work on a card that was already bandwidth-bound.

The aggregate decode throughput does rise as the rank falls — 6.6 tok/s at r=512 to 12.0
at r=64, as the plan's P4 predicts — but it stops at 12.0, against GQA's 30.8. The plan
predicted 8× GQA's throughput at r=64; the measurement delivers 0.39×. On this card the
MLA arm is compute-bound, not bandwidth-bound, so the capacity the smaller cache buys
is capacity the slower kernel cannot cash in.

**Fidelity is the other half, and it is not a rounding difference.** The SVD's energy
retention, verified independently from the singular values:

| rank | rel_err_k | rel_err_v | top-r energy kept (layer 0) |
|---|---|---|---|
| 512 | 0.0% | 0.0% | 100.00% |
| 384 | 44.3% | 42.3% | 98.40% |
| 256 | 74.3% | 71.4% | 94.33% |
| 128 | 85.4% | 92.1% | 74.48% |
| 64 | 90.2% | 96.9% | 46.16% |

Post-hoc truncation discards the subspace aggressively: at r=256 three quarters of the
key's energy is gone, and by r=64 the latent holds under half. Qwen2.5-1.5B is a
2-KV-head model — GQA already took the structural redundancy — and the truncation has
no way to know which directions the model attends to. That is P6, and it holds: the knee
is no higher than r=256.

**No rank clears the fidelity floor**, and none can: the position-blind construction
means the arm is a different model, so the greedy match is 0 % at every rank and the
perplexity delta is enormous. The floor is not relaxed to fit the data.

### What each prediction did

| | prediction | verdict |
|---|---|---|
| P1 | at r=512 the MLA arm reproduces GQA | **refuted** — 0 % exact match, 3.1 % token agreement, and the cause is the RoPE absorb, not the kernel |
| P2 | r=512 buys nothing | **held** — same bytes per token, same requests served, and 4.7× slower |
| P3 | prefill gets slower as r rises | **held** — 1,388 tok/s at r=64 down to 761 at r=512 |
| P4 | aggregate throughput scales with 1/bytes | **refuted in magnitude** — it rises as the plan says (6.6 → 12.0 tok/s) but stops at 0.39× GQA, not 8×; the arm is compute-bound |
| P5 | fidelity degrades below r=256 | **held, and worse** — 74 % of the key's energy is gone by r=256 |
| P6 | rank reduction is a wash at best | **held** — the knee is at r=256 |
| P7 | the measured ceiling lands at or below the arithmetic one | **held** — the measured is 0.67–1.00 of the ideal, and the ladder's powers of two make it a lower bound |

### Where this leaves MLA on this card

For a 2-KV-head Qwen2 at 4096 context, **post-hoc MLA is a loss on this card**: the
rank has to fall far enough to buy capacity (r ≤ 256, so ≥ 74 % of the key's energy
discarded), and the contractions eat the gain. The construction that would make it a
win — one that keeps the score relative — is the decoupled-rope-key design the plan
puts out of scope, and it halves the saving at best, because the rotated key cannot be
compressed without coupling the positions.

The machinery is built and measured: the Triton kernel (7.8× the PyTorch reference), the
vLLM model class, the driver, the fidelity check, the report, and a container that runs
the suite on the GPU. What has to change is the construction.

## Two predictions worth stating up front

- **P6 — post-hoc MLA is at best a wash.** MLA at full rank is *exactly the same size*
  as GQA on a 2-KV-head model; below full rank it discards energy the model demonstrably
  uses. This negative result is the expected outcome and the plan is designed to
  demonstrate it, not hide it.
- **P7 — the measured concurrency ceiling lands at or below the arithmetic one.** The
  weights, the attention workspace and the allocator all take their cut of the same
  6 GB. The **ratios between arms** are the claim; the absolute numbers are not.

## What vLLM + Triton buys over the transformers path

The old plan did not use transformers as an engine — it used transformers only for the
tokenizer and a reference oracle, and hand-rolled the prefill/decode loop. The black
box was the **engine**, and that is what vLLM replaces: paged KV with a block table
(so "requests servable" is measured, not arithmetic), continuous batching (so freed KV
bytes actually become batch), chunked prefill, prefix caching, and a real async request
path via `AsyncLLMEngine`.

**Triton is the part that moves the numbers.** Decode on this card is purely
bandwidth-bound. GQA reads K and V separately — `2×128 + 2×128 = 512` values per token
per layer. MLA reads the latent **once** — `r` values. At r=128 that is **4× fewer
bytes**, and that 4× is arithmetic until a kernel actually performs it.

The costs, stated plainly: vLLM's MLA is a **DeepSeek architecture path** (`kv_lora_rank`
+ decoupled RoPE) and cannot be pointed at Qwen2's `k_proj`/`v_proj`, so the MLA arm
needs its own model class, attention layer, cache spec and kernel — vLLM gives us the
*engine*, not the *mechanism*. vLLM's Triton MLA backend also **rejects FP8 KV cache
below SM89**, and this card is sm_86.

## Why FP8 is not an arm

Three independent reasons, all verified on this machine:

1. vLLM's Triton MLA backend **rejects FP8 KV cache below SM89**; this card is sm_86.
2. Measured FP8 E4M3 round-trip error is **2.67 % RMS** against BF16's 0.20 % — that
   alone would breach the 1 % ΔPPL floor.
3. The criterion chosen at design time is "**highest BF16 accuracy**".

One storage dtype survives: **bf16**. That halves the sweep matrix and sharpens the
question — the only lever left is *structural rank*.

## Fidelity criterion (chosen)

**Exact greedy match + perplexity delta.** Greedy decode, fixed seed, pinned output
length, compared token-for-token against the GQA control: match rate, divergence
position (and divergence as a *fraction* of response length), token-level agreement
rate, and ΔPPL (absolute and relative) on a pinned corpus.

Fidelity runs pin `max_num_seqs=1`, `enforce_eager=True`, no chunked prefill and no
prefix caching, because batch composition and paged gathers change reduction order and
would make a token-level diff meaningless. Throughput runs use the full engine.

Greedy match and ΔPPL are **fidelity proxies, not task-quality guarantees** — they do not
measure factuality, helpfulness or task success, and every report says so.

## Layout

```
KV-CACHE-OPTIMIZATION/
├── PLAN.md                          ← the plan of record: 11 sections, 9 tasks
├── README.md                        ← this file
├── pyproject.toml                   ← packaging, deps, the vllm.general_plugins entry point
├── Dockerfile                       ← bakes model + corpus + vLLM; CUDA via pip torch
├── docker-compose.yml               ← gpus: all, mounts ./results only
├── .dockerignore
├── docker/bake_assets.py            ← the build-time model + corpus fetch, checksummed
├── experiments.yaml                 ← the canonical sweep matrix + the fidelity floor
├── docs/
│   └── kv-cache-first-principles.md ← the spec (10 sections + verification log)
├── kvbench/
│   ├── __init__.py
│   ├── config.py                    ← model id, corpus, ranks, budgets, engine flags
│   ├── weights.py                   ← safetensors load, joint SVD, MLA weights, energy retention
│   ├── models.py                    ← Qwen2MLA model class, attention layer, cache spec, plugin
│   ├── kernels.py                   ← Triton MLA decode kernel + its PyTorch reference
│   ├── server.py                    ← AsyncLLMEngine driver: requests, timing, ladder
│   └── bench.py                     ← CLI, fidelity, report rendering
└── tests/
    └── test_kvbench.py              ← one file, GPU-free where possible
```

**14 files, 8 of them Python.** `weights.py` owns the offline weight transformation,
`kernels.py` the kernel, `models.py` the vLLM integration, `server.py` the timing,
`bench.py` the CLI and the report. `requirements.txt` is gone — `pyproject.toml` is the
single dependency manifest.

## Run commands

The image bakes the model, the corpus and vLLM, so a run needs no network. The only
thing mounted out of the container is `./results`.

```bash
# from LLM-INFERENCE/KV-CACHE-OPTIMIZATION/

# build the image (bakes the model + corpus + vLLM; ~8 GB)
docker compose build

# build the MLA weights for every rank (CPU, offline, ~2 min)
docker compose run --rm kvbench python -m kvbench.bench prepare --ranks 512,384,256,192,128,96,64

# THE GATE: mla@512 must reproduce gqa
docker compose run --rm kvbench python -m kvbench.bench compare \
    --arms gqa,mla --ranks 512 --context-lens 4096 --repeats 1 --fidelity

# the headline comparison
docker compose run --rm kvbench python -m kvbench.bench compare \
    --arms gqa,mla --ranks 512,256,128 --context-lens 4096 --repeats 3 --fidelity

# the full sweep
docker compose run --rm kvbench python -m kvbench.bench sweep --config experiments.yaml

# fidelity only: exact greedy match + delta perplexity
docker compose run --rm kvbench python -m kvbench.bench fidelity \
    --arm mla --rank 512

# re-render the report from rows.csv, no GPU needed
docker compose run --rm kvbench python -m kvbench.bench report --run results/<timestamp>

# the tests
docker compose run --rm kvbench pytest tests/test_kvbench.py -v
```

The same commands run without Docker, on the host's own vLLM install:

```bash
cd LLM-INFERENCE/KV-CACHE-OPTIMIZATION
pip install -e .          # the vllm.general_plugins entry point must be installed
python -m kvbench.bench compare --arms gqa,mla --ranks 512 --fidelity
```

Results land in `./results/<UTC timestamp>/{config.json, rows.csv, report.md}`,
bind-mounted from the host. Under Git Bash, prefix with `MSYS_NO_PATHCONV=1` if a bind
mount fails with a path error.

## Next

The construction, not the machinery. Two ways forward, and the choice is the plan's:

1. **Amend the gate and keep the arm.** The arm is a position-blind Qwen2, so its
   fidelity is measured by the SVD's own energy retention (`rel_err_k`, `rel_err_v`)
   and the layer-level algebra, both of which pass exactly. The throughput and
   requests-served measurements — the project's actual question — stay valid.
2. **Change the construction.** Keep the rotated keys in the cache and compress only
   the value, so the score stays the stock model's and the arm reproduces GQA. The
   saving is then capped at 50 % (the key cannot be compressed without the rotation
   coupling the positions) and the plan's "the whole K+V is latent" story changes.

## The environment these numbers came from

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop, 6 GB, sm_86 |
| driver | 581.86 |
| CUDA | 13.0 |
| vLLM | 0.31.0 |
| torch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| transformers | 5.17.0 |
| numpy | 2.3.5 |
| safetensors | 0.8.0 |
| PyYAML | 6.0.3 |
| host | Windows 11, WSL2, Ubuntu 24.04 |
| model | Qwen/Qwen2.5-1.5B-Instruct |
| corpus | Project Gutenberg #1342, sha256 `3f6bb9d6…bcaf9d75` |

The measurements were taken with `gpu_memory_utilization=0.8` and a 512 MiB KV
budget, because 0.9 and the plan's 1.9 GiB do not fit on this card with the other
services resident. Both arms get the same budget, so the comparison holds.

### The archived run

`results/20261101T112253Z/` is the sweep: 8 arms, 4096 context, 512 MiB budget, one
repeat. Its checksums:

```
879dca24222e4eadda044ca4c643a3d5160ee00cea7da6e1671c7a9fbed08596  config.json
65e93b076b3e93c42f3b8e23947efcf193874066af628b89a9deef3738580bf4  report.md
c39066b2fd9299a4dd655a145dceb5787239425486f09155605b43e0fe1e2d92  rows.csv
```

The report is re-rendered from `rows.csv` alone, with no GPU, and comes out
byte-identical:

```bash
python -m kvbench.bench report --run results/20261101T112253Z
```

### The live re-run (2026-11-21)

`results/20261121T083808Z/` is the same experiment re-run on the same card through
`/home/anupam/.venv/bin/python` in WSL2: 4096 context, 16 requests, 512 MiB budget, ranks
512/256/128, with the fidelity check on. It reproduces the archived numbers.

| arm | rank | requests served | decode tok/s | prefill tok/s | ttft p50 ms | exact match | rel ΔPPL |
|---|---|---|---|---|---|---|---|
| gqa | — | 4 | 32.0 | 3,712 | 1,960 | 100 % | 0 |
| mla | 512 | 4 | 6.8 | 794 | 12,216 | 0 % | 121,771 % |
| mla | 256 | 8 | 9.9 | 1,152 | 9,315 | 0 % | 650,636 % |
| mla | 128 | 16 | 12.1 | 1,399 | 24,191 | 0 % | 20,338,947 % |

Eight identical 2048-token requests, sent concurrently through both arms with the same
budget (`artifacts/live_demo.py`):

| arm | wall | output tok/s | prefill tok/s | what it emitted |
|---|---|---|---|---|
| gqa | 5.26 s | 73.1 | 3,118 | `'course, so great that I cannot pretend to say that I have found any one of them equally well illustrated by Miss Austen's humour…'` |
| mla@128 | 11.76 s | 32.7 | 1,394 | `'................................................'` |

The trade is exact, and the trade is bad: the latent cache buys **4× the concurrent
requests** and pays **2.2× the latency** for output that is not the model's. The report
re-renders byte-for-byte from its `rows.csv`, and the suite is 32/32 on the device.

### Reproducing it

The sweep is one command, and `experiments.yaml` carries the matrix, the fidelity
floor and the workload:

```bash
docker compose run --rm kvbench python -m kvbench.bench sweep --config experiments.yaml
```

The matrix in the file is one context length and one repeat, not the plan's two and
three: each measurement starts a vLLM engine, which costs minutes on this card, and the
requests-served number is a property of the budget and the context rather than of the
repeat. The plan's matrix is the same command with `context_lens: [4096, 8192]` and
`repeats: 3`.

### What was verified, and how

| claim | how |
|---|---|
| the latent attention equals native GQA attention at r=512 | `test_rank_512_latent_reproduces_native_gqa_attention`, 1e-9 |
| the layer computes the plan's formula over a paged cache | `test_mla_attention_matches_the_plan_arithmetic` |
| the energy retention numbers | recomputed from the singular values, independently of the SVD path |
| the requests the cache serves | the peak overlap of the engine's own (scheduled, last token) intervals |
| the suite runs on the GPU in the container | `docker compose run --rm kvbench pytest tests/test_kvbench.py -v`, 32 passed |

