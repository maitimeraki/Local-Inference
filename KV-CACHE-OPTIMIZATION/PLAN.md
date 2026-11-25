# KV-Cache-Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how many more concurrent long-context requests one RTX 3050 6 GB can
serve with a Multi-head-Latent-Attention KV cache than with its native grouped-query
cache — and what that costs in output fidelity.

**Architecture:** One vLLM 0.31 engine, two model arms. The `gqa` arm is vLLM's stock
`Qwen2ForCausalLM` — the control. The `mla` arm is a custom model class that replaces
Qwen2's `k_proj`/`v_proj` with a single low-rank down-projection `W_DKV`, stores the
resulting latent `c_KV` in a paged KV cache, and reads it back through a custom Triton
decode kernel using the absorb trick (`W_UK` folded into the query, `W_UV` applied after
the attention). Everything else — weights, tokenizer, scheduler, sampler, block allocator
— is shared. The arms differ in exactly one thing: what the KV cache holds.

**Tech Stack:** Python 3.12, vLLM 0.31.0 (pinned; it requires `torch==2.13.0`, the
version already in the image), Triton (via vLLM), PyTorch 2.13.0+cu126, safetensors
0.8.0, PyYAML, pytest. Docker Compose v2 with `gpus: all`.

**Spec:** `docs/kv-cache-first-principles.md` — the plan argues from the spec; executors
read both.

## Global Constraints

- Every read, write and command for this project happens inside
  `LLM-INFERENCE/KV-CACHE-OPTIMIZATION/`. Do not read sibling projects.
- No code is written until the user says so explicitly. This document is the plan.
- `rtk` prefixes every command; if `rtk` is not on PATH, use the plain command.
- GPU work runs through `wsl -e bash -lc` with `/home/anupam/.venv/bin/python`.
  Docker owns Windows `:9000`/`:9001`; this project uses `:9100`/`:9101`.
- After stopping a vLLM engine, run `nvidia-smi` and `kill -9` any orphaned
  `EngineCore` holding ~5.3 GB.
- Commit messages end with
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- The model is `Qwen2.5-1.5B-Instruct` (GQA 12:2, 28 layers, head_dim 128, bf16).
- The corpus is Pride and Prejudice (`pg1342.txt`), sha256
  `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75`.
- The KV budget is pinned to **1.9 GiB** for every arm, via `num_gpu_blocks_override`.
  Both arms get the same *bytes*; the arm with fewer bytes per token fits more
  sequences in them. This is what makes the comparison fair.

---

## 1. Current state

`KV-CACHE-OPTIMIZATION/` contains `PLAN.md` and `README.md` only. There is no code, no
Dockerfile, no `kvbench/` package. The model and corpus are not downloaded.

| | |
|---|---|
| GPU | RTX 3050 6 GB Laptop, **sm_86** (Ampere — BF16 native, **no FP8 tensor cores**) |
| Model | `Qwen2.5-1.5B-Instruct` — 12 query heads, 2 KV heads, head_dim 128, 28 layers |
| Native KV | 2 heads × 128 × 2 (K,V) × 2 B × 28 layers = **28,672 B/token** |
| Weights | 3.09 GB bf16 — the rest of the card is the KV budget |
| Budget | 1.9 GiB ⇒ **17** concurrent 4096-token sequences at native GQA |

vLLM 0.31.0 is installed in the WSL venv and is the engine this plan targets. Verified
against the installed source (`~/.venv/lib/python3.12/site-packages/vllm`):

| fact | evidence |
|---|---|
| vLLM 0.31.0 pins `torch==2.13.0` | `vllm-0.31.0.dist-info/METADATA` |
| Triton MLA backend supports BF16 KV on sm_86 | `v1/attention/backends/mla/triton_mla.py` — BF16 requires SM80+ |
| Triton MLA backend **rejects FP8 KV on sm_86** | same file — FP8 requires SM89+ ("native FP8 (fp8e4nv) requires SM89+") |
| vLLM's MLA is a DeepSeek architecture path | `v1/attention/backends/mla/` implements `DeepseekV2/V3/V32ForCausalLM` with a fixed `kv_lora_rank` + `qk_rope_head_dim` + decoupled RoPE |
| The absorb trick already exists upstream | `model_executor/layers/attention/mla_attention.py` — `ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)` |
| A custom attention backend can be registered | `v1/attention/backends/registry.py` — `register_backend()`, `Backend.CUSTOM` |
| A custom model can be registered | `model_executor/models/registry.py` — `ModelRegistry.register_model(arch, cls)` |
| The latent cache shape is a first-class spec | `v1/kv_cache_interface.py` — `MLAAttentionSpec` (`head_size_v = 0`: *"MLA stores a single latent vector per state; there is no separate V"*) |
| The async engine exists | `v1/engine/async_llm.py` — `AsyncLLMEngine` |

The last four rows are the ones that make this plan possible: vLLM will host a model and
a KV-cache format we define.

---

## 2. The experiment

### 2.1 The two arms

| | `gqa` (control) | `mla` (treatment) |
|---|---|---|
| model class | vLLM's stock `Qwen2ForCausalLM` | `Qwen2MLAForCausalLM` (ours) |
| KV written per token per layer | K `[2,128]` + V `[2,128]` | latent `c` `[r]` |
| attention read | vLLM's stock backend | our Triton kernel |
| cache spec | `FullAttentionSpec` | `MLAAttentionSpec` (head_size = r, head_size_v = 0) |
| bytes/token (28 layers, bf16) | 28,672 | `28 × r × 2` → 28,672 (r=512) … 3,584 (r=64) |

Both arms are served by the same vLLM engine, the same scheduler, the same block
allocator, the same tokenizer, the same sampler, the same corpus and the same KV byte
budget. The arms differ in what the cache holds and in the kernel that reads it.

### 2.2 Why the comparison is fair

- **Same bytes.** `num_gpu_blocks_override` is computed per arm from the same 1.9 GiB
  target and that arm's own `page_size_bytes`, so both arms hold exactly 1.9 GiB of KV.
  A smaller cache therefore fits more sequences — the comparison is "same memory, how
  many more sequences", which is the question.
- **Same weights.** The MLA arm starts from the same `model.safetensors`; only the K/V
  projection is replaced (§3).
- **Same engine.** One vLLM version, one scheduler, one sampler. Not "our harness vs
  vLLM".
- **Same measurement code.** `kvbench/server.py` drives both arms; only the model
  registry key changes.

### 2.3 The correctness gate — `mla@512` must reproduce `gqa`

At r=512 the joint SVD is exact (§3.2): the latent holds the same information as the
native K/V, and the absorb trick is an algebraic identity. So `mla@512` is GQA with a
different rounding. The gate:

> `mla@512` and `gqa` must agree on **≥ 99 %** of tokens of the fidelity corpus and
> differ in ΔPPL by **< 0.05**.

If it fails, the construction or the kernel is wrong. **No other rank is run until this
gate passes.** This is the single most important check in the project: it is what
distinguishes "MLA at r=512 is GQA" from "we wrote a broken kernel".

### 2.4 What this plan measures that the previous one could not

| quantity | previous plan | this plan |
|---|---|---|
| requests servable | arithmetic on a byte budget | **measured**: the engine refuses the request |
| decode throughput | bandwidth model, batch of N drawn by hand | **measured** under continuous batching |
| TTFT / TPOT | not measured | **measured** p50/p95 |
| concurrency ceiling | arithmetic | **measured** rung where the ladder degrades |
| fidelity | measured | measured, plus the r=512 equivalence gate |

---

## 3. The construction

### 3.1 The algebra

Let `x` be the hidden state entering a layer, `W_K, W_V ∈ R^{256×1536}` the stacked
K/V projections (rows `0:128` = KV head 0, `128:256` = KV head 1), `b_K, b_V ∈ R^{256}`
the biases, `W_Q ∈ R^{1536×1536}`, and `W_O ∈ R^{1536×1536}`.

Native GQA, for query head `h` (KV head `g = h // 6`):

```
k_g[p] = W_K[g] x_p + b_K[g]        # [128]
v_g[p] = W_V[g] x_p + b_V[g]        # [128]
score_h[p] = (R_p q_h) · (R_p k_g[p]) * head_dim**-0.5
out_h = Σ_p softmax(score_h)[p] * v_g[p]
```

`R_p` is the RoPE rotation at position `p`; it is orthogonal.

**RoPE absorbs into the query.** `R_p` orthogonal ⇒ `(R_p q)·(R_p k) = q·k`. So

```
score_h[p] = q_h · k_g[p] * 128**-0.5
```

and RoPE never touches the cache. **The scale stays `head_dim**-0.5` = `128**-0.5` in
every arm and at every rank — not `r**-0.5`.** Getting this wrong silently breaks the
model; it is asserted in Task 1.

**The bias `b_K` is droppable, `b_V` is not.**

```
score_h[p] = (W_K[g] x_p) · q_h + q_h · b_K[g]
```

The second term is constant across `p`, so it cancels in the shift-invariant softmax.
`b_V` does not cancel and must be added *after* the up-projection.

### 3.2 The joint SVD

Stack the two projections:

```
A = [W_K ; W_V] ∈ R^{512×1536}          # rows   0:256 = W_K, 256:512 = W_V
A = U Σ V^T                             # economy SVD, float64, on CPU
```

Truncate to rank `r`:

```
W_DKV   = V[:, :r]^T                    ∈ R^{r×1536}      # written at cache-write time
W_UK[g] = (U Σ)[128g : 128g+128, :r]    ∈ R^{128×r}       # folded into the query
W_UV[g] = (U Σ)[256+128g : 256+128g+128, :r] ∈ R^{128×r}  # folded into the output
```

Then, with `c_p = W_DKV x_p ∈ R^r`:

```
W_UK[g] c_p = A_r[128g : 128g+128] x_p ≈ W_K[g] x_p
W_UV[g] c_p = A_r[256+128g : ...] x_p ≈ W_V[g] x_p
```

`A` has rank 512, so **r = 512 is lossless**: `A_512 = A`, `W_UK[g] = W_K[g]`,
`W_UV[g] = W_V[g]` exactly. Below 512 the latent is a rank-`r` approximation of the
model's key/value subspace. The discarded energy is reported per rank as

```
rel_err_k = ‖W_UK[g] W_DKV − W_K[g]‖_F / ‖W_K[g]‖_F
rel_err_v = ‖W_UV[g] W_DKV − W_V[g]‖_F / ‖W_V[g]‖_F
```

for `g ∈ {0,1}`, and the sweep is reported against them — not assumed.

### 3.3 The read path

With `q_hat[h] = W_UK[g]^T q_h ∈ R^r`:

```
score_h[p] = (q_hat[h] · c_p) * 128**-0.5
ctx[h]     = Σ_p softmax(score_h)[p] * c_p        ∈ R^r
out_h      = W_UV[g] ctx[h] + b_V[g]              ∈ R^{128}
```

Two consequences the kernel depends on:

- **One array, one pass.** K and V come from the *same* latent, so the kernel reads
  `r` values per token per layer, not `2 × 2 × 128`. At r=512 that is the same traffic
  as native GQA; at r=128 it is **4× less**.
- **The scale is `128**-0.5`.** The softmax scale belongs to the model's head_dim, not
  to the latent's rank.

### 3.4 Why the arm is a custom vLLM model

vLLM's MLA is a DeepSeek architecture path with a fixed `kv_lora_rank` +
`qk_rope_head_dim` + decoupled-RoPE layout. It cannot be pointed at Qwen2's plain
`k_proj`/`v_proj`. So the `mla` arm is a subclass of vLLM's `Qwen2ForCausalLM` that:

1. replaces `k_proj`/`v_proj` with `W_DKV` (a `Linear` 1536 → r);
2. keeps `W_UK`/`W_UV` as per-group matrices folded into the query and the output;
3. declares `MLAAttentionSpec(head_size=r, head_size_v=0)` for its KV cache;
4. uses a custom attention layer that scatters the latent into the paged cache and
   reads it back — prefill through a gathered `scaled_dot_product_attention`, decode
   through our Triton kernel.

### 3.5 The Triton decode kernel

Decode on this card is purely bandwidth-bound. The kernel is a flash-decode over the
latent:

- grid `(num_seqs, num_head_blocks)`; `BLOCK_H = 16` (12 heads padded to 16);
- `BLOCK_D = next_pow2(r)` — 512, 512, 256, 256, 128, 128, 64 for r = 512, 384, 256,
  192, 128, 96, 64 — with masked loads so **bytes read equal `r` exactly**;
- `BLOCK_N = 64` tokens per step; the block table maps logical block → physical block;
- per step: `s = tl.dot(q_blk [16, BLOCK_D], tl.trans(c_blk [BLOCK_N, BLOCK_D]))`
  → `[16, BLOCK_N]`, scaled by `128**-0.5`; online softmax (running max `m`, sum `l`);
  `acc += tl.dot(p [16, BLOCK_N], c_blk [BLOCK_N, BLOCK_D])` → `[16, BLOCK_D]`;
- write `ctx = acc / l` to `OUT[num_seqs, num_heads, r]`.

`ctx` then goes through `W_UV` and `+ b_V` in PyTorch — the value up-projection is
outside the kernel, because it is per-head work that does not scale with sequence
length.

**The reference implementation is the spec.** `kernels.py` carries a pure-PyTorch
version of the same computation, and Task 2's test asserts the Triton kernel matches it
to `1e-3` absolute. The PyTorch version is also the fallback if the Triton path fails
to compile on sm_86.

### 3.6 Prefill

Prefill is compute-bound, so it does not need a custom kernel. The attention layer
gathers the prompt's latent out of the paged cache into a contiguous `[S, r]` tensor
(one 4 MB copy at r=512, S=4096) and runs
`F.scaled_dot_product_attention(q_hat [1, 12, S, r], latent [1, 1, S, r], scale=128**-0.5)`.
`q_hat` is materialised once per layer per prompt: `12 × S × r × 2 B` = 50 MB at
r=512, S=4096 — acceptable, and halved by the same chunking the score tensor needs.

**Chunked prefill is disabled** (`enable_chunked_prefill=False`,
`max_num_batched_tokens ≥ max_model_len`). A prompt is then prefilled in one piece and
the gather is a straight copy; a partially-prefilled request would need the gather to
respect the current chunk boundary.

---

## 4. Metrics

### 4.1 What is recorded per run

| metric | unit | how |
|---|---|---|
| `kv_bytes_per_token` | B | `28 × r × 2` (mla) or `28 × 2 × 128 × 2 × 2` (gqa) |
| `kv_budget_bytes` | B | pinned at 1.9 GiB for every arm |
| `blocks` | count | `floor(budget / page_size_bytes)` |
| `requests_served` | count | max concurrent requests the engine accepts at the target length, found by raising concurrency until the scheduler starts preempting or the request errors |
| `prefill_tok_per_s` | tok/s | **inbound** — prompt tokens ÷ wall time |
| `decode_tok_per_s` | tok/s | **outbound** — aggregate over all streams ÷ wall time |
| `decode_tok_per_s_per_seq` | tok/s | per-request output rate |
| `ttft_p50`, `ttft_p95` | ms | time to first token |
| `tpot_p50`, `tpot_p95` | ms | time per output token after the first |
| `e2e_p50`, `e2e_p95` | ms | end-to-end request latency |
| `exact_match` | % | token-for-token against the `gqa` arm's greedy output |
| `rel_dppl` | % | perplexity delta against the `gqa` arm |
| `rel_err_k`, `rel_err_v` | % | SVD energy retention, per rank (§3.2) |

### 4.2 Requests servable

Measured, not computed. The ladder is 1, 2, 4, 8, 16, 32, 64, 128 concurrent requests at
a fixed prompt length. A rung "serves" if every request in it completes without a
preemption event and without an engine error; the ceiling is the last rung that serves.
The arithmetic figure (17 at 4096 for GQA) is reported next to it as the *ideal*, and the
ratio measured ÷ arithmetic is reported so the gap is visible rather than hidden.

### 4.3 Fidelity

**Exact greedy match + perplexity delta** (the criterion chosen at design time).

- Greedy: `temperature=0`, `top_p=1`, `max_tokens=256`, fixed `seed`, `n=1`. Both arms
  are run with `max_num_seqs=1`, `enable_prefix_caching=False`,
  `enable_chunked_prefill=False`, `enforce_eager=True` — batch composition and
  prefix caching both perturb reduction order and would make a token-level diff
  meaningless.
- Per response: match rate, first divergence position, divergence position as a
  fraction of response length, token-level agreement rate.
- ΔPPL: the corpus is sent with `max_tokens=1, prompt_logprobs=0`; the per-token
  logprobs come back on the prompt and `PPL = exp(−Σ logprob / n)`. Reported absolute and
  relative.

Greedy match and ΔPPL are **fidelity proxies, not task-quality guarantees** — they do
not measure factuality, helpfulness or task success, and every report says so.

### 4.4 The fidelity floor

Declared in `experiments.yaml` **before** the sweep runs:

```
exact_match >= 95 %   AND   rel_dppl <= 1 %
```

### 4.5 The recommendation rule

Fixed in advance: *the arm with the largest requests-served that still clears the floor.*
If no arm clears it, the report says so and **does not relax the floor to fit the data**.

### 4.6 Per-rank energy retention

`rel_err_k` and `rel_err_v` (§3.2) are reported for every rank, so the fidelity cliff
can be read against the truncation that caused it.

---

## 5. Predictions (falsifiable — the run must be able to refute these)

**P1 — At r=512 the MLA arm reproduces GQA.** The construction is exact there; only the
rounding differs (rotation applied to the query instead of the key; the value bias added
at read instead of baked into the cache). Expect ≥ 99 % token agreement and ΔPPL < 0.05.
**If `mla@512` shows ΔPPL > 0.5, the construction or the kernel has a bug** — this is
the primary correctness gate (§2.3) and it blocks every other rank.

**P2 — r=512 buys nothing.** Identical KV bytes, identical requests servable, identical
throughput. The entire MLA case rests on r < 512.

**P3 — Prefill gets slower as r rises.** Attention MACs/token/layer scale as `r/128`
(GQA = 3072·S; MLA = 24·r·S). At S=4096, total prefill MACs/token/layer: GQA 59.4 M;
r=512 → 97.9 M (**1.65×**); r=384 → 84.9 M (1.43×); r=256 → 72.0 M (1.21×); r=192 →
65.5 M (1.10×); r=128 → 59.0 M (0.99×, break-even); r=96 → 55.7 M (0.94×); r=64 →
52.5 M (0.88×). **Break-even is r=128.** Note this is the *arithmetic* cost; the
gather-based prefill adds a copy that this does not model, so the measured prefill gap
will be wider than 1.65× at r=512.

**P4 — Decode latency per request barely moves; aggregate throughput scales with
1/bytes.** With the KV budget fixed at 1.9 GiB and B sized to fill it, `B × KV` is
*constant*, so the step time is constant and per-request latency is unchanged. What
changes is how many requests fit: aggregate tok/s ∝ B ∝ 1/bytes-per-token. Predicted
aggregate decode throughput at 4096 context: GQA bf16 17 seqs → ~642 tok/s; MLA r=256 →
34 seqs → ~1283 tok/s (**2.0×**); MLA r=128 → 69 seqs → ~2590 tok/s (**4.0×**); MLA
r=64 → 138 seqs → ~5179 tok/s (**8.0×**). These are **upper bounds** from a pure
bandwidth model; measured numbers will be lower because kernels are not perfect and
because the engine's overhead does not shrink with the cache. **The ratio is the
prediction, not the absolute value.**

**P5 — Fidelity degrades monotonically below r=256.** The reconstruction error grows as
rank falls, and at r ≤ 128 the attention logits are computed from a 128-dim compression
of a 128-dim key — the joint SVD is already truncating the key subspace. Expect the
exact-match rate to fall off a cliff somewhere in r ∈ [128, 256]. **The interesting
result is where the cliff is.** If fidelity survives at r=256, MLA is a straight win on
this card; if it collapses at r=256, the honest answer is "the model needed the rank".

**P6 — Rank reduction is a wash at best on this model.** Qwen2.5-1.5B is a 2-KV-head GQA
model; GQA already captured the structural redundancy, and post-hoc truncation has no way
to know which directions the model actually attends to (unlike a trained `c_KV`). Expect
the sweep to find its knee no higher than r=256. **This negative result is the expected
outcome and the plan is designed to demonstrate it, not hide it.**

**P7 — The measured concurrency ceiling lands at or below the arithmetic one.** The
3.09 GB of weights, the attention workspace, the allocator's fragmentation and the
engine's own activation peak all take their cut of the same 6 GB. Expect the measured
ceiling **below** the arithmetic one, scaling *sub-linearly* with the rank — so P4's
2×/4×/8× aggregate figures are upper bounds the measurement will not reach. **The ratio
between arms is the claim; the absolute numbers are not.**

**P8 — The custom kernel is not slower than the stock one at equal traffic.** At r=512
the MLA arm reads exactly as many bytes as GQA. If `mla@512` is more than ~15 % slower
per token than `gqa`, the kernel (or the eager-mode path) is the bottleneck, and every
other rank's number inherits that handicap. This is reported as a **kernel tax** and
subtracted in the report's ratio column.

---

## 6. Layout

```
KV-CACHE-OPTIMIZATION/
├── PLAN.md                          ← this file: the plan of record
├── README.md                        ← what the project is, how to run it
├── pyproject.toml                   ← packaging + deps + the vllm.general_plugins entry point
├── Dockerfile                       ← bakes model + corpus + vLLM; CUDA via pip torch
├── docker-compose.yml               ← gpus: all, mounts ./results only
├── .dockerignore
├── experiments.yaml                 ← the canonical sweep matrix + the fidelity floor
├── docs/
│   └── kv-cache-first-principles.md ← the spec (10 sections + verification log)
├── kvbench/
│   ├── __init__.py
│   ├── config.py                    ← model id, corpus, ranks, budgets, engine flags — one source of truth
│   ├── weights.py                   ← safetensors load, joint SVD, MLA weight synthesis, energy retention
│   ├── models.py                    ← Qwen2MLAForCausalLM, the MLA attention layer, cache spec, plugin registration
│   ├── kernels.py                   ← Triton MLA decode kernel + its PyTorch reference
│   ├── server.py                    ← AsyncLLMEngine driver: requests, timing, concurrency ladder
│   └── bench.py                     ← CLI, fidelity, report rendering
└── tests/
    └── test_kvbench.py              ← one file, GPU-free where possible
```

**13 files, 7 of them Python.** `weights.py` owns the offline weight transformation,
`kernels.py` owns the kernel, `models.py` owns the vLLM integration, `server.py` owns
timing, `bench.py` owns the CLI and the report. No `server.py`/`scheduler.py` split, no
`fidelity.py`, no `report.py` — each is merged into the module that uses it.

`requirements.txt` is gone: `pyproject.toml` is the single dependency manifest, so
there is one list to keep honest, not two.

---

## 7. Tasks

Each task ends with a runnable check. **No task starts before the one above it passes
its gate.**

### Task 1 — Package skeleton, config, and the weight transformation

**Files:**
- Create: `pyproject.toml`, `kvbench/__init__.py`, `kvbench/config.py`, `kvbench/weights.py`
- Test: `tests/test_kvbench.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `config.MODEL_ID: str`, `config.CORPUS_URL: str`, `config.CORPUS_SHA256: str`
  - `config.RANKS: list[int]` = `[512, 384, 256, 192, 128, 96, 64]`
  - `config.KV_BUDGET_BYTES: int` = `2_040_109_465` (1.9 GiB)
  - `config.ENGINE_FLAGS: dict` — the pinned engine settings (§7.0)
  - `weights.load_model_state(model_dir) -> dict[str, torch.Tensor]` (CPU, the
    checkpoint's own dtype by default; pass `dtype` to cast)
  - `weights.joint_svd(state, rank) -> MLASubspace` where
    `MLASubspace` has `w_dkv [r,1536]`, `w_uk [2,128,r]`, `w_uv [2,128,r]`,
    `rel_err_k`, `rel_err_v`
  - `weights.build_mla_weights(model_dir, rank, out_dir) -> Path` — writes
    `mla_rank{r}.safetensors` and returns the path
  - `weights.load_mla_weights(path) -> dict[str, torch.Tensor]`

**The SVD, in code:**

```python
def joint_svd(state: dict[str, torch.Tensor], rank: int) -> MLASubspace:
    w_k = state["k_proj.weight"].double()          # [256, 1536]
    w_v = state["v_proj.weight"].double()          # [256, 1536]
    a = torch.cat([w_k, w_v], dim=0)               # [512, 1536]
    u, s, vh = torch.linalg.svd(a, full_matrices=False)   # [512,512], [512], [1536,512]
    u_r = u[:, :rank] * s[:rank]                   # [512, r]
    w_dkv = vh[:rank, :].contiguous()              # [r, 1536]; vh is already V^T,
                                                   # so no transpose -- and the
                                                   # slice must be contiguous for
                                                   # safetensors.save_file
    w_uk = torch.cat([u_r[0:128].unsqueeze(0),
                      u_r[128:256].unsqueeze(0)], dim=0)   # [2, 128, r]
    w_uv = torch.cat([u_r[256:384].unsqueeze(0),
                      u_r[384:512].unsqueeze(0)], dim=0)   # [2, 128, r]
    ...
```

`rel_err_k[g] = ‖w_uk[g] @ w_dkv − w_k[g]‖_F / ‖w_k[g]‖_F`, and likewise for `v`.

**The scale assertion** (the one that silently breaks models):

```python
def test_softmax_scale_is_head_dim_not_rank():
    # A rank-128 latent must still use 128**-0.5, because the score is a
    # projection of a head_dim-128 dot product.
    assert config.ATTENTION_SCALE == 128 ** -0.5
    assert config.ATTENTION_SCALE != 64 ** -0.5   # never the rank's own scale
```

plus a numeric test: build a random `q`, `c`, compute the score both ways
(`(W_UK^T q) · c * 128**-0.5` vs `(W_UK^T q) · c * r**-0.5`) and assert they differ,
and that only the first reproduces `q · (W_K x) * 128**-0.5`.

- [x] **Step 1:** Write `tests/test_kvbench.py` with the scale test and a
      `test_rank_512_is_lossless` that asserts `w_uk[g] @ w_dkv == w_k[g]` to `1e-8`
      and `rel_err_k`/`rel_err_v` below `1e-9` at r=512 — float64 SVD roundoff, so
      "zero" is a tolerance and not an equality.
- [x] **Step 2:** `pytest tests/test_kvbench.py -v` — expected: FAIL, no module.
- [x] **Step 3:** Write `pyproject.toml` (deps, `[project.entry-points."vllm.general_plugins"]`
      pointing at `kvbench.models:register_plugin`), `kvbench/config.py`,
      `kvbench/weights.py`.
- [x] **Step 4:** `pytest tests/test_kvbench.py -v` — expected: PASS.
- [x] **Step 5:** `python -c "from kvbench.weights import joint_svd; ..."` against the real
      downloaded weights; confirm `rel_err_k`/`rel_err_v` are at float64 roundoff
      (~4e-15 measured) at r=512 and grow monotonically as r falls.
- [x] **Step 6:** Commit — `git add pyproject.toml kvbench tests && git commit -m "feat(weights): joint-SVD MLA weight construction and the head_dim scale assertion"`.

**Gate:** r=512 is provably lossless and the scale is provably `128**-0.5`.

---

### Task 2 — The Triton MLA decode kernel

**Files:**
- Create: `kvbench/kernels.py`
- Modify: `tests/test_kvbench.py`

**Interfaces:**
- Consumes: `weights.MLASubspace` (Task 1)
- Produces:
  - `kernels.mla_decode_reference(q_hat, latent, seq_lens, scale) -> torch.Tensor`
    — pure PyTorch, `[num_seqs, num_heads, r]`; `q_hat` `[num_seqs, num_heads, r]`,
    `latent` `[num_seqs, seq_len, r]`
  - `kernels.mla_decode_triton(q_hat, latent_cache, block_table, seq_lens, scale,
    head_dim) -> torch.Tensor` — the paged version; `latent_cache`
    `[num_blocks, block_size, r]`, `block_table` `[num_seqs, max_blocks]`
  - `kernels.triton_available() -> bool`

- [ ] **Step 1:** Write the failing test: random `q_hat`, random `latent`, random
      `seq_lens`; assert `mla_decode_triton` == `mla_decode_reference` to `1e-3`
      absolute, for `r ∈ {64, 128, 256, 512}` and `num_seqs ∈ {1, 3}`.
- [ ] **Step 2:** `pytest tests/test_kvbench.py -k triton -v` — expected: FAIL.
- [ ] **Step 3:** Write `mla_decode_reference` first, make the test pass against it
      (the reference defines correctness).
- [ ] **Step 4:** Write the Triton kernel. Grid `(num_seqs, cdiv(num_heads, BLOCK_H))`;
      `BLOCK_H = 16` (12 heads padded, masked); `BLOCK_D = next_pow2(r)` (masked loads,
      so bytes read = r exactly); `BLOCK_N = 64`. Per block: load `c_blk
      [BLOCK_N, BLOCK_D]`, `s = tl.dot(q_blk, tl.trans(c_blk)) * scale`, online softmax
      over `m`/`l`, `acc += tl.dot(p, c_blk)`, store `acc / l`.
- [ ] **Step 5:** `pytest tests/test_kvbench.py -k triton -v` — expected: PASS.
- [ ] **Step 6:** Benchmark the kernel against the reference on the real shapes
      (`num_seqs=17, r=128, seq_len=4096`) and record the ratio in the PR description.
      A kernel more than 2× slower than the reference means it will not be the
      bottleneck — if it is, the reference becomes the shipping path and this task
      records that.
- [ ] **Step 7:** Commit — `git add kvbench/kernels.py tests && git commit -m "feat(kernels): Triton MLA latent decode kernel with a PyTorch reference"`.

**Gate:** the kernel matches the reference to `1e-3` at every rank in the sweep.

---

### Task 3 — The vLLM MLA model class, attention layer and cache spec

**Files:**
- Create: `kvbench/models.py`
- Modify: `tests/test_kvbench.py`

**Interfaces:**
- Consumes: `kernels.mla_decode_triton` (Task 2), `weights.load_mla_weights` (Task 1)
- Produces:
  - `models.Qwen2MLAForCausalLM` — a `Qwen2ForCausalLM` subclass
  - `models.MLAAttentionLayer` — the custom attention layer
  - `models.register_plugin()` — calls
    `ModelRegistry.register_model("Qwen2MLAForCausalLM", Qwen2MLAForCausalLM)`
  - `models.register_backend()` — calls
    `register_backend(Backend.CUSTOM, "kvbench.models.MLABackend")`

**What the model class must do:**

1. **Weights.** Load `W_DKV` as a `Linear(1536, r, bias=False)` replacing `k_proj` and
   `v_proj`; keep `W_UK` `[2,128,r]` and `W_UV` `[2,128,r]` as parameters; keep `b_V`
   `[2,128]`; keep `W_Q`, `W_O`, the norms, the MLP, the embeddings untouched.
2. **Query absorption.** For query head `h` with KV group `g = h // 6`:
   `q_hat[h] = W_UK[g]^T q_h`, i.e. `q_hat = einsum("bhsd,grd->bhsr", q.view(b,s,12,128), w_uk.repeat_interleave(6,0))`.
   This is a `[12,128,r]` contraction — done once per layer per step.
3. **Write.** `c = x @ W_DKV.T` → scatter into the paged cache at the slot positions the
   allocator gave us.
4. **Prefill read.** Gather the prompt's latent from the paged cache into `[S, r]`, then
   `F.scaled_dot_product_attention(q_hat[:, :, :S, :], latent[None, None], scale=128**-0.5)`.
5. **Decode read.** `kernels.mla_decode_triton(...)` → `ctx [B, 12, r]`.
6. **Value path.** `out = einsum("bhsr,grd->bhsd", ctx, w_uv.repeat_interleave(6,0)) + b_v.repeat_interleave(6,0)`,
   then `o_proj`.
7. **Cache spec.** `get_kv_cache_spec` returns
   `MLAAttentionSpec(block_size=..., head_size=r, head_size_v=0, dtype=torch.bfloat16)`.
8. **No RoPE anywhere.** RoPE is absorbed into the query in step 2; the latent stored in
   the cache is position-independent.

**Interfaces between the layer and vLLM:** the layer reads `forward_context`'s
`attn_metadata` and the cache tensors, writes the latent, and returns the attention
output in the shape vLLM expects.

- [ ] **Step 1:** Write the failing test: `ModelRegistry` resolves
      `"Qwen2MLAForCausalLM"` after `register_plugin()`; `get_kv_cache_spec` reports
      `head_size == r` and `head_size_v == 0`.
- [ ] **Step 2:** `pytest tests/test_kvbench.py -k model -v` — expected: FAIL.
- [ ] **Step 3:** Write `models.py`.
- [ ] **Step 4:** `pytest tests/test_kvbench.py -k model -v` — expected: PASS.
- [ ] **Step 5:** **Smoke test on the GPU** — start the engine with the `mla` arm at
      r=512, `max_model_len=4096`, `gpu_memory_utilization=0.9`, and generate 64 tokens
      from a fixed prompt. It must produce coherent text and the engine must report a
      KV cache of `28 × 512 × 2` bytes per token. Record the output text in the commit
      message — a run of garbage tokens means the absorb algebra is wrong, and it is
      caught here rather than in the sweep.
- [ ] **Step 6:** Commit — `git add kvbench/models.py tests && git commit -m "feat(vllm): Qwen2MLA model class, latent attention layer and MLAAttentionSpec"`.

**Gate:** the `mla` arm runs on the GPU and generates coherent text at r=512.

---

### Task 4 — The engine driver: async requests and timing

**Files:**
- Create: `kvbench/server.py`
- Modify: `tests/test_kvbench.py`

**Interfaces:**
- Consumes: `models.register_plugin` (Task 3), `config.ENGINE_FLAGS` (Task 1)
- Produces:
  - `server.EngineConfig` — dataclass: `arm: str`, `rank: int | None`,
    `max_model_len: int`, `kv_budget_bytes: int`, `gpu_memory_utilization: float`
  - `server.build_engine(cfg) -> AsyncLLMEngine`
  - `server.blocks_for(cfg) -> int` — `kv_budget_bytes // page_size_bytes`
  - `server.make_requests(prompts, output_tokens) -> list[Request]` — the async client
  - `server.run_load(engine, requests, concurrency) -> LoadResult` — `LoadResult` has
    `prefill_tok_per_s`, `decode_tok_per_s`, `decode_tok_per_s_per_seq`,
    `ttft_ms`, `tpot_ms`, `e2e_ms` (each a list of samples)
  - `server.greedy_outputs(engine, prompts, max_tokens) -> list[list[int]]` — token ids

**Engine flags, pinned:**

```python
ENGINE_FLAGS = {
    "enforce_eager": True,          # no CUDA graphs: equal treatment for both arms, and a static grid
    "enable_prefix_caching": False, # changes reduction order
    "enable_chunked_prefill": False,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.90,
    "swap_space": 0,
    "seed": 0,
}
```

`enforce_eager=True` costs both arms the same ~10–25 % of absolute decode throughput
versus a graph-captured run. It is stated in the report as the ceiling, not hidden.

- [ ] **Step 1:** Write the failing test: `blocks_for` returns the expected block count
      for a known budget and a known `page_size_bytes` (GQA: `block_size × 2 × 128 × 2`
      B; MLA: `block_size × r × 2` B); and `make_requests` produces N distinct request
      objects with the right prompt and output length.
- [ ] **Step 2:** `pytest tests/test_kvbench.py -k server -v` — expected: FAIL.
- [ ] **Step 3:** Write `server.py`. `run_load` uses `asyncio.gather` over
      `concurrency` in-flight requests, restarting the next as each completes, and
      timestamps first-token and last-token per request.
- [ ] **Step 4:** `pytest tests/test_kvbench.py -k server -v` — expected: PASS.
- [ ] **Step 5:** Smoke test on the GPU: run `gqa` and `mla@512` for 64 requests of
      1024 tokens at concurrency 8; confirm both complete, and record the two aggregate
      decode rates.
- [ ] **Step 6:** Commit — `git add kvbench/server.py tests && git commit -m "feat(server): async engine driver with TTFT/TPOT and the concurrency ladder"`.

**Gate:** both arms serve real concurrent requests and report timings.

---

### Task 5 — Fidelity: exact greedy match and ΔPPL

**Files:**
- Create: `kvbench/bench.py` (the `fidelity` subcommand)
- Modify: `tests/test_kvbench.py`

**Interfaces:**
- Consumes: `server.greedy_outputs` (Task 4), `config.CORPUS_URL`
- Produces:
  - `bench.fidelity(cfg, prompts) -> FidelityResult` with `exact_match`,
    `first_divergence`, `divergence_fraction`, `token_agreement`, `ppl`, `rel_dppl`
  - `bench.corrode` — the corpus is fetched at build time and verified by sha256

- [ ] **Step 1:** Write the failing test: `first_divergence([1,2,3], [1,2,4]) == 2`;
      `first_divergence([1,2], [1,2]) is None`; `token_agreement` on a known pair.
- [ ] **Step 2:** `pytest tests/test_kvbench.py -k fidelity -v` — expected: FAIL.
- [ ] **Step 3:** Write the fidelity functions. Greedy: `SamplingParams(temperature=0,
      top_p=1.0, max_tokens=256, n=1, seed=0)`. ΔPPL: `SamplingParams(max_tokens=1,
      prompt_logprobs=0)`; sum the returned prompt logprobs over the corpus.
- [ ] **Step 4:** `pytest tests/test_kvbench.py -k fidelity -v` — expected: PASS.
- [ ] **Step 5:** **Run the §2.3 gate**: `gqa` vs `mla@512` on the fidelity corpus.
      Record exact-match and ΔPPL. **If exact-match < 99 % or ΔPPL > 0.05, stop and
      debug — no other rank runs until it passes.**
- [ ] **Step 6:** Commit — `git add kvbench/bench.py tests && git commit -m "feat(fidelity): exact greedy match and prompt-logprob perplexity delta"`.

**Gate:** `mla@512` reproduces `gqa`.

---

### Task 6 — The comparison CLI and the report

**Files:**
- Modify: `kvbench/bench.py`
- Modify: `tests/test_kvbench.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `bench.main(argv)` with subcommands `prepare`, `compare`, `sweep`, `fidelity`,
    `report`
  - `bench.render_report(rows, out_dir) -> Path` writing
    `rows.csv`, `report.md`, `config.json`

- [ ] **Step 1:** Write the failing test: a synthetic `rows` list renders to a Markdown
      table containing the arm names and the fidelity-floor line.
- [ ] **Step 2:** `pytest tests/test_kvbench.py -k report -v` — expected: FAIL.
- [ ] **Step 3:** Write the CLI and the report renderer. The report has three tables:
      the headline (requests served, throughput, latency per arm), the trade curve
      (rank vs exact-match vs requests served, with the floor drawn), and the fidelity
      detail (exact match, ΔPPL, `rel_err_k`, `rel_err_v`). Every table carries the
      arithmetic ideal next to the measured value and the measured ÷ arithmetic ratio.
- [ ] **Step 4:** `pytest tests/test_kvbench.py -k report -v` — expected: PASS.
- [ ] **Step 5:** Commit — `git add kvbench/bench.py tests && git commit -m "feat(bench): comparison CLI and the measurement report"`.

**Gate:** `python -m kvbench.bench report --run <dir>` renders a report from a
`rows.csv` with no GPU.

---

### Task 7 — The container and the runbook

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- Modify: `README.md`

- [ ] **Step 1:** Write the `Dockerfile`: `python:3.12-slim` + `apt-get install libgomp1`;
      `pip install torch==2.13.0 --index-url .../cu126`; `pip install vllm==0.31.0`
      (verified: it pins the same torch); `pip install -e .`; download the model and the
      corpus into `/opt/models` and `/opt/data`, verifying the corpus sha256; set
      `HF_HUB_OFFLINE=1`.
- [ ] **Step 2:** Write `docker-compose.yml`: service `kvbench`, `gpus: all`, the
      environment above, and only `./results` bind-mounted. WSL2 paths do not
      bind-mount — the model and corpus are baked in.
- [ ] **Step 3:** `docker compose build` — expected: the image builds and the model is
      present at `/opt/models/Qwen2.5-1.5B-Instruct`.
- [ ] **Step 4:** `docker compose run --rm kvbench pytest tests/test_kvbench.py -v` —
      expected: PASS in the container.
- [ ] **Step 5:** `docker compose run --rm kvbench python -m kvbench.bench compare
      --arms gqa,mla --ranks 512 --context-lens 4096` — expected: a populated
      `results/<timestamp>/` with the gate measurement.
- [ ] **Step 6:** Update `README.md` with the layout, the commands, and what each
      number means.
- [ ] **Step 7:** Commit — `git add Dockerfile docker-compose.yml .dockerignore README.md && git commit -m "build: container image, compose service and the runbook"`.

**Gate:** the whole pipeline runs end to end inside Docker.

---

### Task 8 — The full sweep and the write-up

**Files:**
- Create: `experiments.yaml`
- Modify: `README.md`

**The sweep matrix** (16 combinations — 2 arms × 8 ranks minus the duplicate r=512 GQA,
× 2 context lengths × 3 repeats):

```yaml
arms: [gqa, mla]
ranks: [512, 384, 256, 192, 128, 96, 64]
context_lens: [4096, 8192]
repeats: 3
storage: bf16
fidelity_floor:
  exact_match_min: 0.95
  rel_dppl_max: 0.01
```

- [ ] **Step 1:** Write `experiments.yaml`.
- [ ] **Step 2:** `docker compose run --rm kvbench python -m kvbench.bench sweep
      --config experiments.yaml` — expected: a full `results/<timestamp>/`.
- [ ] **Step 3:** Render the report and **read it against §5**. For each prediction,
      record whether it held. A prediction that failed is written into the report as a
      failed prediction with the measurement that refuted it.
- [ ] **Step 4:** Write the conclusion into `README.md`: the arm with the largest
      requests-served that clears the floor, the measured ratios, and — explicitly —
      whether the answer is "MLA helps" or "the model needed the rank" (§5 P5/P6).
- [ ] **Step 5:** Commit — `git add experiments.yaml README.md results && git commit -m
      "feat: full sweep results and the measured conclusion"`.

**Gate:** the report answers the question, whether the answer is yes or no.

---

### Task 9 — Reproducibility and the archive

**Files:**
- Modify: `README.md`, `experiments.yaml`

- [ ] **Step 1:** Record in `README.md`: the exact vLLM commit/version, the Triton
      version, the driver, the CUDA version, and the `nvidia-smi` output at run time.
- [ ] **Step 2:** Record the sha256 of every artifact in `results/<timestamp>/`.
- [ ] **Step 3:** Confirm `docker compose run --rm kvbench python -m kvbench.bench report
      --run results/<timestamp>` reproduces the report byte-for-byte from `rows.csv`
      alone.
- [ ] **Step 4:** Commit — `git add README.md experiments.yaml && git commit -m "docs: reproducibility record and result checksums"`.

**Gate:** the report is reproducible from the archived rows.

---

## 8. Commands

```bash
# from LLM-INFERENCE/KV-CACHE-OPTIMIZATION/

# build the image (bakes the model + corpus + vLLM; ~8 GB)
docker compose build

# build the MLA weights for every rank (CPU, offline, ~2 min)
docker compose run --rm kvbench python -m kvbench.bench prepare --ranks 512,384,256,192,128,96,64

# THE GATE: mla@512 must reproduce gqa
docker compose run --rm kvbench python -m kvbench.bench compare \
    --arms gqa,mla --ranks 512 --context-lens 4096 --repeats 1

# the headline comparison
docker compose run --rm kvbench python -m kvbench.bench compare \
    --arms gqa,mla --ranks 512,256,128 --context-lens 4096 --repeats 3

# the full sweep
docker compose run --rm kvbench python -m kvbench.bench sweep --config experiments.yaml

# fidelity only: exact greedy match + delta perplexity
docker compose run --rm kvbench python -m kvbench.bench fidelity \
    --arms gqa,mla --ranks 512,256,128

# re-render the report from rows.csv, no GPU needed
docker compose run --rm kvbench python -m kvbench.bench report --run results/<timestamp>

# the tests
docker compose run --rm kvbench pytest tests/test_kvbench.py -v
```

Results land in `./results/<UTC timestamp>/{config.json, rows.csv, report.md}`,
bind-mounted from the host.

> If a bind mount fails with a path error under Git Bash, prefix the command with
> `MSYS_NO_PATHCONV=1`.

**Running the two arms is two container runs, not one:** the engine holds the weights and
the KV budget, so only one arm fits in the 6 GB at a time. `compare` runs them in
sequence inside one container.

---

## 9. Constraints and risks

| Risk | Handling |
|---|---|
| `mla@512` ≠ `gqa` | Expected to differ slightly (rotation on the query vs the key; `b_V` added at read). The gate is ≥ 99 % agreement and ΔPPL < 0.05, not bitwise equality. A ΔPPL > 0.5 means a bug and blocks the sweep. |
| The joint SVD starves the K block at low rank | `rel_err_k` / `rel_err_v` are reported per rank. The fallback — separate SVDs of `W_K` and `W_V` with a concatenated latent — is deferred to §11 until the measurement says it is needed. |
| vLLM's MLA is DeepSeek-shaped and cannot be reused | Confirmed by reading the installed source. The MLA arm is our own model class, attention layer, cache spec and kernel. The *engine* is reused, the *mechanism* is not. |
| FP8 is wanted but unavailable | vLLM's Triton MLA backend rejects FP8 KV cache below SM89 and this card is sm_86. Measured E4M3 round-trip error is 2.67 % RMS against BF16's 0.20 %, which alone would breach the 1 % ΔPPL floor. One storage dtype: bf16. |
| The Triton kernel is slower than the stock path | Predicted and bounded: P8 treats it as a **kernel tax** reported next to every ratio. The PyTorch reference in `kernels.py` is the fallback and the ceiling. |
| CUDA graphs would have helped both arms | `enforce_eager=True` for every run. It costs both arms the same ~10–25 % of absolute decode throughput and removes a whole class of numerical and capture-safety confound. Stated in the report. |
| Continuous batching perturbs the fidelity diff | Fidelity runs pin `max_num_seqs=1`, `enable_prefix_caching=False`, `enable_chunked_prefill=False`, `enforce_eager=True`. Throughput runs use the full engine. |
| The 6 GB card OOMs | `gpu_memory_utilization=0.90`, `swap_space=0`, `max_model_len=8192`, the KV budget pinned at 1.9 GiB. Weights 3.09 GB + 1.9 GiB KV + the activation peak fits in 6 GB. |
| The engine OOMs during the sweep | One arm per container run; the engine is torn down and `nvidia-smi` checked for an orphaned `EngineCore` before the next arm starts. |
| Thermal throttling over a long sweep | `repeats: 3`; the report prints the spread. |
| vLLM internals move between versions | The version is pinned to 0.31.0 and the custom code touches only `ModelRegistry`, `register_backend`, `MLAAttentionSpec` and the forward-context API — four narrow seams. |
| WSL2 paths do not bind-mount into the container | The model and corpus are baked into the image; only `./results` is mounted. |
| The model download fails at build time | The build requires network. The image contains the model afterwards and `HF_HUB_OFFLINE=1` makes the runtime hermetic. |

---

## 10. Out of scope

- **A trained MLA.** Post-hoc weight conversion only — nothing is retrained.
- **DeepSeek's decoupled RoPE.** Not needed here: RoPE absorbs fully into the query
  (§3.1), which is why the latent is position-independent and the cache holds `r`
  values with no separate rope key.
- **Token eviction / SnapKV / prefix caching as a *measurement*.** Prefix caching is
  disabled during the runs so it cannot contaminate them; eviction changes *what is
  read*, which would break the one-variable rule of §2.2.
- **Quantised weights (AWQ/GPTQ/FP8).** The weights are BF16 in both arms.
- **Tensor parallelism, speculative decoding, multi-GPU.** One 3050.
- **A public HTTP server as the primary measurement path.** `AsyncLLMEngine` in-process
  is the measurement path — it returns token ids and engine timings directly, without an
  HTTP layer to parse. An OpenAI-compatible server is one flag away from the same code
  and is noted in the README as the way to drive it by hand.

---

## 11. Open items

1. **Does the balanced construction earn its place?** Separate SVDs of `W_K` and `W_V`
   with a concatenated latent would keep the K and V subspaces from competing for the
   same `r` directions. It is deferred until `rel_err_k` / `rel_err_v` from the sweep
   show the joint SVD starving one of them. It needs a different split point inside the
   same kernel (`K_DIM = r/2`), not a second kernel.
2. **Context length 16384.** The model supports 32768 positions and the KV arithmetic
   gets more interesting there (4 sequences for GQA instead of 17). The sweep runs 4096
   and 8192.
3. **How much of the gap is the eager-mode penalty.** The stock arm and the MLA arm
   both run eager. A supplementary run of the stock arm with CUDA graphs enabled would
   quantify the ceiling; it is not part of the headline numbers.
4. **Whether the MLA arm's prefill gather is worth a real kernel.** §3.6 copies the
   latent out of the paged cache before each prefill. At r=512, S=4096 that is a 4 MB
   copy against ~26 ms of prefill — negligible, but it is a measurement, not an
   assumption.
