# KV Cache Optimization — From First Principles

**Status:** understanding phase. No code yet, by request.
**Date:** 2026-10-28
**Scope:** what the KV cache is, how it can be optimized, the process for doing it,
the best method for the hardware we actually have, and where DeepSeek's MLA genuinely
helps (and where it cannot be reached).

Everything numeric in here is either (a) computed from configs I read off disk or off
Hugging Face, or (b) attributed to a source. Section 9 is the verification log.

---

## Table of contents

1. [What the KV cache actually is](#1-what-the-kv-cache-actually-is)
2. [Two axes of compression](#2-two-axes-of-compression)
3. [The law: restructure first, then store at the highest precision you can afford](#3-the-law-restructure-first-then-store-at-the-highest-precision-you-can-afford)
4. [DeepSeek's MLA, derived from scratch](#4-deepseeks-mla-derived-from-scratch)
5. [The full menu of KV-cache optimizations](#5-the-full-menu-of-kv-cache-optimizations)
6. [What the best method is on this box](#6-what-the-best-method-is-on-this-box)
7. [The process](#7-the-process)
8. [How DeepSeek's MLA is actually helpful here](#8-how-deepseeks-mla-is-actually-helpful-here)
9. [Verification log](#9-verification-log)
10. [Open questions](#10-open-questions)

---

## 1. What the KV cache actually is

### 1.1 Attention is a function of the prefix

For one token position `t` in one layer, a transformer computes:

```
q_t ∈ R^{n_h × d_h},   k_t ∈ R^{n_h × d_h},   v_t ∈ R^{n_h × d_h}
o_t = softmax( q_t k_{1:t}^T / sqrt(d_h) + mask ) · v_{1:t}
```

During **prefill** the whole prefix is present, so this is one big matrix product.
During **decode** (one token at a time) `q_t` is a single vector, but the sum still runs
over every token `1..t`.

The KV cache is simply: *"keep `k` and `v` for every previous token, so decode does not
have to recompute them."*

### 1.2 Why it exists (the arithmetic that forces it)

Recomputing `k` and `v` for all `N` prefix tokens at every decode step costs `O(N)` FLOPs
per step, i.e. `O(N²)` FLOPs over a whole generation. With a cache, decode reads `O(N)`
values per step and writes `O(1)`. For `N = 4096` that is the difference between ~16.7M
and ~4K units of attention work per step.

This is the whole reason the cache exists, and it is also the whole reason it is
expensive: **it is the only part of the model that grows with sequence length and with
number of concurrent users.**

### 1.3 The invariant that makes compression measurable

> The cache is a **pure function of the prefix**. Same prefix ⇒ same `k`, same `v` ⇒
> same output distribution.

Three consequences:

- **Cacheable.** Two requests with a shared prefix can share cached `k`/`v`. This is why
  prefix caching is free quality-wise (see §5.5).
- **Deterministic.** Given a fixed prefix and fixed kernel, the output is fixed. So a
  compressed cache can be compared to a full cache *token for token*.
- **Everything else in this document is a violation of this invariant, deliberately.**
  Any change to the cache changes the output. The only question is *how much*, and how
  many bytes you saved to buy that change. That ratio is the entire field.

### 1.4 What it costs — arithmetic on your actual model

`Qwen2.5-1.5B-Instruct` (read from the local HF cache):

```
hidden_size          = 1536
num_hidden_layers    = 28
num_attention_heads  = 12
num_key_value_heads  = 2      ← GQA: 12 query heads share 2 KV heads
head_dim             = 1536 / 12 = 128
torch_dtype          = bfloat16
max_position_embeddings = 32768
rope_theta           = 1_000_000
sliding_window       = 32768  (== max len, so it never engages)
```

Bytes per token, per layer, for `K` and `V` together:

```
2 (K and V) × n_kv_heads (2) × head_dim (128) × 2 bytes (bf16) = 1,024 B
```

Across all 28 layers:

```
1,024 B × 28 layers = 28,672 B = 28 KiB per token
```

Per sequence at the 4096-token window you are already using:

```
28,672 B × 4,096 = 117,440,512 B ≈ 112 MiB per sequence
```

**For comparison, plain MHA on this same model** (12 KV heads instead of 2):

```
2 × 12 × 128 × 2 B × 28 layers = 172,032 B = 168 KiB per token  →  672 MiB per sequence
```

So your 1.5B model is *already* 6× compressed by **GQA** before anyone touches a
quantizer. Keep that in mind: the GQA-vs-MLA comparison in §4.2 is the honest
"starting line", not MHA.

### 1.5 The economics: cache capacity *is* user capacity

Your card and budget:

```
RTX 3050 6GB Laptop GPU, 6,144 MiB total
gpu_memory_utilization = 0.80            → 4,915 MiB budget
weights ≈ 1.54B params × 2 B (bf16)       → 3,080 MiB
remaining for KV + activations            → ≈ 1,835 MiB  (call it ~1.9 GB)
```

Tokens of KV that fit:

```
1,835 MiB × 1024 × 1024 / 28,672 B ≈ 67,000 tokens  ≈ 16 sequences at 4,096 tokens
```

This is an **upper bound** — activations, the CUDA-graph pool, and block fragmentation
eat into it. But the order of magnitude is the finding, and it matches what your
`BENCHMARK-OBSERVABILITY/config/engine/baseline.yaml` already says out loud:

> "roughly 2 GB of KV cache at 0.80 utilisation … comfortably enough for short contexts
> and tight for 3072-token ones — the ceiling the concurrency experiment will find is
> real, and reporting it is the finding."

**So the KV cache is ~40% of your entire VRAM budget and it is what caps concurrency.**
Compressing it is not a micro-optimization; on this card it is *the* lever, because
concurrency is the product you are selling.

---

## 2. Two axes of compression

Everything in this field lives on one of two axes, and confusing them is the source of
most bad KV-cache work.

### 2.1 Numeric compression — change the *precision*

Keep the same representation, store fewer bits per element.

| Format | Mantissa bits | Max relative error (round-to-nearest) | Notes |
|---|---|---|---|
| BF16 | 8 (7 explicit) | 2⁻⁹ ≈ **0.20%** | Native on your sm_86. The baseline. |
| FP16 | 11 (10 explicit) | 2⁻¹² ≈ 0.024% | Range risk (max 65504), not accuracy risk |
| FP8 E4M3 | 4 (3 explicit) | 2⁻⁵ ≈ **3.1%** | No FP8 tensor cores on sm_86 |
| FP8 E5M2 | 3 (2 explicit) | 2⁻⁴ ≈ 6.3% | Wider range, worse precision |
| INT8 (per-token scale) | — | step/2 of the token's range | Scale choice dominates |
| INT4 (group-64) | — | step/2 of the group's range | Needs group-wise scales |
| NVFP4 | E2M1 + FP8 per-16 block scale | — | Newest; 0.5 B/element + 1 B per 16 |

**Error model:** numeric compression injects an *independent, roughly uniform* rounding
error into **every element of every K and every V**. It is unbiased, but it is everywhere.

### 2.2 Structural compression — change *what is stored*

Keep the precision, store less information.

| Technique | What changes | Needs training? |
|---|---|---|
| MQA | 1 KV head for all query heads | Yes (or fine-tune) |
| GQA | KV head shared by a group of query heads | Yes (or fine-tune) |
| **MLA** | Store a **shared low-rank latent**, reconstruct K/V on the fly | **Yes** |
| Token eviction | Drop tokens that received little attention mass | No |
| Pooling | Merge evicted tokens into a summary token | No |
| Grouping for quantization | Reorder so similar tokens share a scale | No |

### 2.3 Why the two behave so differently

Attention output for one query position:

```
p_j = exp(z_j) / Σ_k exp(z_k)     with   z_j = q · k_j / sqrt(d_h)
o   = Σ_j p_j v_j
```

**Numeric compression** perturbs `k_j` ⇒ perturbs `z_j` ⇒ perturbs `p_j` ⇒ perturbs `o`.
And it perturbs `v_j` directly. Two things compound:

- The softmax is a *normalized exponential*. A relative perturbation `ε` on the logits
  produces roughly a relative perturbation `ε` on the weights — and it is applied at
  **every one of N positions**, in **every layer**, and each layer's output feeds the next.
- `K` has a **heavy-tailed** distribution across the sequence. A few tokens have very
  large magnitudes. A single per-tensor scale is therefore set by the tail, and the
  *many* small-magnitude tokens — which are most of the sequence — get crushed. This is
  exactly why `int8_per_token_head` and `int4_per_token_head` exist as separate dtypes
  from a plain `int8`: **the scale granularity is the whole game**, far more than the
  bit-width.

**Structural compression** does something categorically different. It does not round
anything that the model cares about. It changes *what the model is given*, so that the
thing given is a **sufficient statistic** for the attention the model will perform.

That is the deep difference, and it is the entire answer to your question.

---

## 3. The law: restructure first, then store at the highest precision you can afford

This is the organizing principle of the whole document. Here it is as an explicit
statement, with the error budget that justifies it.

> **Restructure.** Change the representation of the cache so that what you keep is the
> information the model actually needs, and the discarded part is information the model
> provably never uses.
>
> **Then quantize.** Only after restructuring, spend bits on the reduced representation.
> Because the representation is now small and well-conditioned, you can often stop at
> BF16 and never pay quantization error at all.

### 3.1 The claim, made numerically

Suppose after restructuring you store a quantity `z` in BF16. The relative error on `z`
is ≤ 2⁻⁹ ≈ 0.2% (half-ulp, round-to-nearest).

| Scheme | Relative error on the quantity the attention consumes | Bytes/element |
|---|---|---|
| MLA latent in BF16, reconstructed | **≈ 0.2%** | 2 |
| Original per-head K/V in FP8 E4M3 | ≈ 3.1% | 1 |
| Original per-head K/V in FP8 E5M2 | ≈ 6.3% | 1 |

**MLA-in-BF16 is ~15× more accurate than FP8-of-the-original — while using twice the
bytes per element, and while also being 7–20× smaller overall.**

That is the whole thing your question was circling. "Restructure the cache and the output
accuracy improves, especially if you store the cache with BF16" is not a subtle
empirical observation; it is a direct consequence of the fact that a *restructured* cache
is a learned bottleneck rather than a rounded transcript.

### 3.2 Why MLA's latent is not a lossy summary of K

This is the part that is easy to get wrong, so let me state it precisely.

In a standard transformer, `K` is computed from the residual stream and then *used* by
attention. In MLA, the model computes:

```
c_KV = W_DKV · h        (a learned low-rank projection, dim d_c)
k_h   = W_UK[h] · c_KV  (per head, dim d_h)
v_h   = W_UV[h] · c_KV
```

**`c_KV` is not an approximation of `K`. It is the thing `K` was computed from.** The
model was *trained* with the rank-`d_c` bottleneck in the forward pass, so the
information that `W_DKV` discards was never available to the model in the first place.
Storing `c_KV` in BF16 and reconstructing `K` loses only the BF16 rounding of `c_KV` —
the same 0.2% you would have lost storing `K` itself in BF16, and nothing more.

Whereas FP8-of-the-original-KV injects 3.1% error into a quantity the model *does* use
directly.

Same idea, one level up, is why **DeepSeek-V3 stores the MLA latent in FP8** and gets
away with it: the latent was already structurally compressed, so the extra numeric
compression is applied to a well-conditioned, small, low-dimensional object rather than
to a heavy-tailed full-rank tensor.

### 3.3 Ordering matters and it is not symmetric

```
WRONG ORDER:  quantize the original cache → lose accuracy → try to restructure to recover it
RIGHT ORDER:  restructure → the cache is now small and clean → quantize (or don't)
```

The wrong order's mistake is not recoverable: once you have rounded the K/V tensors, you
cannot re-derive a low-rank latent from them, because the rounding error is not in the
low-rank subspace the model wants. Restructuring must happen **at write time**, in the
forward pass, before anything is stored.

This is also why MLA is an *architecture change* and not a *cache-manager feature*. See §4.5.

---

## 4. DeepSeek's MLA, derived from scratch

### 4.1 The mechanism

`Multi-head Latent Attention`, DeepSeek-V2 (arXiv:2405.04434). Per layer, per token:

```
c_KV  = W_DKV · h            dim d_c      (shared by ALL heads — this is the cache)
k_R   = RoPE(W_DKR · h)      dim d_rope   (decoupled positional key — also cached)
k_h   = W_UK[h] · c_KV       dim d_h      (reconstructed, never stored)
v_h   = W_UV[h] · c_KV       dim d_h      (reconstructed, never stored)
```

Cache contents per token per layer: `(c_KV, k_R)` — a single latent plus a small
positional key, **shared across all heads**.

Compare to MHA, which stores `n_h` separate K's and `n_h` separate V's.

### 4.2 The bytes — worked arithmetic from real configs

**DeepSeek-V2-Lite** (`kv_lora_rank=512`, `qk_rope_head_dim=64`,
`qk_nope_head_dim=128`, `num_attention_heads=16`, `num_key_value_heads=16`,
`hidden_size=2048`, `num_hidden_layers=27`):

```
MHA  : 2 (K,V) × 16 heads × 128 dim × 2 B = 8,192 B / token / layer
MLA  : (512 + 64) × 2 B                    = 1,152 B / token / layer
       → 7.11× smaller  (85.9% reduction)
```

**DeepSeek-V2** (full: `kv_lora_rank=1536`, `qk_rope_head_dim=64`,
`num_attention_heads=128`, `hidden_size=5120`, 60 layers):

```
MHA  : 2 × 128 × 128 × 2 B = 65,536 B / token / layer
MLA  : (1536 + 64) × 2 B  =  3,200 B / token / layer
       → 20.5× smaller  (95.1% reduction)
```

The paper's headline number, quoted from the abstract: DeepSeek-V2 *"reduces the KV cache
by 93.3%"* relative to DeepSeek 67B, and *"boosts the maximum generation throughput to
5.76 times"* (236B total / 21B activated parameters, 128K context).

The exact percentage depends on the baseline model's KV-head count, which is why I show
the arithmetic rather than only the headline. The mechanism is what matters, and it
generalizes to *your* model:

**What would MLA give `Qwen2.5-1.5B` if it were trained that way?**

```
Qwen2.5-1.5B today (GQA, 2 KV heads):
  2 × 2 × 128 × 2 B = 1,024 B / token / layer      →  28,672 B / token (28 layers)

MLA with d_c = 128, d_rope = 64  (a plausible choice for this width):
  (128 + 64) × 2 B = 384 B / token / layer        →  10,752 B / token
  → 2.67× smaller than the GQA cache

MLA with d_c = 256, d_rope = 64:
  (256 + 64) × 2 B = 640 B / token / layer        →  17,920 B / token
  → 1.60× smaller than the GQA cache
```

Note the gap: **MLA's headline 7–20× is measured against MHA. Your model is already GQA,
so the honest headroom is 1.6–2.7×, not 7–20×** — and only if the model is *retrained* to
use it. This is worth internalizing before comparing any paper's KV-cache number to your
own baseline.

### 4.3 Decoupled RoPE — why a separate key

If all heads shared one latent, the latent would have to carry the positional signal too.
But RoPE's rotation is applied per-head to the *nope* dimensions, and the rope
dimensions are not shared across heads in the usual formulation.

The fix: keep a **separate, small, per-head-free rope key** `k_R ∈ R^{d_rope}` (64 dims in
DeepSeek-V2), which *is* shared, and let `c_KV` carry only the content. The attention
score then has two parts:

```
q_h · k_h  =  (q_h^nope · W_UK[h]^nope c_KV)   +   (q_h^rope · k_R)
                 └──────── content, via the latent ────────┘   └─ position, decoupled ─┘
```

`k_R` costs 64 elements against the latent's 512–1536 — a small price for letting the
bulk of the cache be head-shared.

### 4.4 Absorb — why MLA gets *faster*, not just smaller

The second trick. Start from the score for head `h`:

```
z_h = q_h · k_h = q_h · (W_UK[h] c) = (W_UK[h]^T q_h) · c
```

Define `q̃_h = W_UK[h]^T q_h` — a query in **latent space**, dim `d_c`. Then:

```
z_h = q̃_h · c        ← attention against the CACHE, in d_c dimensions
```

And the output:

```
o_h = Σ_j p_h,j v_h,j = Σ_j p_h,j (W_UV[h] c_j) = W_UV[h] · (Σ_j p_h,j c_j)
```

so `W_UV` factors out of the sum and can be folded into the output projection `W_O`.

**Consequence:** at decode time you never materialize any per-head `K` or `V`. You read
the cache once as an `N × d_c` matrix, run attention with `d_c`-dimensional queries, and
apply one absorbed GEMM at the end.

```
memory traffic per decode step:  N × d_c   instead of   N × n_h × d_h
```

Decode is **memory-bandwidth bound**, so this is where the throughput comes from — the
5.76× in the DeepSeek-V2 abstract. Making this fast in practice is what **FlashMLA**
(DeepSeek's own kernel library) is for. The algebra is exact; the engineering is a
Triton/CUDA kernel problem, not a Python problem.

**Important consequence of "exact":** absorb is *exact algebra*, but it is exact only
because the model was trained with `k_h = W_UK[h] c`. See next section.

### 4.5 Why MLA cannot be retrofitted onto Qwen2.5-1.5B

This is the crux, and it is a *rank* argument, not an engineering-difficulty argument.

Forcing `k_h = W_UK[h] c` with `c ∈ R^{d_c}` and `d_c < n_h·d_h` says: **the entire key
space across all heads is constrained to a `d_c`-dimensional subspace.** For a model
trained with GQA, the per-head keys are *not* low-rank in that sense — they carry
information that the model demonstrably uses, and that information does not live in any
`d_c`-dimensional subspace.

Quantitatively, the approximation error of the best rank-`d_c` approximation to the true
key set is the energy in the discarded singular directions:

```
error = Σ_{i > d_c} σ_i²        (σ = singular values of the stacked key matrix)
```

For a model that was never trained with that constraint, `σ_{d_c+1}, …` are **not**
small — typically the tail carries a meaningful fraction of the energy. The output
degrades immediately and visibly. There is no way around this except training (or
distillation, which is what DeepSeek did when producing the MLA/DSA checkpoints).

**Two honest consequences:**

1. You cannot get MLA's benefit on `Qwen2.5-1.5B` by editing the serving stack. The
   benefit lives in the *weights*, and the weights were trained without it.
2. But the *idea* behind it — restructure so the cache holds a sufficient statistic,
   then store it at the highest precision you can afford — transfers to models you
   cannot retrain. §5.3 and §8.

### 4.6 What DeepSeek added afterwards

| Release | Addition | What it does for the cache |
|---|---|---|
| V2 (May 2024) | **MLA** + decoupled RoPE + absorb | Structural: latent cache, 93.3% smaller |
| V3 (Dec 2024) | MLA latent stored in **FP8** | Numeric: further ~2× on top of the structural win |
| V3.2-Exp (Sep 2025) | **DSA** — DeepSeek Sparse Attention | *Read-side*: an indexer picks top-k positions; the cache stays **BF16** |

DSA is the one that most literally matches your phrasing — *"preserve output quality
with the highest BF16 accuracy"*. Its thesis is: **do not compress the cache at all;
compress the *reads*.** The cache stays BF16 (zero quality cost by construction) and a
lightweight indexer selects which positions each query actually attends to. You get
throughput from skipping memory traffic, not from storing less.

vLLM 0.31 (your version) has all of it:
`DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM` (with
`DeepseekV32IndexerCache` in `model_executor/models/deepseek_v2.py`), plus the
`fp8_ds_mla` and `nvfp4_ds_mla` cache dtypes, and an `mla/` attention-backend directory.

None of it is servable on a 6 GB card: V2-Lite is 15.7B (≈31 GB in BF16), V3 is 671B.
But the *code* is right there on your disk to read, and that is the single best
reference implementation available to you.

---

## 5. The full menu

### 5.1 Architectural (require training)

| Technique | Mechanism | Cache reduction | Quality risk | Servable on 6 GB? |
|---|---|---|---|---|
| MHA → MQA | 1 KV head for all query heads | `n_h`× | Moderate | n/a |
| MQA → GQA | KV head shared by a group | `n_h / n_kv`× | Low | **Your model is already here (12:2)** |
| **MLA** | Store shared low-rank latent | 7–20× vs MHA | None *if trained* | No (needs ≥16B) |
| Cross-layer KV sharing | Reuse KV across layers | `n_layers / n_shared`× | Low–moderate | Only if retrained |
| Hybrid Mamba/Attention | KV from a fraction of layers | 2–8× | Moderate | Model-dependent |

### 5.2 Numeric (post-hoc, model-agnostic — this is your table)

vLLM 0.31's `CacheDType` literal, verbatim from `vllm/config/cache.py`:

```
auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla,
nvfp4_ds_mla, turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc,
turboquant_3bit_nc, int4_per_token_head, int8_per_token_head, fp8_per_token_head,
nvfp4, nvfp4_4over6
```

Projected cache footprint for `Qwen2.5-1.5B` at 28 layers (approximate — excludes the
per-token scale overhead, which is a few percent):

| dtype | Bytes/token | vs BF16 | Sequences that fit in ~1.9 GB (at 4096 ctx) |
|---|---|---|---|
| `auto` (bf16) | 28,672 | 1.00× | **~16** |
| `fp8` / `fp8_e4m3` | 14,336 | 2.0× | ~33 |
| `int8_per_token_head` | 14,336 + scales | ~2.0× | ~32 |
| `nvfp4` | 8,064 | 3.56× | ~59 |
| `int4_per_token_head` | 7,392 | 3.88× | ~64 |
| `turboquant_3bit_nc` | 10,752 | 2.67× | ~44 |

**The FP8 caveat, stated precisely:** your GPU is `sm_86` (Ampere). It has **no FP8
tensor cores** (those start at `sm_89`/Ada and `sm_90`/Hopper). That does **not**
automatically disqualify FP8 *KV cache*, because in FP8-KV-cache designs the cache is
stored FP8 and dequantized to BF16 inside the attention kernel — the win is memory
bandwidth and capacity, not FP8 arithmetic. But whether vLLM 0.31's chosen attention
backend for Qwen2 on `sm_86` actually provides an FP8 KV path is an **empirical
question**, not something I will assert from a source file. §7.2 exists to answer it.

### 5.3 Token-level structural reuse (post-hoc, model-agnostic)

These change *which* tokens are stored, or *how they are grouped*, without retraining.

| Technique | Mechanism | Cache reduction | Quality risk |
|---|---|---|---|
| **Attention sinks** | The first few tokens absorb a large share of attention mass; dropping them is catastrophic (Xiao et al., 2024) | — | **Must never be dropped** |
| **StreamingLLM** | Sliding window + re-anchored sinks | Unbounded | Degrades on retrieval tasks |
| **H2O** | Evict tokens with lowest accumulated attention score | Tunable | Degrades on long-context recall |
| **SnapKV** | Observation window → keep top-k important tokens, pool the rest | Tunable | Better than H2O at same budget |
| **PyramidKV** | Non-uniform budget across layers | Tunable | Better than uniform |
| **Grouping for quantization** (K-Flash style) | Reorder tokens so similar ones are adjacent → they can **share a quantization scale** → lower error at the same bit-width | Same bytes, **better fidelity** | Low if grouping is right |

**The grouping idea is the model-agnostic shadow of MLA.** MLA says: *the cache should
hold a low-rank sufficient statistic*. K-Flash says: *the cache should hold tokens that
are numerically similar, so that quantization error can be shared away*. Same underlying
instinct — restructure so the representation is well-conditioned, then compress.

**The hard part, stated honestly:** physically reordering tokens inside vLLM's paged KV
cache is invasive. The page table maps logical blocks to physical blocks; permuting
tokens means (a) carrying RoPE positions in a side table, since position is no longer
implicit in index order, and (b) making sure the causal mask still excludes future
tokens. It is mathematically doable — attention over a permuted set with explicit
per-token positions is identical to attention over the original set — but it touches
vLLM's hottest, least-tested code path.

**A cheaper approximation that gets most of the benefit:** skip the reordering and use
**per-token or per-head scales** (`int8_per_token_head`, `int4_per_token_head`,
`fp8_per_token_head`). Those already exist in vLLM 0.31 and get you the "each token's
magnitudes are scaled by its own range" benefit without touching the page table.

### 5.4 Read-side sparsity (capacity-neutral, latency-positive)

Keep the full cache, read less of it.

| Technique | Mechanism | Helps |
|---|---|---|
| **DSA** (DeepSeek V3.2) | Learned indexer selects top-k KV positions per query | Throughput, BF16 preserved |
| Quest | Query-aware page selection using a min/max summary | Throughput on long context |
| MInference | Offline-computed per-head sparse patterns | Throughput |
| Block-sparse attention | Attend only to selected blocks | Throughput |

On a 6 GB card, read-side sparsity does **not** let you fit more sequences (the cache is
still resident). It raises throughput per sequence. It becomes valuable when the model
is much larger than your VRAM and you are paging, or when you are already at the
concurrency ceiling and want more tokens/sec.

### 5.5 Operational (free quality)

| Technique | Mechanism | Why it is free |
|---|---|---|
| **Prefix caching** | Reuse the cache across requests sharing a prefix | The cache is a pure function of the prefix (§1.3) — this is *definitionally* free |
| **PagedAttention** | Non-contiguous block allocation | Removes fragmentation waste |
| **Chunked prefill** | Split long prefills | Lowers *peak transient* memory, not steady-state cache |
| `max_num_seqs` | Cap concurrency | Trades throughput for KV headroom |

You already have all four in `BENCHMARK-OBSERVABILITY`. For a chat workload with a shared
system prompt, prefix caching is frequently worth more than any quantizer, and costs
nothing in quality.

---

## 6. What the best method is on this box

### 6.1 The constraint set

```
GPU            RTX 3050 6GB, sm_86 (Ampere)  → BF16 native, NO FP8 tensor cores
Model          Qwen2.5-1.5B-Instruct — GQA 12:2, 28 layers, bf16
Engine         vLLM 0.31.0 (WSL2), V1 engine, in-process via VLLM_ENABLE_V1_MULTIPROCESSING=0
Budget         ~1.9 GB for KV ≈ 28,672 B/token ≈ 67K tokens ≈ 16 full-length sequences
Serving goal   more concurrent users at preserved output quality
```

### 6.2 The ranked recommendation

Ordered by expected value per unit of effort, for *this* box.

**Tier 1 — do this first (near-zero risk, large or free wins)**

1. **Prefix caching.** Already implemented in your other project; for chat workloads with
   a shared system prompt it is the single largest concurrency win available, at exactly
   zero quality cost. Verify it actually hits on your workload before spending effort
   anywhere else.
2. **The KV dtype ladder.** Sweep `auto` → `fp8` → `int8_per_token_head` →
   `int4_per_token_head` → `nvfp4` → `turboquant_3bit_nc` and find the knee of the
   fidelity-vs-capacity curve *for your model and your prompts*. The answer is an
   empirical property of `Qwen2.5-1.5B` under your traffic, not a rule of thumb, and
   it is cheap to measure (§7).
3. **Sweep the operating point, not just the dtype.** `gpu_memory_utilization` and
   `max_num_seqs` interact with the cache budget. A 2× compressed cache at a higher
   utilization is not the same product as a 1× cache at a lower one, and the latency
   trade-off differs.

**Tier 2 — do this if Tier 1's fidelity curve degrades faster than expected**

4. **Per-token/per-head scales** rather than per-tensor. If `int8_per_token_head`
   holds fidelity but plain `int8` does not, that tells you the problem was scale
   granularity, not bit-width — which points at grouping (§5.3) as the next lever.
5. **Token eviction (SnapKV-style).** If your prompts are long and the fidelity cliff
   shows up at long context, evicting low-importance tokens with a preserved attention
   sink attacks the problem directly. Higher implementation cost; treat as Tier 2
   because it needs a vLLM hook, not just a flag.

**Tier 3 — the research arm**

6. **Structural grouping (K-Flash-style) before quantization.** This is the arm that
   directly tests *your* claim: does restructuring preserve output quality better than
   uniform quantization? Same bytes, better fidelity — if the claim holds, this is the
   most interesting result in the project. It is also the most invasive (§5.3), so it
   comes after the cheap wins have been banked.

**Not available here, and worth building anyway as a reference**

7. **MLA, from scratch, in PyTorch.** Not servable, but implementable and verifiable on
   any hardware: latent cache + decoupled RoPE + absorb, with tests that prove
   (a) absorb is numerically identical to materializing per-head K/V, (b) BF16-latent
   error ≪ FP8-original-K error, (c) the byte reduction matches §4.2. This is the piece
   that makes the project *about* the DeepSeek idea rather than about vLLM flags, and
   it gives you a permanent, fast, CPU-only regression test for the mechanism.

### 6.3 What I will not claim

- No DeepSeek MLA model is servable on this card. Any plan that depends on one is wrong.
- FP8 on `sm_86` is not a compute speedup; if it works it is a *capacity* win only.
- A perplexity delta on a small corpus is a proxy, not a task-quality guarantee.
- The "16 sequences" figure is an upper bound from a static memory calculation. The real
  ceiling is what vLLM's block pool reports, and it is lower.

---

## 7. The process

The protocol, in order. Each step is a gate: do not start the next until this one
produces a number you trust.

### 7.1 Establish the control

- `auto` (BF16) cache, `Qwen2.5-1.5B-Instruct`, `max_model_len=4096`,
  `gpu_memory_utilization=0.80`, `enforce_eager=true` (so step timings mean something).
- A **fixed prompt set** — long enough to fill the context, diverse enough to be
  representative, and *pinned* to a file so every arm sees identical input.
- A **fixed corpus** for perplexity — a fixed slice of a public dataset, pinned by
  checksum.
- Fixed sampling: `temperature=0`, `seed` fixed, `min_tokens=max_tokens`,
  `ignore_eos=true`. (You already do this in `inferbench.engine.build_sampling_params` —
  reuse that exact pattern.)
- Record: every greedy output, and the per-token logprobs.

Without this step, every later number is unreproducible and the comparison is
meaningless.

### 7.2 Capability probe

Enumerate, on the actual hardware, which `CacheDType` values vLLM 0.31 will accept for
this model and which actually **initialize** rather than fail at startup. Some will not:
FP8 paths have compute-capability guards in the attention backend, and `sm_86` is
below the bar for several of them.

Record the answer as a table: `dtype → {accepted?, initialized?, block size, notes}`.
This is a five-minute probe that prevents an entire afternoon of chasing a startup error
that was never going to work.

### 7.3 Memory accounting

From vLLM internals (`kv_cache_manager.block_pool` in 0.31 — your `inferbench` inspector
already reads this):

```
tokens_resident  = block_pool.num_free_blocks × block_size
bytes_per_token  = 2 × n_kv_heads × head_dim × dtype_size × n_layers
max_seqs@4096    = tokens_resident / 4096
```

Cross-check against the static calculation in §1.5. If they disagree by more than
fragmentation explains, something is wrong with the model config you are using.

### 7.4 Fidelity measurement

Per your instruction: **exact greedy match + perplexity delta.** Precisely defined:

**(a) Exact greedy match**

- Greedy decode, identical prompts, identical sampling parameters.
- A response matches if it is **token-identical** to the BF16 control.
- Report: match rate, and for the non-matches, the **divergence position** as a
  fraction of the response length. This matters more than the match rate: a divergence
  at token 3 is a different failure from a divergence at token 400, and a scheme can
  have a low match rate while being perfectly usable (e.g. it changes a synonym).
- Also report a **token-level agreement rate** (fraction of positions where the argmax
  token matches) as a finer-grained companion. This is the metric that actually
  separates "harmless" from "broken", and it degrades gracefully where exact match is
  binary.

**(b) Perplexity delta**

- Fixed corpus, same tokenizer, same context window, same forward pass.
- `ΔPPL = PPL_compressed − PPL_bf16`, and the relative form `ΔPPL / PPL_bf16`.
- Report both absolute and relative. Relative is the one that transfers across corpora.

Neither of these is a task-quality guarantee. Say so in the report.

### 7.5 Serving sweep

For each dtype that survived §7.2:

- Concurrency ladder 1 → 2 → 4 → 8 → 16 → 32 → 64 (go past the wall).
- At each point: throughput (tok/s), TTFT p50/p95, TPOT p50/p95, preemption count.
- Find the **concurrency ceiling** — the point where preemption starts or the engine
  fails. That number *is* the "how many users can we serve" answer.

Your `inferbench` harness already does most of this (`experiments.yaml` with
`concurrency: [1, 4, 8, 16, 32]`, `repeats: 3`). Reuse the pattern; extend the axis
to include the dtype.

### 7.6 The deliverable: one trade curve

The output of the project is a single chart and a single table:

```
                 users served
                 ▲
   16 (bf16)     │
    33 (fp8)     │          ●
    44 (tq3)     │                    ●
    59 (nvfp4)   │                              ●
                 │  ← fidelity (exact match %)
                 └──────────────────────────────────►
```

plus the fidelity numbers at each point, and **one recommendation**: the dtype that
maximizes users-served subject to a fidelity floor you state explicitly.

**State the fidelity floor before you run the sweep.** Otherwise you will pick the
answer after seeing the curve, and the recommendation will be worthless. A defensible
floor might be "exact match ≥ 95% and relative ΔPPL ≤ 1% on the pinned corpus".

### 7.7 Config as data

Same discipline as `BENCHMARK-OBSERVABILITY`: the sweep is a YAML file, not a Python
loop. Each arm is an entry. Adding a dtype or a concurrency point is an edit to the
config, not a code change.

---

## 8. How DeepSeek's MLA is actually helpful here

Stripped of the parts you cannot use on a 6 GB card, MLA contributes four things.

**1. It reframes the problem.** The question stops being *"how many bits per element?"*
and becomes *"what is the minimal representation of the prefix from which this model's
attention can be computed?"* That reframe is why the whole field has a quality cliff at
"structural" vs "numeric" rather than a smooth bits-vs-quality curve.

**2. It explains why BF16 can be enough.** The claim that restructuring makes BF16 safe
is not folklore; it follows from the latent being the model's own sufficient statistic
(§3.2). This is the single most useful idea to carry into a project that cannot
retrain anything.

**3. It shows that changing the cache means changing the kernel.** Absorb (§4.4) is
exact algebra, but you only get the speed if the attention kernel reads the latent
directly. This is the lesson for any structural work you do inside vLLM: a cache-format
change without a matching kernel change tends to make things *slower*, because you have
added a data movement step. vLLM's own `mla/` backend directory and DeepSeek's
`FlashMLA` are the reference.

**4. It defines the ceiling for what you can get without retraining.** MLA's 7–20× is
against MHA. Your model is already GQA. Realistic post-hoc headroom on
`Qwen2.5-1.5B` is the dtype ladder (2×–4×) plus eviction (unbounded but lossy), plus
prefix caching (free). The MLA number is the thing you are *not* getting, and knowing
that precisely is what keeps you from chasing it.

**The one-liner:** DeepSeek's MLA is the reason we understand that *restructuring the
cache is what buys you the right to keep high precision* — and the model-agnostic
version of that lesson, which you can actually implement here, is
**per-token/per-head scales and importance-grouped quantization**, plus the operational
freebie of prefix caching.

---

## 9. Verification log

| Claim | Source | Status |
|---|---|---|
| DeepSeek-V2 "reduces the KV cache by 93.3% … boosts max generation throughput to 5.76×" | arXiv:2405.04434 abstract | read directly |
| DeepSeek-V2-Lite: `kv_lora_rank=512`, `qk_rope_head_dim=64`, `qk_nope_head_dim=128`, `num_attention_heads=16`, `num_key_value_heads=16`, `hidden_size=2048`, `num_hidden_layers=27` | `huggingface.co/deepseek-ai/DeepSeek-V2-Lite/config.json` | read directly |
| Qwen2.5-1.5B-Instruct: `hidden_size=1536`, `num_hidden_layers=28`, `num_attention_heads=12`, `num_key_value_heads=2`, `torch_dtype=bfloat16`, `max_position_embeddings=32768`, `rope_theta=1e6` | local HF cache `config.json` | read directly |
| vLLM 0.31.0, torch 2.13.0+cu130 | WSL `/home/anupam/.venv` | probed |
| vLLM 0.31 `CacheDType` literal (18 values incl. `int8_per_token_head`, `int4_per_token_head`, `nvfp4`, `turboquant_*`, `fp8_ds_mla`, `nvfp4_ds_mla`) | `vllm/config/cache.py` | read directly |
| vLLM 0.31 ships `DeepseekV2/V3/V32ForCausalLM`, `DeepseekV32IndexerCache` (DSA), `mla/` backend | `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/` | read directly |
| GPU = RTX 3050 6GB Laptop, compute capability **sm_86** | `torch.cuda.get_device_capability(0)` | probed |
| DeepSeek-V2 (full) dims `kv_lora_rank=1536`, `qk_rope_head_dim=64`, `num_attention_heads=128`, `hidden_size=5120` | DeepSeek-V2 paper / model card | from the paper, not re-read from disk |
| No built-in token eviction (SnapKV/H2O) in vLLM 0.31's `kv_cache_manager` — only block eviction for preemption | `vllm/v1/core/kv_cache_manager.py` | grepped |
| Per-byte accuracy figures in §3.1 | derived from IEEE-754 mantissa widths (BF16 8, E4M3 4, E5M2 3) | arithmetic |
| "67,000 tokens / ~16 sequences" | derived from 6,144 MiB, 0.80 utilization, 1.54B params | estimate — upper bound, must be confirmed against vLLM's block pool in §7.3 |
| `fp8` KV cache viability on sm_86 | — | **unverified — see §7.2** |

---

## 10. Open questions

1. **What is the real workload?** Long-context analysis, short chat, code, or
   summarization? This single fact reorders the entire recommendation list:
   - Shared system prompt → **prefix caching** is the top lever, quantizers are second.
   - Long contexts → **eviction** and **read-side sparsity** matter more than dtype.
   - Short chats → the KV cache is small anyway and **dtype ladder** is the whole story.
2. **Is `Qwen2.5-1.5B` a stand-in?** If you are really serving a 7B–70B model elsewhere
   and this box is the lab, then the *method* matters more than the numbers, and the MLA
   reference implementation (§6.2 item 7) deserves to be Tier 1 rather than Tier 3.
3. **Is the goal capacity or latency?** §5.4 (read-side sparsity) buys latency;
   §5.2 (dtype) buys capacity. They are different products and the answer differs.
4. **What is the fidelity floor?** Needed *before* the sweep (§7.6). My default would be
   "exact greedy match ≥ 95% and relative ΔPPL ≤ 1%", but you should set it.
