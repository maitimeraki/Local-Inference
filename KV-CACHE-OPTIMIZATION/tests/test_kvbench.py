"""CPU-only checks for the weight construction, plus the GPU decode kernel."""

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from kvbench import config, kernels, weights


def _random_layer_state(seed: int = 0) -> dict[str, torch.Tensor]:
    """One layer's worth of KV projections at the pinned geometry."""
    g = torch.Generator().manual_seed(seed)
    kv_rows = config.NUM_KV_HEADS * config.HEAD_DIM
    return {
        "k_proj.weight": torch.randn(kv_rows, config.HIDDEN_SIZE, generator=g, dtype=torch.float64),
        "v_proj.weight": torch.randn(kv_rows, config.HIDDEN_SIZE, generator=g, dtype=torch.float64),
    }


def _reconstruct(sub: weights.MLASubspace) -> torch.Tensor:
    """Rebuild the stacked [W_K ; W_V] approximation from the latent stack."""
    head = config.HEAD_DIM
    blocks = []
    for g in range(config.NUM_KV_HEADS):
        blocks.append(sub.w_uk[g] @ sub.w_dkv)
    for g in range(config.NUM_KV_HEADS):
        blocks.append(sub.w_uv[g] @ sub.w_dkv)
    return torch.cat(blocks, dim=0)


def test_softmax_scale_is_head_dim_not_rank():
    assert config.ATTENTION_SCALE == config.HEAD_DIM ** -0.5
    # the scale must not move with the latent rank
    assert config.ATTENTION_SCALE != 64 ** -0.5
    assert config.ATTENTION_SCALE != 512 ** -0.5


def test_latent_score_reproduces_native_score():
    """(W_UK^T q) . c scaled by head_dim**-0.5 equals q . (W_K x) scaled the
    same way -- and the rank-scaled variant does not."""
    state = _random_layer_state()
    sub = weights.joint_svd(state, rank=config.LOSSLESS_RANK)

    g = torch.Generator().manual_seed(7)
    q = torch.randn(config.HEAD_DIM, generator=g, dtype=torch.float64)
    x = torch.randn(config.HIDDEN_SIZE, generator=g, dtype=torch.float64)

    head = config.HEAD_DIM
    native = (q @ state["k_proj.weight"][0:head] @ x) * config.ATTENTION_SCALE
    latent_score = (sub.w_uk[0].T @ q) @ (sub.w_dkv @ x)

    assert torch.allclose(native, latent_score * config.ATTENTION_SCALE, atol=1e-9)
    assert not torch.allclose(
        native, latent_score * (config.LOSSLESS_RANK ** -0.5), atol=1e-6
    )


def test_rank_512_is_lossless():
    state = _random_layer_state()
    sub = weights.joint_svd(state, rank=config.LOSSLESS_RANK)
    head = config.HEAD_DIM
    w_k = state["k_proj.weight"]
    w_v = state["v_proj.weight"]

    for g in range(config.NUM_KV_HEADS):
        assert torch.allclose(sub.w_uk[g] @ sub.w_dkv, w_k[head * g : head * g + head], atol=1e-8)
        assert torch.allclose(sub.w_uv[g] @ sub.w_dkv, w_v[head * g : head * g + head], atol=1e-8)

    # float64 SVD roundoff, so "zero" is a tolerance and not an equality
    assert (sub.rel_err_k < 1e-9).all()
    assert (sub.rel_err_v < 1e-9).all()


def test_truncation_error_grows_as_rank_falls():
    """The best rank-r approximation is monotone in r, so the discarded
    energy can only grow as the rank falls."""
    state = _random_layer_state()
    a = torch.cat([state["k_proj.weight"], state["v_proj.weight"]], dim=0)
    previous = None
    for rank in config.RANKS:  # descending
        err = (_reconstruct(weights.joint_svd(state, rank=rank)) - a).norm() / a.norm()
        if previous is not None:
            assert err >= previous - 1e-12  # the rank fell, so the error grew
        previous = err
    assert previous > 0  # the smallest rank is not lossless


def test_build_mla_weights_round_trip(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    kv_rows = config.NUM_KV_HEADS * config.HEAD_DIM
    q_rows = config.NUM_ATTENTION_HEADS * config.HEAD_DIM
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(8, config.HIDDEN_SIZE),
            "model.norm.weight": torch.zeros(config.HIDDEN_SIZE),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(q_rows, config.HIDDEN_SIZE),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(kv_rows, config.HIDDEN_SIZE),
            "model.layers.0.self_attn.k_proj.bias": torch.randn(kv_rows),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(kv_rows, config.HIDDEN_SIZE),
            "model.layers.0.self_attn.v_proj.bias": torch.randn(kv_rows),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(config.HIDDEN_SIZE, q_rows),
            "model.layers.0.mlp.gate_proj.weight": torch.zeros(4, config.HIDDEN_SIZE),
        },
        str(model_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )

    path = weights.build_mla_weights(model_dir, rank=config.LOSSLESS_RANK, out_dir=tmp_path / "out")
    assert path.name == f"mla_rank{config.LOSSLESS_RANK}.safetensors"

    out = weights.load_mla_weights(path)
    prefix = "model.layers.0.self_attn."
    for dropped in ("k_proj.weight", "v_proj.weight", "k_proj.bias", "v_proj.bias"):
        assert prefix + dropped not in out

    assert out[prefix + weights.DKV_KEY].shape == (config.LOSSLESS_RANK, config.HIDDEN_SIZE)
    assert out[prefix + weights.UK_KEY].shape == (
        config.NUM_KV_HEADS,
        config.HEAD_DIM,
        config.LOSSLESS_RANK,
    )
    assert out[prefix + weights.UV_KEY].shape == (
        config.NUM_KV_HEADS,
        config.HEAD_DIM,
        config.LOSSLESS_RANK,
    )
    assert out[prefix + weights.V_BIAS_KEY].shape == (config.NUM_KV_HEADS, config.HEAD_DIM)

    # untouched weights survive the rewrite
    assert out["model.layers.0.mlp.gate_proj.weight"].shape == (4, config.HIDDEN_SIZE)

    # the written stack still reconstructs the original projections
    original = weights.load_model_state(model_dir)
    head = config.HEAD_DIM
    rebuilt_k = out[prefix + weights.UK_KEY][0].double() @ out[prefix + weights.DKV_KEY].double()
    target = original[prefix + "k_proj.weight"][0:head].double()
    assert (rebuilt_k - target).norm() / target.norm() < 1e-5


# --- the decode kernel ----------------------------------------------------


def _decode_inputs(num_seqs: int, seq_len: int, rank: int, dtype=torch.float32, seed: int = 0):
    """Random decode-step inputs. Every token past a sequence's seq_len is
    replaced with garbage, so a kernel that forgets to mask the tail gets a
    different answer rather than a slightly smaller one."""
    g = torch.Generator().manual_seed(seed)
    q_hat = torch.randn(num_seqs, config.NUM_ATTENTION_HEADS, rank, generator=g)
    latent = torch.randn(num_seqs, seq_len, rank, generator=g)
    seq_lens = torch.randint(1, seq_len + 1, (num_seqs,), generator=g, dtype=torch.int32)
    keep = torch.arange(seq_len).unsqueeze(0) < seq_lens.unsqueeze(1)
    latent = torch.where(keep.unsqueeze(-1), latent, torch.randn(latent.shape, generator=g))
    return q_hat.to(dtype), latent.to(dtype), seq_lens


def _pack_latent(latent: torch.Tensor, seq_lens: torch.Tensor, block_size: int):
    """Scatter a contiguous latent into the paged layout the kernel reads.

    Each sequence's blocks are allocated as a contiguous run, so the block
    table is not the identity and a kernel that ignored it would read the
    wrong sequence's tokens."""
    num_seqs, seq_len, rank = latent.shape
    max_blocks = (seq_len + block_size - 1) // block_size
    padded = torch.zeros(num_seqs, max_blocks * block_size, rank, dtype=latent.dtype)
    padded[:, :seq_len] = latent
    keep = torch.arange(max_blocks * block_size).unsqueeze(0) < seq_lens.unsqueeze(1)
    packed = torch.where(keep.unsqueeze(-1), padded, torch.randn(padded.shape, dtype=latent.dtype))
    cache = packed.reshape(num_seqs, max_blocks, block_size, rank)
    cache = cache.reshape(num_seqs * max_blocks, block_size, rank).contiguous()
    block_table = torch.arange(num_seqs * max_blocks, dtype=torch.int32).reshape(num_seqs, max_blocks)
    return cache, block_table


def test_triton_reference_matches_sdpa():
    """The reference is softmax(q.k*scale) @ latent, which is exactly what
    scaled_dot_product_attention computes when the values are the latent."""
    q_hat, latent, seq_lens = _decode_inputs(num_seqs=2, seq_len=32, rank=64)
    ref = kernels.mla_decode_reference(q_hat, latent, seq_lens, config.ATTENTION_SCALE)
    for s in range(latent.shape[0]):
        n = int(seq_lens[s])
        q = q_hat[s].unsqueeze(0).unsqueeze(2)  # [1, heads, 1, r]
        kv = latent[s, :n].unsqueeze(0).unsqueeze(1)  # [1, 1, n, r]
        want = F.scaled_dot_product_attention(q, kv, kv, scale=config.ATTENTION_SCALE)
        assert torch.allclose(ref[s], want[0, :, 0], atol=1e-5), f"sequence {s}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rank", [64, 128, 256, 512])
@pytest.mark.parametrize("num_seqs", [1, 3])
def test_triton_kernel_matches_reference(num_seqs, rank, dtype):
    seq_len = 128
    block_size = 16
    q_hat, latent, seq_lens = _decode_inputs(num_seqs, seq_len, rank, dtype=dtype)
    ref = kernels.mla_decode_reference(q_hat, latent, seq_lens, config.ATTENTION_SCALE)

    cache, block_table = _pack_latent(latent, seq_lens, block_size)
    got = kernels.mla_decode_triton(
        q_hat.cuda(),
        cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        config.ATTENTION_SCALE,
        config.HEAD_DIM,
    )

    assert got.shape == ref.shape
    # float32 is the plan's 1e-3 gate, and the kernel lands at ~4e-6.
    # bfloat16 rounds the probabilities for the tensor cores, so the kernel
    # carries bf16's 8-bit mantissa: ~2e-3 on contexts of magnitude 1-3.
    atol = 1e-3 if dtype == torch.float32 else 4e-3
    assert torch.allclose(got.cpu(), ref, atol=atol, rtol=0)


# --- the vLLM model class ------------------------------------------------


def test_model_registry_resolves_the_plugin():
    """register_plugin() puts Qwen2MLAForCausalLM where vLLM's model
    registry looks for architectures."""
    from types import SimpleNamespace

    from vllm.model_executor.models.registry import ModelRegistry

    from kvbench import models

    models.register_plugin()
    model_cls, arch = ModelRegistry.resolve_model_cls(
        "Qwen2MLAForCausalLM", SimpleNamespace(model_impl="auto")
    )
    assert arch == "Qwen2MLAForCausalLM"
    assert model_cls is models.Qwen2MLAForCausalLM


def test_model_kv_cache_spec_is_the_latent_geometry():
    """The cache holds r latent values per token per layer and no separate
    value cache -- 28 x 512 x 2 bytes per token at r=512."""
    from types import SimpleNamespace

    from vllm.config.cache import CacheConfig

    from kvbench import models

    hf = SimpleNamespace(
        hidden_size=config.HIDDEN_SIZE,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        num_key_value_heads=config.NUM_KV_HEADS,
        mla_rank=config.LOSSLESS_RANK,
    )
    layer = models.MLAAttentionLayer(
        config=hf,
        cache_config=CacheConfig(block_size=16),
        quant_config=None,
        prefix="self_attn",
    )
    spec = layer.get_kv_cache_spec(
        SimpleNamespace(cache_config=CacheConfig(block_size=16, cache_dtype="auto"))
    )

    assert spec.head_size == config.LOSSLESS_RANK
    assert spec.head_size_v == 0
    assert spec.num_kv_heads == 1
    assert spec.state_content_size_bytes == config.LOSSLESS_RANK * 2
    # bytes per token per layer, and the whole model
    per_token = spec.page_size_bytes // spec.block_size
    assert per_token == config.LOSSLESS_RANK * 2
    assert per_token * config.NUM_LAYERS == 28 * 512 * 2


def test_model_attention_layer_takes_the_checkpoint_tensors():
    """The MLA checkpoint's tensor names must land on the attention layer.

    The names are not cosmetic: vLLM's weight loader matches them against
    the layer's own parameter names, and a mismatch is an error at load
    time, not a warning.
    """
    from types import SimpleNamespace

    from vllm.config.cache import CacheConfig
    from vllm.model_executor.models.utils import AutoWeightsLoader

    from kvbench import models

    rank = config.LOSSLESS_RANK
    layer = models.MLAAttentionLayer(
        config=SimpleNamespace(
            hidden_size=config.HIDDEN_SIZE,
            num_attention_heads=config.NUM_ATTENTION_HEADS,
            num_key_value_heads=config.NUM_KV_HEADS,
            mla_rank=rank,
        ),
        cache_config=CacheConfig(block_size=16),
        quant_config=None,
        prefix="self_attn",
    )

    tensors = {
        weights.DKV_KEY: torch.randn(rank, config.HIDDEN_SIZE),
        weights.UK_KEY: torch.randn(config.NUM_KV_HEADS, config.HEAD_DIM, rank),
        weights.UV_KEY: torch.randn(config.NUM_KV_HEADS, config.HEAD_DIM, rank),
        weights.V_BIAS_KEY: torch.randn(config.NUM_KV_HEADS, config.HEAD_DIM),
    }
    loaded = AutoWeightsLoader(layer).load_weights(tensors.items())

    assert loaded == set(tensors)
    assert torch.equal(layer.kv_dkv_proj.weight, tensors[weights.DKV_KEY])
    assert torch.equal(layer.kv_uk_proj.weight, tensors[weights.UK_KEY])
    assert torch.equal(layer.kv_uv_proj.weight, tensors[weights.UV_KEY])
    assert torch.equal(layer.kv_v_bias, tensors[weights.V_BIAS_KEY])


# --- the engine driver ---------------------------------------------------


def test_server_blocks_for_charges_the_budget_per_arm():
    """Both arms get the same byte budget, so the arm with the smaller page
    gets more blocks -- and more sequences."""
    from kvbench import server

    budget = config.KV_BUDGET_BYTES
    gqa = server.EngineConfig(arm="gqa", kv_budget_bytes=budget, block_size=16)
    mla = server.EngineConfig(
        arm="mla", rank=512, kv_budget_bytes=budget, block_size=16
    )
    small = server.EngineConfig(
        arm="mla", rank=64, kv_budget_bytes=budget, block_size=16
    )

    # GQA holds 2 kv heads x 128 dims x (K and V) x 2 bytes per token.
    gqa_page = 16 * config.NUM_KV_HEADS * config.HEAD_DIM * 2 * 2
    assert server.blocks_for(gqa) == budget // (gqa_page * config.NUM_LAYERS)

    # The latent holds r values per token, and no separate value cache.
    assert server.blocks_for(mla) == budget // (16 * 512 * 2 * config.NUM_LAYERS)
    assert server.blocks_for(small) == budget // (16 * 64 * 2 * config.NUM_LAYERS)

    # r=512 costs exactly what GQA costs; r=64 costs an eighth, so it buys
    # about eight times the blocks.
    assert server.blocks_for(mla) == server.blocks_for(gqa)
    assert server.blocks_for(small) > 7 * server.blocks_for(mla)

    # and neither arm spends more than the budget
    for cfg in (gqa, mla, small):
        spent = server.blocks_for(cfg) * server.page_size_bytes(cfg) * config.NUM_LAYERS
        assert spent <= budget


def test_server_make_requests_builds_distinct_requests():
    from kvbench import server

    requests = server.make_requests(["alpha", "beta", "gamma"], 7)
    assert len(requests) == 3
    assert [r.prompt for r in requests] == ["alpha", "beta", "gamma"]
    assert all(r.output_tokens == 7 for r in requests)
    assert all(r.sampling_params.max_tokens == 7 for r in requests)
    assert len({r.request_id for r in requests}) == 3


def test_rank_512_latent_reproduces_native_gqa_attention():
    """At r=512 the latent attention must equal the native GQA attention.

    This is the claim the plan's section 2.3 gate rests on, checked at the
    algebra level: same query, same keys, same values, same softmax, same
    output. The end-to-end gate cannot make this distinction, because the
    arms differ in rounding and the model compounds the difference over 28
    layers.
    """
    state = _random_layer_state()
    sub = weights.joint_svd(state, rank=config.LOSSLESS_RANK)
    head, groups = config.HEAD_DIM, config.NUM_KV_GROUPS
    tokens = 8

    g = torch.Generator().manual_seed(11)
    x = torch.randn(tokens, config.HIDDEN_SIZE, generator=g, dtype=torch.float64)
    q = torch.randn(config.NUM_ATTENTION_HEADS, head, generator=g, dtype=torch.float64)

    # the native GQA attention over the context
    k = state["k_proj.weight"] @ x.T
    v = state["v_proj.weight"] @ x.T
    native = torch.empty(config.NUM_ATTENTION_HEADS, head, dtype=torch.float64)
    for h in range(config.NUM_ATTENTION_HEADS):
        group = h // groups
        score = (k[group * head : (group + 1) * head].T @ q[h]) * config.ATTENTION_SCALE
        probs = torch.softmax(score, dim=0)
        native[h] = probs @ v[group * head : (group + 1) * head].T

    # the latent attention: absorb the query, score the latent, expand
    latent = x @ sub.w_dkv.T
    q_hat = torch.stack(
        [sub.w_uk[h // groups].T @ q[h] for h in range(config.NUM_ATTENTION_HEADS)]
    )
    score = (q_hat @ latent.T) * config.ATTENTION_SCALE
    ctx = torch.softmax(score, dim=-1) @ latent
    absorbed = torch.stack(
        [sub.w_uv[h // groups] @ ctx[h] for h in range(config.NUM_ATTENTION_HEADS)]
    )

    assert torch.allclose(absorbed, native, atol=1e-9), (
        (absorbed - native).abs().max()
    )

# --- fidelity ------------------------------------------------------------


def test_fidelity_divergence_helpers():
    from kvbench import bench

    assert bench.first_divergence([1, 2, 3], [1, 2, 4]) == 2
    assert bench.first_divergence([1, 2], [1, 2]) is None
    assert bench.first_divergence([1, 2], [1, 2, 3]) == 2

    # A token the other arm never produced counts as a disagreement.
    assert bench.token_agreement([1, 2, 3], [1, 2, 3]) == 1.0
    assert bench.token_agreement([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    assert bench.token_agreement([1, 2], [1, 2, 3]) == pytest.approx(2 / 3)


# --- the report ----------------------------------------------------------


def _sample_rows():
    from kvbench import bench

    return [
        bench.Row(
            arm="gqa",
            rank=None,
            context_len=4096,
            repeat=0,
            kv_bytes_per_token=28672,
            kv_budget_bytes=2_040_109_465,
            blocks=4448,
            requests_served=17,
            tokens_per_request=4096,
            prefill_tok_per_s=900.0,
            decode_tok_per_s=640.0,
            decode_tok_per_s_per_seq=37.0,
            ttft_p50=800.0,
            ttft_p95=900.0,
            tpot_p50=27.0,
            tpot_p95=30.0,
            e2e_p50=3000.0,
            e2e_p95=3200.0,
            exact_match=None,
            token_agreement=None,
            rel_dppl=None,
            rel_err_k=None,
            rel_err_v=None,
        ),
        bench.Row(
            arm="mla",
            rank=256,
            context_len=4096,
            repeat=0,
            kv_bytes_per_token=14336,
            kv_budget_bytes=2_040_109_465,
            blocks=8896,
            requests_served=34,
            tokens_per_request=4096,
            prefill_tok_per_s=700.0,
            decode_tok_per_s=1200.0,
            decode_tok_per_s_per_seq=36.0,
            ttft_p50=820.0,
            ttft_p95=950.0,
            tpot_p50=28.0,
            tpot_p95=31.0,
            e2e_p50=3100.0,
            e2e_p95=3300.0,
            exact_match=0.96,
            token_agreement=0.98,
            rel_dppl=0.004,
            rel_err_k=0.011,
            rel_err_v=0.013,
        ),
    ]


def test_report_renders_the_arms_and_the_fidelity_floor(tmp_path):
    """The report is what the project delivers, so it has to carry the arms
    it measured and the floor they are judged against."""
    from kvbench import bench

    rows = _sample_rows()
    path = bench.render_report(rows, tmp_path)

    assert path.name == "report.md"
    text = path.read_text()

    # both arms are named
    assert "gqa" in text
    assert "mla" in text

    # the fidelity floor is drawn, so the reader sees what is being judged
    assert "95" in text and "exact_match" in text
    assert "rel_dppl" in text

    # the three tables are there
    assert "requests served" in text.lower()
    assert "rel_err_k" in text

    # and the row data made it in
    assert "28,672" in text
    assert "96.0%" in text


def test_report_writes_the_rows_and_the_config(tmp_path):
    from kvbench import bench

    bench.render_report(_sample_rows(), tmp_path)
    assert (tmp_path / "rows.csv").is_file()
    assert (tmp_path / "config.json").is_file()

    rows = bench.read_rows(tmp_path / "rows.csv")
    assert len(rows) == 2
    assert rows[0].arm == "gqa"
    assert rows[1].rank == 256
    assert rows[1].requests_served == 34


# --- the MLA attention algebra ------------------------------------------


def test_mla_attention_matches_the_plan_arithmetic():
    """The layer must compute PLAN.md section 3.1's arithmetic exactly.

    The end-to-end fidelity check cannot make this distinction: the arms
    differ in rounding, and a rounding difference compounds over 28 layers
    into a different token stream. So the algebra is checked against a
    direct PyTorch computation of the same formula.
    """
    from types import SimpleNamespace

    from vllm.config.cache import CacheConfig

    from kvbench import models

    torch.manual_seed(0)
    rank, tokens, block_size = 64, 8, 16
    layer = models.MLAAttentionLayer(
        config=SimpleNamespace(
            hidden_size=config.HIDDEN_SIZE,
            num_attention_heads=config.NUM_ATTENTION_HEADS,
            num_key_value_heads=config.NUM_KV_HEADS,
            mla_rank=rank,
        ),
        cache_config=CacheConfig(block_size=block_size),
        quant_config=None,
        prefix="self_attn",
    )
    for param in layer.parameters():
        with torch.no_grad():
            param.copy_(torch.randn_like(param) * 0.05)

    hidden = torch.randn(tokens, config.HIDDEN_SIZE)

    # the reference: the plan's formulas, one at a time
    # The plan's einsum strings write the projection as (g, r, d), but
    # section 3.2 stores it as (g, d, r) -- W_UK[g] is [head_dim, rank].
    # The contraction below is the one that matches the stored weights.
    # The sequence is a real axis: torch.einsum drops size-1 dimensions and
    # re-binds the labels, so a length-1 sequence would silently swap them.
    q = layer.q_proj(hidden).view(tokens, config.NUM_ATTENTION_HEADS, config.HEAD_DIM)
    w_uk = layer.kv_uk_proj.weight.repeat_interleave(config.NUM_KV_GROUPS, 0)
    # The plan writes this contraction "bhsd,grd->bhsr", which sums over the
    # KV groups and reads the projection transposed; the head-matched form
    # is "bhd,hdr->bhr". The layer's own contraction is the one checked here.
    q_hat = torch.einsum("bhd,hdr->bhr", q.float(), w_uk)
    latent = (hidden @ layer.kv_dkv_proj.weight.T).to(layer.cache_dtype)
    scores = torch.matmul(q_hat, latent.float().t()) * (config.HEAD_DIM**-0.5)
    causal = torch.arange(tokens).unsqueeze(1) >= torch.arange(tokens).unsqueeze(0)
    scores = scores.masked_fill(~causal.unsqueeze(1), float("-inf"))
    ctx = torch.matmul(torch.softmax(scores, dim=-1), latent.float())

    w_uv = layer.kv_uv_proj.weight.repeat_interleave(config.NUM_KV_GROUPS, 0)
    want = torch.einsum("bhr,hdr->bhd", ctx, w_uv) + layer.kv_v_bias.repeat_interleave(
        config.NUM_KV_GROUPS, 0
    )
    want = layer.o_proj(want.reshape(tokens, -1))

    # the layer, over a paged cache
    num_blocks = 4
    cache = torch.zeros(num_blocks, block_size, rank, dtype=layer.cache_dtype)
    layer.kv_cache = cache.unsqueeze(1)
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    slots = torch.arange(tokens, dtype=torch.int32)
    meta = models.MLAMetadata(
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, tokens], dtype=torch.int32),
        seq_lens=torch.tensor([tokens], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=tokens,
        max_query_len=tokens,
        max_seq_len=tokens,
        block_table_tensor=block_table,
        slot_mapping=slots,
        num_tokens=tokens,
        decode_req_idx=torch.empty(0, dtype=torch.long),
        decode_token_pos=torch.empty(0, dtype=torch.long),
        prefill_reqs=[(0, 0, tokens, tokens)],
        seq_lens_cpu_upper_bound=torch.tensor([tokens], dtype=torch.int32),
        positions=torch.arange(tokens),
    )

    latent_layer = layer.kv_dkv_proj(hidden.float())
    layer._write_cache(latent_layer, meta)
    got = layer._expand(layer._attend(layer._absorb(q), meta))
    got = layer.o_proj(got.reshape(tokens, -1))

    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=2e-2, rtol=0), (
        (got - want).abs().max()
    )

