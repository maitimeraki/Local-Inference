"""The vLLM side of the MLA arm.

A Qwen2 whose KV cache holds the rank-r latent instead of the per-head keys
and values. The algebra is PLAN.md section 3.1:

    q_hat[h] = W_UK[g]^T q_h            g = h // 6   (the query is absorbed)
    c_p      = x_p W_DKV^T                            (what the cache holds)
    ctx[h]   = sum_p softmax(q_hat[h] . c_p) c_p      (the attention)
    out[h]   = ctx[h] W_UV[g] + b_V[g]               (the value is re-expanded)

The softmax scale stays head_dim**-0.5 at every rank: the rank changes where
the projection lands, not the scale the original attention used.

The rotation R_p applies to the query and the key alike, so (R_p q).(R_p k) =
q.k and the latent is position-independent -- the cache holds no rope key.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImplBase,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.registry import register_backend as _register_backend
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from . import kernels
from .config import LOSSLESS_RANK

# Queries are attended in chunks, so the prefill never materialises the whole
# [tokens, heads, context] score matrix at once.
PREFILL_CHUNK = 256

_CACHE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_cache_dtype(cache_config, hf_config) -> torch.dtype:
    """The dtype the latent is stored in: the cache's, or the model's."""
    name = getattr(cache_config, "cache_dtype", "auto") or "auto"
    if name != "auto":
        return _CACHE_DTYPES[name]
    model_dtype = getattr(hf_config, "dtype", None) or getattr(
        hf_config, "torch_dtype", None
    )
    return model_dtype or torch.bfloat16


class LatentProjection(nn.Module):
    """A stack of per-KV-head projections, held as one parameter.

    A module rather than a bare parameter, so the checkpoint's
    `kv_uk_proj.weight` lands on it through vLLM's weight loader.
    """

    def __init__(self, *shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))


@dataclass
class MLAMetadata(CommonAttentionMetadata):
    """What the MLA layer needs from the runner.

    The split between prefill and decode is worked out once per step here, so
    the 28 layers do not each redo it.
    """

    num_tokens: int = 0
    """Tokens actually scheduled, without the padding the runner adds."""

    decode_req_idx: torch.Tensor | None = None
    """The requests with one token each, [num_decode]."""

    decode_token_pos: torch.Tensor | None = None
    """Where each decode request's single token sits in the batch."""

    prefill_reqs: list[tuple[int, int, int, int]] = field(default_factory=list)
    """(request, first token, one past the last token, context length)."""


class MLAMetadataBuilder(AttentionMetadataBuilder[MLAMetadata]):
    """Turns the runner's common metadata into the MLA layer's metadata."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MLAMetadata:
        starts = common_attn_metadata.query_start_loc_cpu.numpy()
        q_lens = np.diff(starts)
        # The context length of each request: the tokens already in the cache
        # plus the ones this step writes. It is on the CPU, so no sync.
        ctx_lens = common_attn_metadata.seq_lens_cpu_upper_bound.numpy()

        decode_req_idx: list[int] = []
        decode_token_pos: list[int] = []
        prefill_reqs: list[tuple[int, int, int, int]] = []
        for req, q_len in enumerate(q_lens):
            if q_len == 0:
                continue  # padding
            start = int(starts[req])
            if q_len == 1:
                decode_req_idx.append(req)
                decode_token_pos.append(start)
            else:
                prefill_reqs.append(
                    (req, start, start + int(q_len), int(ctx_lens[req]))
                )

        device = common_attn_metadata.query_start_loc.device
        carried = {
            f.name: getattr(common_attn_metadata, f.name)
            for f in fields(common_attn_metadata)
        }
        return MLAMetadata(
            **carried,
            num_tokens=int(starts[-1]),
            decode_req_idx=torch.tensor(decode_req_idx, dtype=torch.long, device=device),
            decode_token_pos=torch.tensor(
                decode_token_pos, dtype=torch.long, device=device
            ),
            prefill_reqs=prefill_reqs,
        )


class MLAAttentionLayer(nn.Module, AttentionLayerBase):
    """Qwen2 attention over a latent KV cache.

    The layer owns the cache write and both read paths: the latent is
    written at the slots the allocator handed out, and the attention reads it
    back -- a masked SDPA over the context for a prefill, the Triton kernel
    over the whole context for a decode.
    """

    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__()
        self.layer_name = prefix
        self.attn_type = AttentionType.DECODER
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.rank = int(getattr(config, "mla_rank", LOSSLESS_RANK))
        self.scaling = self.head_dim**-0.5

        self.cache_config = cache_config
        self.cache_dtype = _resolve_cache_dtype(cache_config, config)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        # The latent stack is float32: it is the projection the fidelity gate
        # measures, and bfloat16's 8-bit mantissa would spend a third of the
        # error budget on roundoff the truncation did not cause.
        self.kv_dkv_proj = nn.Linear(
            self.hidden_size, self.rank, bias=False, dtype=torch.float32
        )
        self.kv_uk_proj = LatentProjection(self.num_kv_heads, self.head_dim, self.rank)
        self.kv_uv_proj = LatentProjection(self.num_kv_heads, self.head_dim, self.rank)
        self.kv_v_bias = nn.Parameter(
            torch.zeros(self.num_kv_heads, self.head_dim, dtype=torch.float32)
        )

        # vLLM finds the attention layers through this registry, and binds the
        # allocated cache through it.
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return MLABackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MLAAttentionSpec:
        """One latent per token per layer, and no separate value cache."""
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.rank,
            head_size_v=0,
            dtype=self.cache_dtype,
        )

    def forward(self, positions, hidden_states: torch.Tensor) -> torch.Tensor:
        raw = get_forward_context().attn_metadata
        if raw is None:
            # The memory profile run: no cache is bound yet and no metadata
            # is built. There is nothing to attend to.
            return torch.zeros(
                hidden_states.shape[0],
                self.hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        if isinstance(raw, list):
            raw = raw[0]
        meta = raw[self.layer_name]

        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        q_hat = self._absorb(q)

        latent = self.kv_dkv_proj(hidden_states.float())
        self._write_cache(latent, meta)

        ctx = self._attend(q_hat, meta)
        out = self._expand(ctx).reshape(-1, self.num_heads * self.head_dim)
        return self.o_proj(out.to(self.o_proj.weight.dtype))

    # --- the algebra --------------------------------------------------------

    def _absorb(self, q: torch.Tensor) -> torch.Tensor:
        """q_hat[h] = W_UK[g]^T q_h, with g = h // 6."""
        tokens = q.shape[0]
        grouped = q.view(tokens, self.num_kv_heads, self.num_groups, self.head_dim)
        absorbed = torch.einsum(
            "tgad,gdr->tgar", grouped.float(), self.kv_uk_proj.weight
        )
        return absorbed.reshape(tokens, self.num_heads, self.rank).to(self.cache_dtype)

    def _expand(self, ctx: torch.Tensor) -> torch.Tensor:
        """out[h] = ctx[h] W_UV[g] + b_V[g]."""
        tokens = ctx.shape[0]
        grouped = ctx.view(tokens, self.num_kv_heads, self.num_groups, self.rank)
        out = torch.einsum("tgar,gdr->tgad", grouped, self.kv_uv_proj.weight)
        out = out + self.kv_v_bias.view(self.num_kv_heads, 1, self.head_dim)
        return out.reshape(tokens, self.num_heads, self.head_dim)

    # --- the cache ----------------------------------------------------------

    def _cache(self) -> torch.Tensor:
        """The per-layer latent cache, [num_blocks, block_size, r]."""
        return self.kv_cache.squeeze(1)

    def _write_cache(self, latent: torch.Tensor, meta: MLAMetadata) -> None:
        """Scatter the latent into the paged cache at the allocated slots."""
        slots = meta.slot_mapping[: meta.num_tokens].long()
        if slots.numel() == 0:
            return
        block_size = self.cache_config.block_size
        block = slots // block_size
        offset = slots - block * block_size
        self._cache()[block, offset] = latent[: meta.num_tokens].to(self.cache_dtype)

    def _gather(self, cache: torch.Tensor, block_table_row, ctx_len: int):
        """One request's context, gathered out of the paged cache."""
        block_size = self.cache_config.block_size
        positions = torch.arange(ctx_len, device=cache.device)
        block = block_table_row[positions // block_size].long()
        offset = positions % block_size
        return cache[block, offset]

    # --- the attention ------------------------------------------------------

    def _attend(self, q_hat: torch.Tensor, meta: MLAMetadata) -> torch.Tensor:
        tokens = q_hat.shape[0]
        ctx = torch.zeros(
            tokens, self.num_heads, self.rank, dtype=torch.float32, device=q_hat.device
        )
        cache = self._cache()

        if meta.decode_token_pos.numel() > 0:
            ctx.index_copy_(0, meta.decode_token_pos, self._decode(q_hat, cache, meta))

        for req, start, end, ctx_len in meta.prefill_reqs:
            ctx[start:end] = self._prefill(
                q_hat[start:end], cache, meta, req, start, ctx_len
            )

        return ctx

    def _decode(self, q_hat: torch.Tensor, cache: torch.Tensor, meta: MLAMetadata):
        """One token per request, over its whole context."""
        q = q_hat[meta.decode_token_pos]
        block_table = meta.block_table_tensor[meta.decode_req_idx]
        seq_lens = meta.seq_lens[meta.decode_req_idx]

        if kernels.triton_available():
            return kernels.mla_decode_triton(
                q, cache, block_table, seq_lens, self.scaling, self.head_dim
            )

        out = []
        for i in range(q.shape[0]):
            latent = self._gather(cache, block_table[i], int(seq_lens[i]))
            out.append(
                kernels.mla_decode_reference(
                    q[i : i + 1], latent.unsqueeze(0), seq_lens[i : i + 1], self.scaling
                )
            )
        return torch.cat(out)

    def _prefill(
        self,
        q: torch.Tensor,
        cache: torch.Tensor,
        meta: MLAMetadata,
        req: int,
        start: int,
        ctx_len: int,
    ) -> torch.Tensor:
        """Causal attention over one request's whole context."""
        latent = self._gather(cache, meta.block_table_tensor[req], ctx_len)
        positions = meta.positions[start : start + q.shape[0]]
        mask = (
            torch.arange(ctx_len, device=q.device) <= positions.unsqueeze(1)
        )

        out = []
        for lo in range(0, q.shape[0], PREFILL_CHUNK):
            hi = min(lo + PREFILL_CHUNK, q.shape[0])
            attn = F.scaled_dot_product_attention(
                q[lo:hi].transpose(0, 1).unsqueeze(0),
                latent.unsqueeze(0).unsqueeze(0),
                latent.unsqueeze(0).unsqueeze(0),
                attn_mask=mask[lo:hi],
                scale=self.scaling,
            )
            out.append(attn[0].transpose(0, 1))
        return torch.cat(out)


class MLABackend(AttentionBackend):
    """The MLA layer does its own attention, so the backend only supplies the
    metadata builder."""

    @staticmethod
    def get_name() -> str:
        return "MLA"

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase]:
        return AttentionImplBase

    @staticmethod
    def get_builder_cls() -> type[MLAMetadataBuilder]:
        return MLAMetadataBuilder


class Qwen2MLADecoderLayer(Qwen2DecoderLayer):
    """A Qwen2 decoder layer whose attention holds the latent.

    The stock layer is not built: its QKV projection and its rope table would
    be dead weight, and its attention layer would claim the layer name in
    vLLM's registry.
    """

    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = MLAAttentionLayer(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen2MLAModel(Qwen2Model):
    """Qwen2 with the MLA decoder layer.

    The stock mapper stacks q/k/v into one qkv_proj; this model keeps the
    query on its own and replaces the keys and values with the latent, so
    only the MLP's gate/up stacking is left.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MLADecoderLayer,
    ):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )


class Qwen2MLAForCausalLM(Qwen2ForCausalLM):
    """Qwen2 with a latent KV cache."""

    hf_to_vllm_mapper = Qwen2MLAModel.hf_to_vllm_mapper
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2MLAModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


class _IdentityRotary(nn.Module):
    """A rotary that does not rotate.

    The MLA arm's score is q.k rather than q^T R_{s-p} k (PLAN.md section
    3.1 applies the rotation to the query and the key alike, so it cancels).
    Swapping the rotation out of the stock Qwen2 gives the arm the MLA arm
    is algebraically equal to at r=512, which is what the two can be checked
    against.
    """

    def forward(self, positions, query, key=None):
        return (query, key) if key is not None else query


class Qwen2NoRopeDecoderLayer(Qwen2DecoderLayer):
    """A Qwen2 decoder layer whose attention does not rotate."""

    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__(config, cache_config, quant_config, prefix)
        self.self_attn.rotary_emb = _IdentityRotary()


class Qwen2NoRopeModel(Qwen2Model):
    """Qwen2 without the rotation."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2NoRopeDecoderLayer,
    ):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )


class Qwen2NoRopeForCausalLM(Qwen2ForCausalLM):
    """Qwen2 without the rotation: the reference the MLA arm is equal to."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2NoRopeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


def register_plugin() -> None:
    """Put the model where vLLM's registry looks for architectures."""
    ModelRegistry.register_model("Qwen2MLAForCausalLM", Qwen2MLAForCausalLM)
    ModelRegistry.register_model("Qwen2NoRopeForCausalLM", Qwen2NoRopeForCausalLM)


def register_backend() -> None:
    """Make MLABackend the CUSTOM backend, so it resolves by path."""
    _register_backend(AttentionBackendEnum.CUSTOM, "kvbench.models.MLABackend")


def register_plugins() -> None:
    """Everything the engine needs, in one call.

    This is the vllm.general_plugins entry point: vLLM runs it in every
    process it spawns, so the engine core finds the model too.
    """
    register_plugin()
    register_backend()
