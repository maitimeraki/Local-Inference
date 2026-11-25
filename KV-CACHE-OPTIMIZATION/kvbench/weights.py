"""Offline transformation of Qwen2 GQA weights into an MLA latent stack.

The construction (PLAN.md section 3.2). Stack the two KV projections into one
matrix and take its SVD:

    A = [W_K ; W_V]  in R^{512 x 1536},  A = U S V^T

Row layout of A: rows 0:128 are KV head 0 of W_K, rows 128:256 are KV head 1
of W_K, rows 256:384 are KV head 0 of W_V, rows 384:512 are KV head 1 of W_V.

The rank-r latent stack is then

    W_DKV   = V[:, :r]^T            [r, 1536]   applied at cache-write time
    W_UK[g] = (U S)[128g : 128g+128, :r]        [128, r]   folded into the query
    W_UV[g] = (U S)[256+128g : 256+128g+128, :r] [128, r]  folded into the output

so that W_UK[g] @ W_DKV reconstructs W_K[g] and W_UV[g] @ W_DKV reconstructs
W_V[g]. A has rank 512, so r = 512 is exact and any lower r is a truncation
whose discarded energy is reported as rel_err_k / rel_err_v rather than assumed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from . import config

# Tensor names written into the MLA checkpoint, under
# "model.layers.{i}.self_attn.".
DKV_KEY = "kv_dkv_proj.weight"
UK_KEY = "kv_uk_proj.weight"
UV_KEY = "kv_uv_proj.weight"
V_BIAS_KEY = "kv_v_bias"


@dataclass
class MLASubspace:
    """The rank-r latent stack for one layer, plus the truncation error."""

    w_dkv: torch.Tensor  # [r, HIDDEN]   write path
    w_uk: torch.Tensor  # [NUM_KV_HEADS, HEAD_DIM, r]  folded into the query
    w_uv: torch.Tensor  # [NUM_KV_HEADS, HEAD_DIM, r]  folded into the output
    rel_err_k: torch.Tensor  # [NUM_KV_HEADS]  ||W_UK W_DKV - W_K|| / ||W_K||
    rel_err_v: torch.Tensor  # [NUM_KV_HEADS]


def load_model_state(model_dir, dtype=None) -> dict[str, torch.Tensor]:
    """Load a safetensors checkpoint onto the CPU.

    `dtype` casts every tensor; the default keeps the checkpoint's own dtype
    (bf16 for the pinned model). Widening bf16 to fp32 is exact, and the SVD
    below casts to float64 anyway, so the default avoids a 6.2 GB copy of a
    3.1 GB model for no precision gain.
    """
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted({model_dir / name for name in weight_map.values()})
    else:
        shards = [model_dir / "model.safetensors"]

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(f"missing safetensors shard: {shard}")
        for name, tensor in load_file(str(shard)).items():
            state[name] = tensor.to(dtype) if dtype is not None else tensor
    return state


def joint_svd(state: dict[str, torch.Tensor], rank: int) -> MLASubspace:
    """Joint SVD of [W_K ; W_V] into a rank-r MLA latent stack."""
    head = config.HEAD_DIM
    if not 0 < rank <= 2 * head * config.NUM_KV_HEADS:
        raise ValueError(f"rank {rank} outside 1..{2 * head * config.NUM_KV_HEADS}")

    w_k = state["k_proj.weight"].double()
    w_v = state["v_proj.weight"].double()
    a = torch.cat([w_k, w_v], dim=0)  # [2*head*NUM_KV_HEADS, HIDDEN]

    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    u_s = u[:, :rank] * s[:rank]  # [512, r]
    w_dkv = vh[:rank, :].contiguous()  # [r, HIDDEN] -- this is V[:, :r]^T already

    w_uk = torch.stack([u_s[head * g : head * g + head] for g in range(config.NUM_KV_HEADS)])
    w_uv = torch.stack(
        [u_s[2 * head + head * g : 2 * head + head * g + head] for g in range(config.NUM_KV_HEADS)]
    )

    rel_err_k = torch.stack(
        [
            (w_uk[g] @ w_dkv - w_k[head * g : head * g + head]).norm()
            / w_k[head * g : head * g + head].norm()
            for g in range(config.NUM_KV_HEADS)
        ]
    )
    rel_err_v = torch.stack(
        [
            (w_uv[g] @ w_dkv - w_v[head * g : head * g + head]).norm()
            / w_v[head * g : head * g + head].norm()
            for g in range(config.NUM_KV_HEADS)
        ]
    )
    return MLASubspace(w_dkv, w_uk, w_uv, rel_err_k, rel_err_v)


def _layer_indices(state: dict[str, torch.Tensor]) -> list[int]:
    indices = set()
    for name in state:
        if name.endswith("k_proj.weight") and ".layers." in name:
            indices.add(int(name.split(".layers.")[1].split(".")[0]))
    return sorted(indices)


def build_mla_weights(model_dir, rank: int, out_dir) -> Path:
    """Write mla_rank{r}.safetensors: the checkpoint with k_proj/v_proj
    replaced by the MLA latent stack.

    k_proj and v_proj are removed; b_K is dropped (it cancels in the
    shift-invariant softmax) and b_V is kept, because it must be added after
    the up-projection rather than folded into it.
    """
    state = load_model_state(model_dir)
    out: dict[str, torch.Tensor] = {}

    for index in _layer_indices(state):
        prefix = f"model.layers.{index}.self_attn."
        layer = {
            "k_proj.weight": state.pop(prefix + "k_proj.weight"),
            "v_proj.weight": state.pop(prefix + "v_proj.weight"),
        }
        state.pop(prefix + "k_proj.bias")  # b_K cancels in the shift-invariant softmax
        v_bias = state.pop(prefix + "v_proj.bias")

        sub = joint_svd(layer, rank)
        # float32 for the new stack: the latent projections are what the
        # fidelity gate measures, and bf16 would spend a third of the error
        # budget on roundoff the truncation did not cause.
        out[prefix + DKV_KEY] = sub.w_dkv.float()
        out[prefix + UK_KEY] = sub.w_uk.float()
        out[prefix + UV_KEY] = sub.w_uv.float()
        out[prefix + V_BIAS_KEY] = v_bias.float().view(config.NUM_KV_HEADS, config.HEAD_DIM)

    out.update(state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mla_rank{rank}.safetensors"
    save_file(out, str(path), metadata={"format": "pt"})
    return path


def load_mla_weights(path) -> dict[str, torch.Tensor]:
    """Load a checkpoint written by build_mla_weights."""
    return load_file(str(Path(path)))
