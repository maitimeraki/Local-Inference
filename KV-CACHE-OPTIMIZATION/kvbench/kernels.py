"""The MLA latent decode kernel: a PyTorch reference and a Triton paged kernel.

Both compute, for every (sequence, query head):

    score[p] = (q_hat[h] . c_p) * scale
    ctx[h]   = sum_p softmax(score)[p] * c_p

where `c_p` is the rank-r latent of token p and `scale` is head_dim**-0.5 --
the model's own softmax scale, not the latent rank's (PLAN.md section 3.3).
The reference is the specification; the Triton kernel is the fast path, and
the fallback when it fails to compile.
"""

import torch

from . import config

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton ships with torch on Linux, not on Windows.
    triton = None
    tl = None

# 12 query heads padded to 16; 64 tokens of latent per step.
BLOCK_H = 16
BLOCK_N = 64
MIN_DOT_N = 16  # tl.dot needs at least 16 in each dimension


def mla_decode_reference(q_hat, latent, seq_lens, scale):
    """Softmax attention over the latent, in pure PyTorch.

    q_hat    [num_seqs, num_heads, r]
    latent   [num_seqs, seq_len, r]
    seq_lens [num_seqs] -- the valid prefix length of each sequence
    returns  [num_seqs, num_heads, r]

    The arithmetic is float32 whatever the latent's dtype: this is the
    specification, so it is the accurate version of the computation. The
    Triton kernel rounds the probabilities to the latent's dtype for its
    tensor cores, which is why the two agree to about 1e-6 in float32 and
    only to about 1e-3 in bfloat16.
    """
    c = latent.float()
    scores = torch.einsum("shd,snd->shn", q_hat.float(), c) * scale
    valid = torch.arange(c.shape[1], device=c.device).unsqueeze(0) < seq_lens.to(c.device).unsqueeze(1)
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("shn,snd->shd", probs, c)


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _dot(a, b, FP32: tl.constexpr):
        # float32 dot defaults to tf32, which is not accurate enough to call
        # the result "the same numbers" as the reference.
        if FP32:
            out = tl.dot(a, b, input_precision="ieee")
        else:
            out = tl.dot(a, b)
        return out

    @triton.jit
    def _mla_decode_kernel(
        Q,  # [num_seqs, num_heads, r]
        CACHE,  # [num_blocks, block_size, r]
        BLOCK_TABLE,  # [num_seqs, max_blocks]
        SEQ_LENS,  # [num_seqs]
        OUT,  # [num_seqs, num_heads, r]
        scale,
        rank,
        block_size,
        num_heads,
        stride_q_seq, stride_q_head, stride_q_dim,
        stride_c_block, stride_c_token, stride_c_dim,
        stride_bt_seq,
        stride_o_seq, stride_o_head, stride_o_dim,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        FP32: tl.constexpr,
    ):
        seq = tl.program_id(0)
        head_block = tl.program_id(1)
        seq_len = tl.load(SEQ_LENS + seq)

        offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        h_mask = offs_h < num_heads
        d_mask = offs_d < rank

        q = tl.load(
            Q + seq * stride_q_seq + offs_h[:, None] * stride_q_head + offs_d[None, :] * stride_q_dim,
            mask=h_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        # online softmax over the sequence's blocks
        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_D], tl.float32)

        num_blocks = (seq_len + block_size - 1) // block_size
        sub_tiles = (block_size + BLOCK_N - 1) // BLOCK_N

        for blk in range(0, num_blocks):
            phys = tl.load(BLOCK_TABLE + seq * stride_bt_seq + blk)
            for sub in range(0, sub_tiles):
                tok = sub * BLOCK_N + offs_n  # token index inside the block
                gpos = blk * block_size + tok  # position in the sequence
                n_mask = (tok < block_size) & (gpos < seq_len)
                c = tl.load(
                    CACHE
                    + phys * stride_c_block
                    + tok[:, None] * stride_c_token
                    + offs_d[None, :] * stride_c_dim,
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )

                s = _dot(q, tl.trans(c), FP32) * scale
                s = tl.where(n_mask[None, :], s, float("-inf"))

                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + _dot(p.to(c.dtype), c, FP32)
                m_i = m_new

        tl.store(
            OUT + seq * stride_o_seq + offs_h[:, None] * stride_o_head + offs_d[None, :] * stride_o_dim,
            acc / l_i[:, None],
            mask=h_mask[:, None] & d_mask[None, :],
        )


def mla_decode_triton(q_hat, latent_cache, block_table, seq_lens, scale=None, head_dim=config.HEAD_DIM):
    """The paged decode: the same computation as mla_decode_reference, over a
    block-table-addressed latent cache.

    q_hat       [num_seqs, num_heads, r]
    latent_cache [num_blocks, block_size, r]
    block_table [num_seqs, max_blocks] -- logical block -> physical block
    seq_lens    [num_seqs]
    returns     [num_seqs, num_heads, r] in float32

    `scale` defaults to head_dim**-0.5, the model's own softmax scale: the
    latent rank changes where the projection lands, not the scale the
    original attention used.
    """
    num_seqs, num_heads, rank = q_hat.shape
    block_size = latent_cache.shape[1]
    out = torch.empty((num_seqs, num_heads, rank), dtype=torch.float32, device=q_hat.device)

    # A tile may not cross a cache-block boundary, so its width is capped by
    # the block size; tl.dot needs at least 16.
    tile = max(MIN_DOT_N, min(BLOCK_N, triton.next_power_of_2(block_size)))

    grid = (num_seqs, triton.cdiv(num_heads, BLOCK_H))
    _mla_decode_kernel[grid](
        q_hat,
        latent_cache,
        block_table,
        seq_lens,
        out,
        float(scale if scale is not None else head_dim ** -0.5),
        rank,
        block_size,
        num_heads,
        *q_hat.stride(),
        *latent_cache.stride(),
        block_table.stride(0),
        *out.stride(),
        BLOCK_H=BLOCK_H,
        BLOCK_D=triton.next_power_of_2(rank),
        BLOCK_N=tile,
        FP32=(q_hat.dtype == torch.float32),
    )
    return out
