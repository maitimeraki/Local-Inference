"""Every constant the benchmark is pinned to.

One file, so the two arms cannot drift apart on anything that would make the
comparison unfair. Nothing here reads the environment at import time beyond
path resolution -- the values below are the contract.
"""

import os
from pathlib import Path

# --- model -----------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Geometry of the pinned checkpoint. These are not free parameters: the
# softmax scale is derived from HEAD_DIM, and the joint SVD's lossless rank
# is derived from the stacked [W_K ; W_V] matrix.
HIDDEN_SIZE = 1536
NUM_LAYERS = 28
NUM_ATTENTION_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
NUM_KV_GROUPS = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 6 query heads per KV head

# The score is a projection of a head_dim-128 dot product, so the softmax
# scale stays 128**-0.5 at every rank. It is NOT rank**-0.5: the latent
# rank changes the dimension the projection lands in, not the scale the
# original attention used.
ATTENTION_SCALE = HEAD_DIM ** -0.5

# --- corpus ----------------------------------------------------------------

CORPUS_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
CORPUS_SHA256 = "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"

# --- the sweep -------------------------------------------------------------

# 512 is the rank of [W_K ; W_V], so it is the only lossless latent: below it
# the sweep is an explicit truncation of the model's key/value subspace.
RANKS = [512, 384, 256, 192, 128, 96, 64]
LOSSLESS_RANK = 512

# 1.9 GiB of KV cache, identical for every arm, enforced through
# num_gpu_blocks_override so no arm can quietly win by caching more.
KV_BUDGET_BYTES = 2_040_109_465

# GQA's cost per token per layer: 2 KV heads x 128 dims x 2 (K and V) x 2 B.
KV_BYTES_PER_TOKEN_GQA = NUM_KV_HEADS * HEAD_DIM * 2 * 2 * NUM_LAYERS  # 28672

# --- engine ----------------------------------------------------------------

# Pinned for every arm. Throughput runs use exactly this; fidelity runs add
# max_num_seqs=1 on top (batch composition and paged gathers change reduction
# order, which would make a token-level diff meaningless).
ENGINE_FLAGS = {
    "enforce_eager": True,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.90,
    "seed": 0,
}
# The plan pins swap_space=0, but vLLM 0.31's engine has no such argument:
# the v1 engine does not swap, so there is nothing to turn off.

# --- paths -----------------------------------------------------------------

# Overridable so the same code runs on the host and inside the image, where
# the model is baked at /models and results are bind-mounted from the host.
MODEL_DIR = Path(
    os.environ.get("KVBENCH_MODEL_DIR", "/models/qwen2.5-1.5b-instruct")
)
CORPUS_PATH = Path(os.environ.get("KVBENCH_CORPUS_PATH", "/data/pg1342.txt"))
RESULTS_DIR = Path(os.environ.get("KVBENCH_RESULTS_DIR", "/results"))

# Where the MLA checkpoints are written: the base model with its K/V
# projections replaced by the latent stack.
MLA_WEIGHTS_DIR = Path(os.environ.get("KVBENCH_MLA_DIR", "/tmp/kvbench-mla"))
