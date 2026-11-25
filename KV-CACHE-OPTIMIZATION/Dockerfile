# The image bakes the model, the corpus and vLLM, so a run needs no network
# and no bind mount -- WSL2 paths do not bind-mount into the container.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PYTHONUNBUFFERED=1 \
    KVBENCH_MODEL_DIR=/opt/models/Qwen2.5-1.5B-Instruct \
    KVBENCH_CORPUS_PATH=/opt/data/pg1342.txt \
    KVBENCH_RESULTS_DIR=/results \
    KVBENCH_MLA_DIR=/opt/mla

# libgomp1 for torch's OpenMP runtime; gcc for Triton's JIT, which compiles
# the kernel at first use and needs a C compiler to do it; curl and
# ca-certificates for the corpus fetch at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 gcc g++ curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kvbench

# vllm pins torch, and the pyproject pins the same version, so the two
# agree without a second torch install.
COPY pyproject.toml ./
COPY kvbench ./kvbench
COPY tests ./tests
RUN pip install vllm==0.31.0 && pip install -e .

# The model and the corpus are baked in. The corpus is checked against its
# checksum at build time: a file that does not match is not the corpus, and
# the benchmark reads it as text and nothing else.
#
# The download is a script rather than a heredoc, because the classic
# builder runs the shell without the heredoc's body.
COPY docker/bake_assets.py /tmp/bake_assets.py
RUN mkdir -p /opt/models /opt/data /opt/mla /results \
    && python /tmp/bake_assets.py \
    && rm /tmp/bake_assets.py

# From here on there is no network: the model and the corpus are on disk.
ENV HF_HUB_OFFLINE=1

# pytest, so the suite runs in the image. It is a separate layer from the
# model bake, so a rebuild does not refetch the model.
RUN pip install pytest==9.1.1

CMD ["python", "-m", "kvbench.bench", "--help"]
