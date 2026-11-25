#!/usr/bin/env bash
# Bootstrap the WSL2 side of this project.
#
# Run this *inside* WSL2, not in PowerShell. vLLM is installed here as a
# library, deliberately not run as a Docker image: the whole reason this
# project exists is to watch the scheduler, the KV cache and the prefix cache
# make their decisions, and a container hides every one of them behind an HTTP
# boundary.
#
#   ./scripts/bootstrap_wsl.sh                # install vLLM from PyPI wheels
#   ./scripts/bootstrap_wsl.sh --from-source  # build vLLM from a local clone
#
# The --from-source path is the "raw code" option: you get a checkout you can
# read, edit and re-install, which is the only way to change scheduler
# behaviour rather than just observe it.
set -euo pipefail

FROM_SOURCE=0
[[ "${1:-}" == "--from-source" ]] && FROM_SOURCE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The venv is WSL-native and shared across the project, not inside the repo
# (which may live on /mnt/c). uv creates/syncs this path when it is exported.
VENV=/home/anupam/.venv
export UV_PROJECT_ENVIRONMENT="$VENV"

say() { printf '\n== %s\n' "$*"; }

say "host"
if ! grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  echo "!! /proc/version does not look like WSL. This script targets WSL2." >&2
  echo "   Continuing, but the GPU steps below assume a Windows host driver." >&2
fi

if ! command -v nvidia-smi >/dev/null; then
  cat >&2 <<'EOF'
!! nvidia-smi not found.

   Install the *Windows* NVIDIA driver only. WSL2 passes the GPU through from
   the host, so a second Linux driver installed inside WSL will fight the
   Windows one and you will end up with no GPU at all.

   Check from PowerShell:
     wsl --update
     nvidia-smi          # inside WSL
EOF
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

say "uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "python 3.11"
uv python install 3.11

if [[ $FROM_SOURCE -eq 1 ]]; then
  say "vLLM from source"
  # Building vLLM needs a real CUDA toolkit; the wheel path does not.
  if ! command -v nvcc >/dev/null; then
    cat >&2 <<'EOF'
!! --from-source needs nvcc. Install a CUDA toolkit first, e.g.
     sudo apt install nvidia-cuda-toolkit      # or the 12.x runfile from NVIDIA
EOF
    exit 1
  fi

  SRC="${VLLM_SRC:-$HOME/src/vllm}"
  if [[ ! -d "$SRC" ]]; then
    git clone --depth 1 https://github.com/vllm-project/vllm "$SRC"
  fi

  cd "$SRC"
  # Editable install: an edit to the scheduler takes effect on the next run,
  # which is the point. Expect this build to take 10-40 minutes.
  uv pip install --python "$VENV" -r requirements/common.txt
  uv pip install --python "$VENV" -e .
else
  say "workspace dependencies (incl. the vllm extra)"
  cd "$ROOT"
  # --all-packages installs the four workspace members (inferbench, infergw,
  # inferobs, infersched) and their console scripts. Without it, --all-extras
  # only reaches the virtual root, which has no dependencies of its own, so
  # nothing installs and `uv run inferbench` dies with "Failed to spawn".
  # --all-extras pulls vllm for inferbench and the OTel SDK for inferobs.
  # On Linux the PyPI torch wheels are already CUDA builds; no extra index
  # is needed for the RTX 3050 (sm_86).
  # --inexact keeps packages other projects installed in the shared venv.
  uv sync --all-packages --all-extras --inexact
fi

say "verify"
cd "$ROOT"
uv run --no-sync python - <<'PY'
import importlib.util

import torch

print(f"torch      {torch.__version__}")
print(f"cuda avail {torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"device     {torch.cuda.get_device_name(0)}")
    print(f"vram       {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")
    print(f"capability sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

spec = importlib.util.find_spec("vllm")
print(f"vllm       {spec.origin if spec else 'NOT INSTALLED'}")
PY

say "next"
cat <<'EOF'
  uv run --no-sync inferbench inspect --config baseline --workload smoke
      step through the engine, one iteration per line

  ./scripts/dev.sh
      scheduler on :9001, gateway on :9000, ready to benchmark

  uv run --no-sync inferbench bench --experiment smoke --base-url http://localhost:9000
EOF
