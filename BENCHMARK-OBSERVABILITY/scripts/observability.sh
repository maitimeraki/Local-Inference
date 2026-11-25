#!/usr/bin/env bash
# Start the observability stack pointed at the grid that is actually running.
#
# The grid's ports are not passed in. `dev.sh` writes them to
# observability/prometheus/targets/, and Prometheus watches those files, so the
# stack follows the grid instead of being told about it once. Run this
# *after* dev.sh, or the stack will scrape the ports the grid used to be on.
#
#   ./scripts/dev.sh          # starts the grid and writes the targets
#   ./scripts/observability.sh
#
# Prometheus http://localhost:9090    Grafana http://localhost:3000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE="observability/compose.yaml"
TARGETS_DIR="$ROOT/observability/prometheus/targets"

# Docker is a Windows daemon here, and the WSL2 image has no docker client.
# So use whichever one exists: the Windows binary from WSL, or a native one.
DOCKER=""
if command -v docker >/dev/null 2>&1; then
  DOCKER="docker"
elif [ -x "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
  DOCKER="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
elif [ -x "/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
  DOCKER="/c/Program Files/Docker/Docker/resources/bin/docker.exe"
fi
if [ -z "$DOCKER" ]; then
  cat >&2 <<'EOF'
docker not found.

The observability stack runs in Docker on the Windows host, while the grid
runs in WSL2. Install Docker Desktop and make sure it is running; from WSL2
this script will use its docker.exe directly.
EOF
  exit 1
fi

if [ ! -f "$TARGETS_DIR/gateway.json" ]; then
  echo "!! $TARGETS_DIR/gateway.json is missing -- start the grid first:" >&2
  echo "   ./scripts/dev.sh" >&2
  exit 1
fi
echo "== scraping $(cat "$TARGETS_DIR/gateway.json") $(cat "$TARGETS_DIR/scheduler.json")"

"$DOCKER" compose -f "$COMPOSE" up -d
# Prometheus picks up a changed target file on its own, but it reads its
# config only at startup, so it still has to be restarted for a changed
# scrape config.
"$DOCKER" compose -f "$COMPOSE" restart prometheus
"$DOCKER" compose -f "$COMPOSE" ps
