#!/usr/bin/env bash
# Local development: the production grid, in this one terminal.
#
#   :9001  infersched   tier 2 — owns the GPU, the KV-block ledger and the hold queue
#   :9000  infergw      tier 1 — the router, the cache and the idempotency registry
#
# The engine is not a service of its own any more: the scheduler loads it in
# process, so the block pool is a readable Python object and the ledger cannot
# lie about it. That is the whole reason for the tier split.
#
# MODE=engine is the A/B baseline path: the engine on :8000 behind the gateway
# with no scheduler in front of it — the comparison the queueing numbers are
# drawn against. It costs a second config path, not a second machine.
#
# Benchmark through the gateway (:9000) rather than the scheduler (:9001): the
# gateway is the only place that knows the arrival rate, the admission
# decisions and the cache hit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Same WSL-native venv the bootstrap script installs into.
export UV_PROJECT_ENVIRONMENT=/home/anupam/.venv

MODE="${MODE:-grid}"
ENGINE_CONFIG="${ENGINE_CONFIG:-baseline}"
ENGINE_PORT="${ENGINE_PORT:-8000}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"
SCHEDULER_PORT="${SCHEDULER_PORT:-9001}"

# Tell Prometheus where the grid ended up, before starting it.
#
# Prometheus reads its scrape targets from files, and it watches them. This
# script is the only thing that knows which ports the grid got -- they move
# when something else on the machine already owns the defaults -- so it is the
# only thing that can write them. Writing the ports into Prometheus's config
# instead means a grid that moves leaves an empty dashboard with no error
# anywhere, which is exactly what happened.
TARGETS_DIR="$ROOT/observability/prometheus/targets"
mkdir -p "$TARGETS_DIR"
printf '[{"targets": ["host.docker.internal:%s"]}]\n' "$GATEWAY_PORT" \
  > "$TARGETS_DIR/gateway.json"
printf '[{"targets": ["host.docker.internal:%s"]}]\n' "$SCHEDULER_PORT" \
  > "$TARGETS_DIR/scheduler.json"
echo "== prometheus targets: gateway :$GATEWAY_PORT   scheduler :$SCHEDULER_PORT"

# How long to wait for a service to become ready. The engine's wait has to be
# generous: on a cold page cache the weights are read from disk (EXT4, no
# auto-prefetch), and a cold start measured 3.5 minutes. A short wait here
# reports "the scheduler never became ready" while the engine is seconds from
# loading — the error names the wrong service.
WAIT_S="${WAIT_S:-600}"

# Where the gateway sends its traffic. In grid mode the scheduler is the only
# upstream; in engine mode it is the engine. Declared here, once, so the port
# and the upstream cannot drift apart.
if [ "$MODE" = "engine" ]; then
  export INFERGW_SCHEDULER_URL=""
  export INFERGW_BACKEND_URL="http://localhost:$ENGINE_PORT"
else
  export INFERSCHED_PORT="$SCHEDULER_PORT"
  export INFERGW_SCHEDULER_URL="http://localhost:$SCHEDULER_PORT"
fi

# Refuse to start on a port that is already taken. A leftover engine from an
# earlier session answers /health, so the readiness wait below would be
# satisfied by the *wrong* process, the real engine would die on bind, and the
# gateway would then be pointed at nothing. Better to say so now.
port_busy() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && return 0
  return 1
}

if [ "$MODE" = "engine" ]; then
  ports=("$ENGINE_PORT" "$GATEWAY_PORT")
else
  ports=("$SCHEDULER_PORT" "$GATEWAY_PORT")
fi

for port in "${ports[@]}"; do
  if port_busy "$port"; then
    echo "!! port $port is already in use." >&2
    echo "   Find the holder with:  ss -ltnp | grep :$port" >&2
    echo "   Stop it, or re-run on free ports:" >&2
    echo "     GATEWAY_PORT=9100 SCHEDULER_PORT=9101 $0" >&2
    exit 1
  fi
done

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    # `uv run` is a wrapper: the service is its child, and the child is the
    # one holding the GPU. Signal it first so it can shut the engine down
    # cleanly — killing the wrapper alone leaves an orphan on the card.
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

if [ "$MODE" = "engine" ]; then
  echo "== starting engine (config: $ENGINE_CONFIG) on :$ENGINE_PORT"
  uv run --no-sync inferbench serve --config "$ENGINE_CONFIG" --port "$ENGINE_PORT" &
  pids+=($!)

  # Wait for readiness rather than sleeping a fixed amount: loading the
  # weights is the slow part and it varies with disk cache state.
  echo -n "   waiting for the engine "
  for _ in $(seq 1 "$WAIT_S"); do
    if curl -sf "http://localhost:$ENGINE_PORT/health" >/dev/null 2>&1; then
      echo " ready"
      break
    fi
    echo -n "."
    sleep 1
  done
  if ! curl -sf "http://localhost:$ENGINE_PORT/health" >/dev/null 2>&1; then
    echo
    echo "!! the engine never became healthy. Its log above is the place to look." >&2
    exit 1
  fi
else
  echo "== starting scheduler (engine: config/scheduler.yaml) on :$SCHEDULER_PORT"
  uv run --no-sync infersched &
  pids+=($!)

  # The scheduler answers /health long before the engine is attached — the
  # weights still have to load. `ready` is the flag that says the ledger has
  # something to account for.
  echo -n "   waiting for the scheduler and its engine "
  for _ in $(seq 1 "$WAIT_S"); do
    if curl -sf "http://localhost:$SCHEDULER_PORT/health" 2>/dev/null | grep -qE '"ready":\s*true'; then
      echo " ready"
      break
    fi
    echo -n "."
    sleep 1
  done
  if ! curl -sf "http://localhost:$SCHEDULER_PORT/health" 2>/dev/null | grep -qE '"ready":\s*true'; then
    echo
    echo "!! the scheduler never became ready. Its log above is the place to look." >&2
    exit 1
  fi
fi

echo "== starting gateway on :$GATEWAY_PORT"
uv run --no-sync infergw --port "$GATEWAY_PORT" &
pids+=($!)

# The gateway is the thing that silently swallows a misconfiguration: it starts
# happily and answers /health while every request it forwards fails. So check
# it too, and check that it can actually reach the engine.
#
# Its wait gets the same generous budget as the engine's. The gateway imports
# its own package over the WSL 9p mount, and while the host is busy streaming
# the checkpoint those imports stall in uninterruptible sleep — measured at
# over 30 s, so a short wait here fails a gateway that is seconds from
# binding.
echo -n "   waiting for the gateway "
for _ in $(seq 1 "$WAIT_S"); do
  if curl -sf "http://localhost:$GATEWAY_PORT/health" >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  echo -n "."
  sleep 1
done
if ! curl -sf "http://localhost:$GATEWAY_PORT/health" >/dev/null 2>&1; then
  echo
  echo "!! the gateway never became healthy. Its log above is the place to look." >&2
  exit 1
fi

if [ "$MODE" = "engine" ]; then
  cat <<EOF

  engine   http://localhost:$ENGINE_PORT/health   /metrics   /v1/models
  gateway  http://localhost:$GATEWAY_PORT/health   /metrics   /gateway/cache

  smoke      uv run --no-sync inferbench bench --experiment smoke --base-url http://localhost:$GATEWAY_PORT
  through the engine (no gateway in the path):
             uv run --no-sync inferbench bench --experiment smoke --base-url http://localhost:$ENGINE_PORT
  report     uv run --no-sync inferbench report --experiment smoke

  ctrl-c stops both.
EOF
else
  cat <<EOF

  scheduler  http://localhost:$SCHEDULER_PORT/health   /metrics   /internal/v1/state
  gateway    http://localhost:$GATEWAY_PORT/health     /metrics   /gateway/cache

  smoke      uv run --no-sync inferbench bench --experiment smoke --base-url http://localhost:$GATEWAY_PORT
  state      curl -s http://localhost:$SCHEDULER_PORT/internal/v1/state
  report     uv run --no-sync inferbench report --experiment smoke

  ctrl-c stops both.
EOF
fi

wait
