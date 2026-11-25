# Kubernetes deployment path

This directory is the **deployment** path for the inference platform. It is
**not** the benchmark path — benchmarking and the observability stack run
natively in WSL2 (scheduler on `:9001`, gateway on `:9000`), where Prometheus
scrapes them over `host.docker.internal` (see `../../observability/compose.yaml`).

The full stack (scheduler + Prometheus + Tempo + Grafana + the control plane)
does **not** fit on a 6 GB GPU. These manifests are therefore validated on a
small local cluster — kind or Docker Desktop — with a small model, rather than
on the target hardware.

```sh
kubectl kustomize .     # render
kubectl apply -k .      # apply (once images exist)
```

## The two tiers

| | grid (default) | A/B baseline |
|---|---|---|
| engine | in-process, inside the scheduler pod | `vllm-deployment.yaml`, its own pod |
| gateway points at | `http://scheduler:9001` | the vLLM service |
| K8s files | `scheduler-deployment.yaml` + `gateway-deployment.yaml` | `vllm-deployment.yaml` + `gateway-deployment.yaml` |

The scheduler is **not** scalable and has no HPA: it owns the GPU and the
KV-block ledger, and a second replica would be a second engine on the same
card. Its `strategy` is `Recreate` for the same reason — a rolling update
would keep the old pod on the GPU while the new one allocates its own.

`model-cache-pvc.yaml` holds the shared model cache, so both tiers can use it.

## All image names

`vllm-server:latest`, `infergw:latest`, `infersched:latest` are placeholders.
They must be built and pushed to a registry the cluster can pull from before
`kubectl apply`. The scheduler image must carry the repo's `config/` directory:
it reads `config/scheduler.yaml` and the engine config that file points at.
