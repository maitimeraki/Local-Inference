"""Command line entry point.

Every subcommand imports what it needs lazily, so `inferbench --help` works on a
machine without vLLM, torch or a GPU — which is most machines that will ever
read this code.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from inferbench.config import (
    SamplingConfig,
    config_dir,
    load_engines,
    load_experiments,
    load_pricing,
    load_workloads,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = WORKSPACE_ROOT / "results"
DEFAULT_DB = RESULTS_DIR / "bench.db"
DEFAULT_REPORT = RESULTS_DIR / "report.md"
COMPOSE_FILE = WORKSPACE_ROOT / "observability" / "compose.yaml"


# --- bench ------------------------------------------------------------------


def cmd_bench(args: argparse.Namespace) -> int:
    from inferbench.runner import load_tokenizer, run_experiment
    from inferbench.store import Store

    experiments = {e.id: e for e in load_experiments()}
    if args.experiment not in experiments:
        print(f"unknown experiment {args.experiment!r}; have: {', '.join(experiments)}")
        return 2
    exp = experiments[args.experiment]

    workloads = load_workloads()
    engines = load_engines()
    workload = workloads[exp.workload]
    engine_cfg = engines[exp.server]

    print(f"experiment {exp.id}: {exp.description or exp.workload}")
    print(f"  workload  {workload.name}: {workload.input_tokens} in / {workload.output_tokens} out")
    print(f"  engine    {engine_cfg.name}: {engine_cfg.model}")
    print(f"  endpoint  {args.base_url}")

    tokenizer = load_tokenizer(engine_cfg.model)
    store = Store(args.db)
    try:
        run_ids = asyncio.run(
            run_experiment(
                store,
                exp,
                workloads,
                engines,
                tokenizer,
                args.base_url,
                load_pricing(),
                SamplingConfig(),
                pushgateway_url=args.pushgateway,
            )
        )
        runs = {r["run_id"]: r for r in store.runs(exp.id)}
        broken = [
            runs[rid]
            for rid in run_ids
            if rid in runs and (runs[rid].get("n_ok") or 0) < (runs[rid].get("n_requests") or 0)
        ]
    finally:
        store.close()

    print(f"\n{len(run_ids)} runs recorded in {args.db}")
    print(f"next: inferbench report --experiment {exp.id}")
    if broken:
        for r in broken:
            print(
                f"!! {r['run_id']} ({r.get('config')}): "
                f"{r.get('n_ok')}/{r.get('n_requests')} requests succeeded",
                file=sys.stderr,
            )
        print(
            f"!! {len(broken)} of {len(run_ids)} runs had failed requests. "
            "The runs are recorded, but a run that failed is not a measurement.",
            file=sys.stderr,
        )
        return 1
    return 0


# --- inspect ----------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> int:
    from inferbench.inspect import SchedulerUnreachable, run_inspection
    from inferbench.runner import load_tokenizer
    from inferbench.workloads import PromptFactory

    engines = load_engines()
    if args.config not in engines:
        print(f"unknown engine config {args.config!r}; have: {', '.join(engines)}")
        return 2
    cfg = engines[args.config]

    workloads = load_workloads()
    if args.workload not in workloads:
        print(f"unknown workload {args.workload!r}; have: {', '.join(workloads)}")
        return 2
    workload = workloads[args.workload]

    tokenizer = load_tokenizer(cfg.model)
    prompt = PromptFactory(tokenizer, workload).build(0).text

    print(f"stepping through {cfg.name} on {workload.name} (V1 engine)")
    try:
        reports = run_inspection(
            cfg, prompt, max_tokens=args.max_tokens, step_limit=args.step_limit
        )
    except SchedulerUnreachable as exc:
        print(f"cannot inspect: {exc}")
        return 1
    print(f"\n{len(reports)} steps")
    return 0


# --- serve ------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    from inferbench.server import serve

    engines = load_engines()
    if args.config not in engines:
        print(f"unknown engine config {args.config!r}; have: {', '.join(engines)}")
        return 2
    print(f"serving {engines[args.config].model} on :{args.port}")
    serve(engines[args.config], host=args.host, port=args.port)
    return 0


# --- report -----------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    from inferbench.report import collect_rows, rank, write_report
    from inferbench.store import Store

    store = Store(args.db)
    try:
        path = write_report(
            store,
            args.out,
            experiment_id=args.experiment,
            max_ttft_p95_s=args.max_ttft_p95,
        )
        rows = collect_rows(store, args.experiment)
        ranked = rank(rows, max_ttft_p95_s=args.max_ttft_p95)
    finally:
        store.close()
    print(f"wrote {path}")
    if rows and not ranked:
        print(
            f"!! {len(rows)} runs recorded, but none qualified for the ranking. "
            "The report says why.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_slo(args: argparse.Namespace) -> int:
    from inferobs.slo import evaluate, load_slo_targets, summary_from_store_rows

    from inferbench.store import Store

    targets = load_slo_targets(config_dir() / "slo_targets.yaml")
    if not targets:
        print("no SLO targets defined in config/slo_targets.yaml")
        return 0

    store = Store(args.db)
    try:
        runs = store.runs(args.experiment)
        if not runs:
            print("no runs recorded")
            return 0
        for run in runs:
            summary = store.summary_for(run["run_id"])
            flat = [
                {"metric": m, "stat": s, "value": v}
                for m, stats in summary.items()
                for s, v in stats.items()
            ]
            print(f"{run['run_id']}  {run.get('config')}  c={run.get('concurrency')}")
            for result in evaluate(targets, summary_from_store_rows(flat)):
                print(f"  {result.render()}")
    finally:
        store.close()
    return 0


# --- gateway ----------------------------------------------------------------


def cmd_gateway(args: argparse.Namespace) -> int:
    from infergw.app import serve

    print(f"gateway on :{args.port}")
    serve(host=args.host, port=args.port)
    return 0


# --- observability stack ----------------------------------------------------


def _compose(args: argparse.Namespace, *cmd: str) -> int:
    if not COMPOSE_FILE.exists():
        print(f"compose file missing: {COMPOSE_FILE}")
        return 2
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), *cmd], check=False
        )
    except FileNotFoundError:
        print("docker not on PATH — the observability stack needs Docker running")
        return 2
    return result.returncode


def cmd_up(args: argparse.Namespace) -> int:
    rc = _compose(args, "up", "-d")
    if rc == 0:
        print("\nGrafana   http://localhost:3000   (anonymous admin)")
        print("Prometheus http://localhost:9090")
    return rc


def cmd_down(args: argparse.Namespace) -> int:
    return _compose(args, "down")


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inferbench", description="vLLM benchmarking harness")
    p.add_argument("--db", default=str(DEFAULT_DB), help="sqlite results database")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run an experiment")
    b.add_argument("--experiment", required=True, help="experiment id from experiments.yaml")
    b.add_argument("--base-url", default="http://localhost:8000", help="server or gateway URL")
    b.add_argument(
        "--pushgateway", default="http://localhost:9091", help="pushgateway URL ('' to skip)"
    )
    b.set_defaults(func=cmd_bench)

    i = sub.add_parser("inspect", help="step through the engine iteration by iteration")
    i.add_argument("--config", required=True, help="engine config name")
    i.add_argument("--workload", required=True, help="workload profile name")
    i.add_argument("--max-tokens", type=int, default=64)
    i.add_argument("--step-limit", type=int, default=500)
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--config", required=True, help="engine config name")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    r = sub.add_parser("report", help="write the comparison report")
    r.add_argument("--experiment", default=None)
    r.add_argument("--out", default=str(DEFAULT_REPORT))
    r.add_argument(
        "--max-ttft-p95",
        type=float,
        default=None,
        help="exclude configurations whose TTFT p95 exceeds this, in seconds",
    )
    r.set_defaults(func=cmd_report)

    sl = sub.add_parser("slo", help="evaluate SLO targets against recorded runs")
    sl.add_argument("--experiment", default=None)
    sl.set_defaults(func=cmd_slo)

    g = sub.add_parser("gateway", help="run the request router")
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--port", type=int, default=9000)
    g.set_defaults(func=cmd_gateway)

    u = sub.add_parser("up", help="start the observability stack")
    u.set_defaults(func=cmd_up)
    d = sub.add_parser("down", help="stop the observability stack")
    d.set_defaults(func=cmd_down)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "pushgateway", None) == "":
        args.pushgateway = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
