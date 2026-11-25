"""Comparison report: the deliverable that answers the brief's question.

"For a particular model and workload, which inference configuration provides the
best balance of throughput, latency, memory consumption and cost?"

Rendered as markdown because it diffs, greps and survives in a pull request.
Plots are optional — a missing matplotlib degrades to tables, not a crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inferbench.store import Store

# (column header, summary metric, stat) rendered left to right.
COLUMNS: list[tuple[str, str, str]] = [
    ("TTFT p50", "ttft", "p50"),
    ("TTFT p95", "ttft", "p95"),
    ("ITL mean", "itl", "mean"),
    ("TPOT mean", "tpot", "mean"),
    ("E2E p95", "e2e", "p95"),
    ("E2E p99", "e2e", "p99"),
    ("out tok/s", "output_tps", "mean"),
    ("req/s", "request_tps", "mean"),
    ("GPU mem", "gpu_mem", "max"),
    ("$/Mtok", "cost_per_mtok_output", "mean"),
    ("$/1k req", "cost_per_1k_requests", "mean"),
]


@dataclass
class RunRow:
    run_id: str
    config: str
    workload: str
    concurrency: int | None
    repeat: int | None
    n_requests: int
    n_ok: int
    values: dict[str, float]

    @property
    def failure_rate(self) -> float:
        if not self.n_requests:
            return 0.0
        return 1.0 - self.n_ok / self.n_requests

    @property
    def is_clean(self) -> bool:
        """Every request in the run succeeded.

        A run that served nothing has no meaningful throughput, latency or
        cost to compare — the metrics are absent, and an absent metric
        normalised to 0.5 scores as a middling run rather than as a failure.
        """
        return self.n_requests > 0 and self.n_ok >= self.n_requests


def collect_rows(store: Store, experiment_id: str | None = None) -> list[RunRow]:
    rows: list[RunRow] = []
    for run in store.runs(experiment_id):
        summary = store.summary_for(run["run_id"])
        values = {
            f"{metric}|{stat}": summary.get(metric, {}).get(stat)
            for _, metric, stat in COLUMNS
        }
        label = run.get("label") or {}
        if isinstance(label, str):
            import json

            label = json.loads(label)
        rows.append(
            RunRow(
                run_id=run["run_id"],
                config=run.get("config") or "-",
                workload=run.get("workload") or "-",
                concurrency=run.get("concurrency"),
                repeat=label.get("repeat"),
                n_requests=run.get("n_requests") or 0,
                n_ok=run.get("n_ok") or 0,
                values=values,
            )
        )
    return rows


def _fmt(value: float | None, metric: str) -> str:
    if value is None:
        return "-"
    if metric == "gpu_mem":
        return f"{value / 1e9:.2f} GB"
    if metric == "cost_per_mtok_output" or metric == "cost_per_1k_requests":
        return f"{value:.4f}"
    if value < 10:
        return f"{value:.3f}"
    return f"{value:.2f}"


def render_markdown(rows: list[RunRow], title: str = "Benchmark comparison") -> str:
    headers = ["config", "workload", "conc", "rep", "ok/req"] + [c[0] for c in COLUMNS]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in sorted(rows, key=lambda x: (x.config, x.concurrency or 0, x.repeat or 0)):
        cells = [
            r.config,
            r.workload,
            str(r.concurrency) if r.concurrency is not None else "-",
            str(r.repeat) if r.repeat is not None else "-",
            f"{r.n_ok}/{r.n_requests}",
        ]
        cells += [_fmt(r.values.get(f"{m}|{s}"), m) for _, m, s in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def rank(
    rows: list[RunRow],
    *,
    max_ttft_p95_s: float | None = None,
    weight_throughput: float = 1.0,
    weight_latency: float = 1.0,
    weight_cost: float = 1.0,
) -> list[tuple[RunRow, float]]:
    """Score configurations by a normalised balance of throughput, latency, cost.

    Min-max normalised across the rows so no metric dominates by unit alone.
    Two filters run first. A latency ceiling (`max_ttft_p95_s`) excludes the
    fastest-but-unusable configurations. Runs with failed requests are
    excluded too: they have no latency to measure, and scoring them would
    rank a broken configuration as a middling one.

    Returns (row, score) sorted best-first. Discarded rows are excluded.
    """
    candidates = [
        r
        for r in rows
        if r.is_clean
        and (max_ttft_p95_s is None or (r.values.get("ttft|p95") or 0) <= max_ttft_p95_s)
    ]
    if not candidates:
        return []

    def span(key: str) -> tuple[float, float]:
        vals = [r.values.get(key) for r in candidates if r.values.get(key) is not None]
        return (min(vals), max(vals)) if vals else (0.0, 0.0)

    def norm(value: float | None, lo: float, hi: float, invert: bool) -> float:
        if value is None or hi == lo:
            return 0.5
        n = (value - lo) / (hi - lo)
        return 1.0 - n if invert else n

    t_lo, t_hi = span("output_tps|mean")
    l_lo, l_hi = span("ttft|p95")
    c_lo, c_hi = span("cost_per_mtok_output|mean")

    scored: list[tuple[RunRow, float]] = []
    for r in candidates:
        score = (
            weight_throughput * norm(r.values.get("output_tps|mean"), t_lo, t_hi, invert=False)
            + weight_latency * norm(r.values.get("ttft|p95"), l_lo, l_hi, invert=True)
            + weight_cost * norm(r.values.get("cost_per_mtok_output|mean"), c_lo, c_hi, invert=True)
        )
        scored.append((r, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def write_report(
    store: Store,
    out_path: str | Path,
    *,
    experiment_id: str | None = None,
    title: str = "Benchmark comparison",
    max_ttft_p95_s: float | None = None,
) -> Path:
    """Write the markdown report, and PNG curves when matplotlib is present."""
    rows = collect_rows(store, experiment_id)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    parts = [render_markdown(rows, title), "", "## Ranking", ""]
    if max_ttft_p95_s is not None:
        parts += [
            f"Configurations whose TTFT p95 exceeds {max_ttft_p95_s:g}s are "
            "excluded from the ranking below.",
            "",
        ]
    failed = [r for r in rows if not r.is_clean]
    if failed:
        parts += [
            f"{len(failed)} of {len(rows)} runs had failed requests and are "
            "excluded from the ranking below: a run that served nothing has no "
            "throughput, latency or cost to compare.",
            "",
        ]
    best = rank(rows, max_ttft_p95_s=max_ttft_p95_s)
    if best:
        parts.append("| rank | config | conc | ok/req | score | out tok/s | TTFT p95 | $/Mtok |")
        parts.append("|---|---|---|---|---|---|---|---|")
        for i, (r, score) in enumerate(best, 1):
            parts.append(
                f"| {i} | {r.config} | {r.concurrency} | {r.n_ok}/{r.n_requests} | "
                f"{score:.3f} | "
                f"{_fmt(r.values.get('output_tps|mean'), 'output_tps')} | "
                f"{_fmt(r.values.get('ttft|p95'), 'ttft')} | "
                f"{_fmt(r.values.get('cost_per_mtok_output|mean'), 'cost_per_mtok_output')} |"
            )
        top = best[0][0]
        parts += [
            "",
            f"**Best balance:** `{top.config}` at concurrency {top.concurrency} "
            f"({top.workload}).",
        ]
    elif rows:
        parts.append(
            f"_{len(rows)} runs recorded, but none qualified for the ranking._"
        )
    else:
        parts.append("_No runs recorded._")

    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    _maybe_plot(rows, out.with_suffix(".png"))
    return out


def _maybe_plot(rows: list[RunRow], path: Path) -> bool:
    """TTFT p95 and throughput against concurrency. Optional by design."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    points = sorted(
        (r for r in rows if r.concurrency is not None and r.is_clean),
        key=lambda r: r.concurrency or 0,
    )
    if not points:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    by_config: dict[str, list[RunRow]] = {}
    for r in points:
        by_config.setdefault(r.config, []).append(r)

    for config, group in by_config.items():
        xs = [r.concurrency for r in group]
        ax1.plot(xs, [r.values.get("ttft|p95") for r in group], marker="o", label=config)
        ax2.plot(xs, [r.values.get("output_tps|mean") for r in group], marker="o", label=config)

    ax1.set(xlabel="concurrency", ylabel="TTFT p95 (s)", title="Latency vs concurrency")
    ax2.set(xlabel="concurrency", ylabel="output tokens/s", title="Throughput vs concurrency")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def slo_section(store: Store, experiment_id: str | None, targets: list) -> str:
    """Evaluate SLO targets against the latest run of each configuration."""
    from inferobs.slo import evaluate, summary_from_store_rows

    lines = ["## SLO compliance", ""]
    runs = store.runs(experiment_id)
    if not runs:
        return "\n".join(lines + ["_No runs recorded._", ""])

    for run in runs:
        summary = store.summary_for(run["run_id"])
        flat = [
            {"metric": m, "stat": s, "value": v}
            for m, stats in summary.items()
            for s, v in stats.items()
        ]
        results = evaluate(targets, summary_from_store_rows(flat))
        if not results:
            continue
        lines.append(f"### {run['run_id']} ({run.get('config')} c={run.get('concurrency')})")
        lines.append("")
        lines += [f"- {r.render()}" for r in results]
        lines.append("")
    return "\n".join(lines)
