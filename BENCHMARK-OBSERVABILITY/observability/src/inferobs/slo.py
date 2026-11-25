"""SLO definitions and compliance evaluation.

Targets live in `config/slo_targets.yaml` and are **derived from measured
results**, never guessed up front. Guessing the target is how an SLO becomes
fiction: you either set it so loose it never fires, or so tight the team stops
believing it.

Each target names a measurement and statistic produced by the benchmark
harness, so an SLO is literally "this measured number must stay on this side of
this line".
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

Comparison = Literal["<=", ">=", "<", ">"]

_OPS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}


class SLOTarget(BaseModel):
    """One objective. `stat` is a key of `metric_names.STATS`."""

    id: str
    description: str = ""
    workload: str | None = None
    config: str | None = None
    measurement: str
    stat: str
    threshold: float
    comparison: Comparison
    window: str = "5m"

    def check(self, value: float) -> bool:
        """True when `value` satisfies the objective."""
        return _OPS[self.comparison](value, self.threshold)

    def description_line(self) -> str:
        return f"{self.measurement} {self.stat} {self.comparison} {self.threshold}"


class SLOResult(BaseModel):
    target: SLOTarget
    observed: float | None
    compliant: bool
    measured: bool = True
    note: str = ""

    def render(self) -> str:
        if not self.measured:
            return f"[SKIP] {self.target.id}: {self.target.description_line()} (not measured)"
        mark = "PASS" if self.compliant else "FAIL"
        return (
            f"[{mark}] {self.target.id}: {self.target.description_line()} "
            f"(observed {self.observed:.4f}) {self.note}".rstrip()
        )


def load_slo_targets(path: str | Path) -> list[SLOTarget]:
    """Load targets from YAML. A missing file is an empty list, not an error."""
    p = Path(path)
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = raw.get("targets", raw) if isinstance(raw, dict) else raw
    return [SLOTarget(**e) for e in entries]


def evaluate(targets: list[SLOTarget], summary: dict[str, dict[str, float]]) -> list[SLOResult]:
    """Check every target against a run summary.

    `summary` is keyed `measurement -> stat -> value`, exactly how the harness
    stores aggregates: `{"ttft": {"p95": 0.31, "mean": 0.12}, ...}`.
    """
    results: list[SLOResult] = []
    for t in targets:
        value = summary.get(t.measurement, {}).get(t.stat)
        if value is None:
            # Never reported as passing: an unmeasured objective is unknown, and
            # rendering unknown as PASS is how a broken pipeline looks healthy.
            results.append(
                SLOResult(target=t, observed=None, compliant=False, measured=False)
            )
            continue
        results.append(SLOResult(target=t, observed=value, compliant=t.check(value)))
    return results


def summary_from_store_rows(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Adapt `store.aggregate()` output into the mapping `evaluate()` expects."""
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        out.setdefault(row["metric"], {})[row["stat"]] = float(row["value"])
    return out
