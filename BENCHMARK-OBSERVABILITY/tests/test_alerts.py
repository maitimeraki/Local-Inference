"""Nothing in the rules or the dashboard may query a metric nobody emits.

A rule that names a metric that does not exist never fires, and a dashboard
panel that queries one is a grey box that looks like an outage. Both fail
silently, so they are checked here rather than in production.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from inferobs import metric_names as N

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RULES = ROOT / "observability" / "alerts" / "slo_rules.yml"
K8S_RULES = ROOT / "deploy" / "k8s" / "prometheusrule.yaml"
LIVE_DASHBOARD = ROOT / "observability" / "grafana" / "dashboards" / "live.json"

# PromQL words that are not metric names.
_KEYWORDS = {
    "and",
    "or",
    "unless",
    "by",
    "without",
    "on",
    "ignoring",
    "group_left",
    "group_right",
    "offset",
    "bool",
    "start",
    "end",
}

_IDENT = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_MATCHERS = re.compile(r"\{[^{}]*\}")
_RANGE = re.compile(r"\[[^\]]*\]")
_GROUPING = re.compile(r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^()]*\)")


def _emitted() -> set[str]:
    """Every metric name something in the stack can produce."""
    names = {
        value
        for key, value in vars(N).items()
        if key.isupper() and not key.startswith("L_") and isinstance(value, str)
    }
    out = set(names)
    for name in names:
        # A Histogram also exposes the bucket, count and sum series.
        out.update({f"{name}_bucket", f"{name}_count", f"{name}_sum"})
    return out


def _metrics_in(expr: str) -> set[str]:
    """The metric names a PromQL expression reads."""
    text = _MATCHERS.sub(" ", expr)
    text = _RANGE.sub(" ", text)  # `[5m]` is a duration, not a metric
    for _ in range(3):  # `sum by (le) (rate(x[5m]))` nests the grouping
        text = _GROUPING.sub(" ", text)
    found = set()
    for m in _IDENT.finditer(text):
        name = m.group(0)
        if text[m.end() :].lstrip().startswith("("):
            continue  # a function call
        if name in _KEYWORDS:
            continue
        found.add(name)
    return found


def _groups(path: Path) -> list[dict]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if path is K8S_RULES:
        return doc["spec"]["groups"]
    return doc["groups"]


def _rules(path: Path) -> list[dict]:
    return [rule for group in _groups(path) for rule in group["rules"]]


def _recorded(path: Path) -> set[str]:
    return {r["record"] for r in _rules(path) if "record" in r}


def _exprs(path: Path) -> list[str]:
    return [r["expr"] for r in _rules(path) if "expr" in r]


def _dashboard_exprs(path: Path) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [t["expr"] for p in doc["panels"] for t in p.get("targets", []) if "expr" in t]


# --- the vocabulary ----------------------------------------------------------


def test_every_metric_the_local_rules_use_is_emitted():
    emitted = _emitted()
    recorded = _recorded(LOCAL_RULES)
    unknown = {
        name
        for expr in _exprs(LOCAL_RULES)
        for name in _metrics_in(expr)
        if name not in emitted and name not in recorded
    }
    assert unknown == set(), f"alert rules query metrics nobody emits: {sorted(unknown)}"


def test_every_metric_the_k8s_rules_use_is_emitted():
    emitted = _emitted()
    recorded = _recorded(K8S_RULES)
    unknown = {
        name
        for expr in _exprs(K8S_RULES)
        for name in _metrics_in(expr)
        if name not in emitted and name not in recorded
    }
    assert unknown == set(), f"PrometheusRule queries metrics nobody emits: {sorted(unknown)}"


def test_every_metric_the_live_dashboard_uses_is_emitted():
    emitted = _emitted()
    unknown = {name for expr in _dashboard_exprs(LIVE_DASHBOARD) for name in _metrics_in(expr)}
    unknown -= emitted
    assert unknown == set(), f"dashboard panels query metrics nobody emits: {sorted(unknown)}"


# --- the two rule stacks are the same alerts --------------------------------


def _alerts(path: Path) -> dict[str, str]:
    return {r["alert"]: " ".join(r["expr"].split()) for r in _rules(path) if "alert" in r}


def _records(path: Path) -> dict[str, str]:
    return {r["record"]: " ".join(r["expr"].split()) for r in _rules(path) if "record" in r}


def test_the_local_and_k8s_alert_stacks_are_identical():
    """A local alert that fires and a production alert that does not is the
    worst kind of divergence: the one nobody notices."""
    assert _alerts(LOCAL_RULES) == _alerts(K8S_RULES)


def test_the_local_and_k8s_recording_rules_are_identical():
    assert _records(LOCAL_RULES) == _records(K8S_RULES)


# --- the gates the design doc names -----------------------------------------


def test_the_scheduler_gates_are_alerts():
    """These are the alarms the whole design rests on: if the ledger drifts
    or the engine preempts, the grid is lying to itself."""
    alerts = _alerts(LOCAL_RULES)
    for name in (
        "ReservationDrift",
        "UnexpectedPreemption",
        "SchedulerHoldQueueGrowing",
        "SchedulerDispatchStalled",
        "SchedulerDown",
    ):
        assert name in alerts, f"{name} is not an alert"


def test_the_drift_alert_is_on_the_drift_metric():
    assert "sched_reservation_drift_blocks" in _alerts(LOCAL_RULES)["ReservationDrift"]


def test_the_preemption_alert_is_on_the_preemption_metric():
    assert "sched_preemption_total" in _alerts(LOCAL_RULES)["UnexpectedPreemption"]
