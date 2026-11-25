"""Result persistence: sqlite3 from the standard library.

One database file, three tables, no ORM and no server. A benchmark result is a
row of numbers plus the context that made it meaningful, and sqlite queries
that better than a directory of JSON does.

A run row records everything needed to reproduce the number months later:
model, dtype, max_model_len, sampling, engine flags, vLLM version, GPU, git SHA.
A comparison that cannot be audited is a rumour.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from inferbench import metrics as M

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    created_at       REAL NOT NULL,
    experiment_id    TEXT,
    workload         TEXT,
    config           TEXT,
    concurrency      INTEGER,
    model            TEXT,
    dtype            TEXT,
    max_model_len    INTEGER,
    sampling         TEXT,
    engine_flags     TEXT,
    vllm_version     TEXT,
    gpu              TEXT,
    git_sha          TEXT,
    label            TEXT,
    n_requests       INTEGER DEFAULT 0,
    n_ok             INTEGER DEFAULT 0,
    duration_s       REAL,
    status           TEXT DEFAULT 'running',
    notes            TEXT
);
CREATE TABLE IF NOT EXISTS requests (
    run_id        TEXT NOT NULL,
    request_id    TEXT NOT NULL,
    status        TEXT,
    error         TEXT,
    ok            INTEGER,
    ttft_s        REAL,
    e2e_s         REAL,
    tpot_s        REAL,
    itl_mean_s    REAL,
    itl_p95_s     REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    queue_time_s  REAL,
    trace_id      TEXT,
    started_at    REAL,
    finished_at   REAL
);
CREATE TABLE IF NOT EXISTS summaries (
    run_id  TEXT NOT NULL,
    metric  TEXT NOT NULL,
    stat    TEXT NOT NULL,
    value   REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_run ON requests(run_id);
CREATE INDEX IF NOT EXISTS idx_summaries_run ON summaries(run_id);
"""

# Per-request latency columns, and the measurement name each maps to.
LATENCY_COLUMNS = {
    "ttft": "ttft_s",
    "itl": "itl_mean_s",
    "tpot": "tpot_s",
    "e2e": "e2e_s",
    "queue": "queue_time_s",
}


class RunMeta(BaseModel):
    """Everything that must be held constant for a comparison to be valid."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    experiment_id: str | None = None
    workload: str | None = None
    config: str | None = None
    concurrency: int | None = None
    model: str | None = None
    dtype: str | None = None
    max_model_len: int | None = None
    sampling: dict = Field(default_factory=dict)
    engine_flags: dict = Field(default_factory=dict)
    vllm_version: str | None = None
    gpu: str | None = None
    git_sha: str | None = None
    label: dict = Field(default_factory=dict)
    notes: str | None = None

    def to_params(self) -> dict:
        d = self.model_dump()
        for k in ("sampling", "engine_flags", "label"):
            d[k] = json.dumps(d[k], sort_keys=True)
        d["created_at"] = time.time()
        return d


class Store:
    def __init__(self, path: str | Path = "results/bench.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- writes ------------------------------------------------------------

    def start_run(self, meta: RunMeta) -> str:
        params = meta.to_params()
        cols = ", ".join(params)
        marks = ", ".join("?" * len(params))
        with self._lock:
            self.conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", list(params.values()))
            self.conn.commit()
        return meta.run_id

    def add_requests(self, run_id: str, results: list[M.RequestMetrics]) -> None:
        rows = [(run_id, *self._request_values(r)) for r in results]
        with self._lock:
            self.conn.executemany(
                """INSERT INTO requests (
                    run_id, request_id, status, error, ok, ttft_s, e2e_s, tpot_s,
                    itl_mean_s, itl_p95_s, input_tokens, output_tokens,
                    queue_time_s, trace_id, started_at, finished_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            self.conn.execute(
                "UPDATE runs SET n_requests = ?, n_ok = ? WHERE run_id = ?",
                (len(results), sum(1 for r in results if r.ok), run_id),
            )
            self.conn.commit()

    @staticmethod
    def _request_values(r: M.RequestMetrics) -> tuple:
        return (
            r.request_id,
            r.status,
            r.error,
            int(r.ok),
            r.ttft_s,
            r.e2e_s,
            r.tpot_s,
            r.itl_mean_s,
            r.itl_p95_s,
            r.input_tokens,
            r.output_tokens,
            r.queue_time_s,
            r.trace_id,
            r.started_at,
            r.finished_at,
        )

    def save_summary(self, run_id: str, summary: dict[str, dict[str, float]]) -> None:
        rows = [
            (run_id, metric, stat, float(value))
            for metric, stats in summary.items()
            for stat, value in stats.items()
        ]
        with self._lock:
            self.conn.execute("DELETE FROM summaries WHERE run_id = ?", (run_id,))
            self.conn.executemany(
                "INSERT INTO summaries (run_id, metric, stat, value) VALUES (?,?,?,?)", rows
            )
            self.conn.commit()

    def finish_run(
        self, run_id: str, duration_s: float, status: str = "ok", notes: str = ""
    ) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE runs SET duration_s = ?, status = ?, notes = ? WHERE run_id = ?",
                (duration_s, status, notes, run_id),
            )
            self.conn.commit()

    # -- reads -------------------------------------------------------------

    def runs(self, experiment_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM runs"
        args: tuple = ()
        if experiment_id:
            sql += " WHERE experiment_id = ?"
            args = (experiment_id,)
        sql += " ORDER BY created_at"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def requests_for(self, run_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM requests WHERE run_id = ?", (run_id,))
        return [dict(r) for r in rows]

    def summary_for(self, run_id: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for row in self.conn.execute(
            "SELECT metric, stat, value FROM summaries WHERE run_id = ?", (run_id,)
        ):
            out.setdefault(row["metric"], {})[row["stat"]] = row["value"]
        return out


def aggregate_requests(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Latency distributions across a run's requests.

    Only non-null values count. A failed request has no TTFT, and treating a
    missing TTFT as 0 would report a fast run for a broken one.
    """
    out: dict[str, dict[str, float]] = {}
    for metric, col in LATENCY_COLUMNS.items():
        values = [r[col] for r in rows if r.get(col) is not None]
        if values:
            out[metric] = M.summarize(values)
    return out


def throughput(rows: list[dict], duration_s: float) -> dict[str, dict[str, float]]:
    """Run-level rates. Duration is wall-clock for the whole batch, which is
    what makes these comparable across concurrency levels.

    `request_tps` counts *successful* requests only. A connection-refused
    failure returns in microseconds, so counting failures makes a run that
    served nothing look faster than one that actually worked.
    """
    if duration_s <= 0:
        return {}
    out_tokens = sum(r.get("output_tokens") or 0 for r in rows)
    in_tokens = sum(r.get("input_tokens") or 0 for r in rows)
    n_ok = sum(1 for r in rows if r.get("ok"))
    return {
        "output_tps": {"mean": out_tokens / duration_s},
        "input_tps": {"mean": in_tokens / duration_s},
        "request_tps": {"mean": n_ok / duration_s},
    }


def failure_summary(rows: list[dict]) -> dict[str, int]:
    """Counts by failure status. Successes are excluded."""
    counts: dict[str, int] = {}
    for r in rows:
        if not r.get("ok"):
            counts[r.get("status") or "unknown"] = counts.get(r.get("status") or "unknown", 0) + 1
    return counts
