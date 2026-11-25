"""The money tests.

If `percentile` or `build_request_metrics` is wrong, every number the platform
publishes is wrong, and wrong in a way that looks plausible. So these check
against an independent reference (numpy) and against hand-computed timelines.
"""

from __future__ import annotations

import numpy as np
import pytest
from inferbench import metrics as M

MS = 1_000_000  # ns per ms


# --- percentiles ------------------------------------------------------------


def test_percentile_matches_numpy_across_sizes():
    rng = np.random.default_rng(1234)
    for n in (1, 2, 3, 10, 100, 1000):
        data = rng.normal(loc=5.0, scale=2.0, size=n)
        values = [float(v) for v in data]
        for q in (0, 1, 25, 50, 75, 90, 95, 99, 100):
            assert M.percentile(values, q) == pytest.approx(
                float(np.percentile(data, q)), rel=1e-9, abs=1e-12
            ), f"n={n} q={q}"


def test_percentile_empty_is_none_not_zero():
    # Zero would report a perfect score for a run that produced nothing.
    assert M.percentile([], 95) is None


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        M.percentile([1.0], 101)


def test_summarize_empty_is_empty_dict():
    assert M.summarize([]) == {}


def test_summarize_has_all_stats():
    stats = M.summarize([1.0, 2.0, 3.0, 4.0])
    assert set(stats) == {"mean", "p50", "p95", "p99", "min", "max"}
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["min"] == 1.0 and stats["max"] == 4.0


# --- latency derivation -----------------------------------------------------


def test_build_request_metrics_hand_computed():
    """Three tokens at 100/150/200 ms; start at 0. TTFT=0.1, ITL=0.05, e2e=0.2."""
    start = 0
    chunks = [(100 * MS, 1), (150 * MS, 1), (200 * MS, 1)]
    m = M.build_request_metrics("r1", start, 210 * MS, chunks, output_tokens=3)

    assert m.ttft_s == pytest.approx(0.1)
    assert m.e2e_s == pytest.approx(0.2)
    assert m.itl_mean_s == pytest.approx(0.05)
    assert m.tpot_s == pytest.approx((0.2 - 0.1) / 2)
    assert m.output_tokens == 3
    assert m.ok


def test_e2e_ignores_time_after_last_token():
    """e2e is last-token arrival minus send, not connection close."""
    chunks = [(10 * MS, 1), (20 * MS, 1)]
    m = M.build_request_metrics("r1", 0, 999 * MS, chunks)
    assert m.e2e_s == pytest.approx(0.02)


def test_multi_token_chunk_splits_the_gap():
    """A chunk carrying 2 tokens must contribute two ITL samples."""
    chunks = [(10 * MS, 1), (50 * MS, 2)]
    m = M.build_request_metrics("r1", 0, 50 * MS, chunks)
    assert len(m.itls) == 2
    assert all(v == pytest.approx(0.02) for v in m.itls)
    assert m.output_tokens == 3


def test_single_token_has_no_tpot():
    """TPOT is a per-token-after-first rate; one token has no such rate."""
    m = M.build_request_metrics("r1", 0, 10 * MS, [(5 * MS, 1)], output_tokens=1)
    assert m.tpot_s is None
    assert m.ttft_s == pytest.approx(0.005)


def test_no_tokens_leaves_ttft_none():
    m = M.build_request_metrics("r1", 0, 10 * MS, [], status=M.STATUS_TRUNCATED)
    assert m.ttft_s is None
    assert not m.ok


def test_output_tokens_falls_back_to_chunk_count():
    m = M.build_request_metrics("r1", 0, 10 * MS, [(1 * MS, 1), (2 * MS, 1)])
    assert m.output_tokens == 2


def test_row_drops_raw_itls_but_keeps_summary():
    m = M.build_request_metrics("r1", 0, 20 * MS, [(5 * MS, 1), (10 * MS, 1)])
    row = m.to_row()
    assert "itls" not in row
    assert row["itl_mean_s"] is not None
    assert row["ok"] is True


# --- SSE parsing ------------------------------------------------------------


def test_parse_sse_data():
    assert M.parse_sse_data(b'data: {"a": 1}') == {"a": 1}
    assert M.parse_sse_data(b"data: [DONE]") is None
    assert M.parse_sse_data(b": keep-alive") is None
    assert M.parse_sse_data(b"") is None
    assert M.parse_sse_data(b"data: not json") is None


def test_chunk_text_handles_both_apis():
    assert M.chunk_text({"choices": [{"text": "hi"}]}) == "hi"
    assert M.chunk_text({"choices": [{"delta": {"content": "x"}}]}) == "x"
    assert M.chunk_text({"choices": [{"delta": {}}]}) == ""
    assert M.chunk_text({}) == ""


def test_chunk_token_count_counts_only_text_bearing_chunks():
    assert M.chunk_token_count({"choices": [{"text": "hi"}]}) == 1
    assert M.chunk_token_count({"choices": [{"text": ""}]}) == 0
    assert M.chunk_token_count({"choices": []}) == 0


def test_usage_counts():
    assert M.usage_counts({"prompt_tokens": 5, "completion_tokens": 7}) == (5, 7)
    assert M.usage_counts(None) == (None, None)


def test_chunk_finish_reason():
    assert M.chunk_finish_reason({"choices": [{"finish_reason": "length"}]}) == "length"
    assert M.chunk_finish_reason({"choices": []}) is None
