"""Workload prompt construction.

Exact token lengths matter because they are what makes two runs comparable: if
the prompt length drifts between runs, the difference you measure is the drift.
"""

from __future__ import annotations

from inferbench.config import WorkloadProfile
from inferbench.workloads import PromptFactory, text_for_tokens


def test_text_for_tokens_is_exact(tokenizer):
    for n in (1, 5, 64, 512):
        assert len(tokenizer.encode(text_for_tokens(tokenizer, n, 0))) == n


def test_text_for_tokens_zero_is_empty(tokenizer):
    assert text_for_tokens(tokenizer, 0) == ""


def test_shared_prefix_is_byte_identical_across_requests(tokenizer):
    """This is what makes the prefix cache able to hit at all."""
    profile = WorkloadProfile(
        name="shared",
        input_tokens=20,
        output_tokens=5,
        prefix_mode="shared",
        shared_prefix_tokens=8,
        requests=4,
    )
    prompts = PromptFactory(tokenizer, profile).build_many(4)

    prefixes = {p.text[:8] for p in prompts}
    assert len(prefixes) == 1, "requests must share one identical prefix"
    assert all(len(tokenizer.encode(p.text)) == 20 for p in prompts)
    assert all(p.shared_prefix_tokens == 8 for p in prompts)


def test_unique_mode_produces_distinct_prompts(tokenizer):
    profile = WorkloadProfile(name="unique", input_tokens=20, output_tokens=5, requests=4)
    prompts = PromptFactory(tokenizer, profile).build_many(4)

    assert len({p.text for p in prompts}) == 4
    assert all(p.shared_prefix_tokens == 0 for p in prompts)
    assert all(len(tokenizer.encode(p.text)) == 20 for p in prompts)


def test_suffix_differs_even_when_prefix_is_shared(tokenizer):
    """Two identically-prefixed requests must still differ, or caching would
    return the wrong answer for one of them."""
    profile = WorkloadProfile(
        name="shared",
        input_tokens=20,
        output_tokens=5,
        prefix_mode="shared",
        shared_prefix_tokens=8,
        requests=4,
    )
    prompts = PromptFactory(tokenizer, profile).build_many(4)
    assert len({p.text for p in prompts}) == 4


def test_prompts_scale_with_request_count(tokenizer):
    profile = WorkloadProfile(name="p", input_tokens=10, output_tokens=2, requests=7)
    assert len(PromptFactory(tokenizer, profile).build_many(7)) == 7


# --- salt: one run must not replay another run's prompts ---------------------


def test_a_salted_factory_asks_a_different_question(tokenizer):
    """A server that already has the prompt cached is not being measured.

    Replaying the same prompts measures the prefix cache the previous run
    warmed, which is exactly the number the benchmark is not supposed to
    report.
    """
    profile = WorkloadProfile(name="p", input_tokens=20, output_tokens=5, requests=4)
    first = {p.text for p in PromptFactory(tokenizer, profile, salt=0).build_many(4)}
    second = {p.text for p in PromptFactory(tokenizer, profile, salt=1).build_many(4)}

    assert first and second
    assert not first & second


def test_the_shared_prefix_is_never_salted(tokenizer):
    """Its cache hit is the experiment; salting it would measure nothing."""
    profile = WorkloadProfile(
        name="shared",
        input_tokens=20,
        output_tokens=5,
        prefix_mode="shared",
        shared_prefix_tokens=8,
        requests=4,
    )
    a = PromptFactory(tokenizer, profile, salt=0).build_many(4)
    b = PromptFactory(tokenizer, profile, salt=1).build_many(4)

    # One prefix across both salts and all requests.
    assert len({p.text[:8] for p in a + b}) == 1


def test_the_same_salt_asks_the_same_question(tokenizer):
    """A run that cannot be replayed is not an experiment."""
    profile = WorkloadProfile(name="p", input_tokens=20, output_tokens=5, requests=4)
    a = PromptFactory(tokenizer, profile, salt=7).build_many(4)
    b = PromptFactory(tokenizer, profile, salt=7).build_many(4)

    assert [p.text for p in a] == [p.text for p in b]
