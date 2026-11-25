"""The CLI, the fidelity check and the report.

The fidelity check is the project's correctness gate: at r=512 the latent
holds the same information as the native K/V, so the MLA arm must reproduce
the GQA arm token for token. If it does not, the construction or the kernel
is wrong, and no other rank may be run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from vllm import SamplingParams

from . import config, server

GREEDY_MAX_TOKENS = 256

# The fidelity floor, declared before the sweep runs (PLAN.md section 4.4).
FIDELITY_FLOOR = {"exact_match": 0.95, "rel_dppl": 0.01}

# The SVD's energy retention, per rank. It costs a full SVD of every layer,
# so it is computed once per rank and reused.
_ENERGY_CACHE: dict[int, dict[str, float]] = {}


def corrode(path=None) -> str:
    """The corpus, fetched once and checked against its checksum.

    The file is data, not code: it is only ever read as text, and the
    checksum is what makes it trustworthy.
    """
    path = Path(path or config.CORPUS_PATH)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(config.CORPUS_URL) as response:
            path.write_bytes(response.read())

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config.CORPUS_SHA256:
        raise ValueError(f"{path} has sha256 {digest}, expected {config.CORPUS_SHA256}")
    return raw.decode("utf-8")


def corpus_prompts(text: str, tokenizer, tokens_per_prompt: int, count: int) -> list[str]:
    """The corpus split into fixed-length prompts."""
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    chunks = [
        tokenizer.decode(ids[i : i + tokens_per_prompt])
        for i in range(0, len(ids) - tokens_per_prompt + 1, tokens_per_prompt)
    ]
    return chunks[:count]


def first_divergence(a, b) -> int | None:
    """Where two token sequences stop agreeing, or None if they agree."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def token_agreement(a, b) -> float:
    """The fraction of positions where the two sequences agree.

    A token the other sequence never produced counts as a disagreement.
    """
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    same = sum(
        1 for i in range(n) if i < len(a) and i < len(b) and a[i] == b[i]
    )
    return same / n


@dataclass
class Row:
    """One measured (arm, rank, context length, repeat)."""

    arm: str
    rank: int | None
    context_len: int
    repeat: int

    kv_bytes_per_token: int
    kv_budget_bytes: int
    blocks: int

    requests_served: int
    """How many concurrent requests the engine served without preempting."""

    tokens_per_request: int
    """What one request actually cost: its prompt plus its output."""

    prefill_tok_per_s: float
    decode_tok_per_s: float
    decode_tok_per_s_per_seq: float
    ttft_p50: float
    ttft_p95: float
    tpot_p50: float
    tpot_p95: float
    e2e_p50: float
    e2e_p95: float

    exact_match: float | None
    token_agreement: float | None
    rel_dppl: float | None
    rel_err_k: float | None
    rel_err_v: float | None

    @property
    def label(self) -> str:
        return f"{self.arm}@{self.rank}" if self.rank else self.arm

    @property
    def ideal_requests_served(self) -> int:
        """How many requests the budget holds, if nothing else is in the way.

        The plan reports this next to the measured number, so the gap
        between the arithmetic and the engine is visible rather than
        hidden. It is the budget divided by what a request actually costs,
        not by the nominal context length: a request that fills the context
        and a request that does not are not the same workload.
        """
        tokens = self.kv_budget_bytes // self.kv_bytes_per_token
        return tokens // max(self.tokens_per_request, 1)


def write_rows(rows, path: Path) -> None:
    """The rows as a CSV, one line each."""
    names = [f.name for f in fields(Row)]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_rows(path: Path) -> list[Row]:
    """The rows back from a CSV, with the types they had."""
    typed = {f.name: f.type for f in fields(Row)}
    rows = []
    with Path(path).open(newline="") as handle:
        for record in csv.DictReader(handle):
            values = {}
            for name, value in record.items():
                if value == "":
                    values[name] = None
                    continue
                kind = typed[name]
                if "int" in kind:
                    values[name] = int(value)
                elif "float" in kind:
                    values[name] = float(value)
                else:
                    values[name] = value
            rows.append(Row(**values))
    return rows


def _pct(value) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _num(value, digits=1) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def render_report(rows, out_dir) -> Path:
    """The report: three tables, written next to the rows they came from.

    The headline is what the arms cost and buy. The trade curve is the
    choice: rank against fidelity, with the floor drawn. The fidelity
    detail is the evidence for the floor.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_rows(rows, out_dir / "rows.csv")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "fidelity_floor": FIDELITY_FLOOR,
                "arms": sorted({r.arm for r in rows}),
                "ranks": sorted({r.rank for r in rows if r.rank}),
                "context_lens": sorted({r.context_len for r in rows}),
                "rows": len(rows),
            },
            indent=2,
        )
    )

    lines = [
        "# KV cache: MLA against grouped-query attention",
        "",
        f"Fidelity floor: `exact_match >= {FIDELITY_FLOOR['exact_match'] * 100:.0f}%` "
        f"and `rel_dppl <= {FIDELITY_FLOOR['rel_dppl'] * 100:.0f}%`.",
        "",
        "## The headline",
        "",
        "| arm | rank | bytes/token | requests served | ideal | measured/ideal | "
        "decode tok/s | per-seq tok/s | prefill tok/s | ttft p50 ms | tpot p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r.arm, r.rank or 0, r.context_len)):
        ideal = row.ideal_requests_served
        ratio = row.requests_served / ideal if ideal else 0.0
        lines.append(
            f"| {row.arm} | {row.rank or '-'} | {row.kv_bytes_per_token:,} | "
            f"{row.requests_served} | {ideal} | {ratio:.2f} | "
            f"{_num(row.decode_tok_per_s)} | {_num(row.decode_tok_per_s_per_seq)} | "
            f"{_num(row.prefill_tok_per_s)} | "
            f"{_num(row.ttft_p50)} | {_num(row.tpot_p50)} |"
        )

    lines += [
        "",
        "The requests-served number is the largest rung of the ladder that "
        "fits, and the ladder is powers of two: 1, 2, 4, 8, 16, 32, 64, 128. "
        "It is therefore a lower bound, and the true capacity lies between "
        "it and the rung above -- which is why a measured of 4 against an "
        "ideal of 6 is the engine holding 6 and the ladder having no rung "
        "for it.",
        "",
        "## The trade curve",
        "",
        "Rank against fidelity, with the floor drawn. A rank that misses the "
        "floor is not a candidate.",
        "",
        "| arm | rank | exact_match | rel_dppl | requests served | ideal | "
        "measured/ideal | clears the floor |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r.arm, r.rank or 0, r.context_len)):
        clears = (
            "-"
            if row.exact_match is None
            else (
                "yes"
                if row.exact_match >= FIDELITY_FLOOR["exact_match"]
                and abs(row.rel_dppl) <= FIDELITY_FLOOR["rel_dppl"]
                else "NO"
            )
        )
        ideal = row.ideal_requests_served
        ratio = row.requests_served / ideal if ideal else 0.0
        lines.append(
            f"| {row.arm} | {row.rank or '-'} | {_pct(row.exact_match)} | "
            f"{_pct(row.rel_dppl)} | {row.requests_served} | {ideal} | "
            f"{ratio:.2f} | {clears} |"
        )

    lines += [
        "",
        "## The fidelity detail",
        "",
        "| arm | rank | exact_match | token agreement | rel_dppl | "
        "rel_err_k | rel_err_v |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r.arm, r.rank or 0, r.context_len)):
        lines.append(
            f"| {row.arm} | {row.rank or '-'} | {_pct(row.exact_match)} | "
            f"{_pct(row.token_agreement)} | {_pct(row.rel_dppl)} | "
            f"{_pct(row.rel_err_k)} | {_pct(row.rel_err_v)} |"
        )

    lines += [
        "",
        "Fidelity is a proxy for output quality, not a task-quality "
        "guarantee: it does not measure factuality, helpfulness or task "
        "success.",
        "",
    ]

    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


@dataclass
class FidelityResult:
    """How close the arm came to the reference."""

    exact_match: float
    """The fraction of responses that matched token for token."""

    token_agreement: float
    """The fraction of the corpus's tokens that matched."""

    first_divergence: list[int | None]
    """Where each response first diverged from the reference."""

    divergence_fraction: list[float]
    """That position, as a fraction of the reference's length."""

    ppl: float
    """This arm's perplexity over the corpus."""

    rel_dppl: float
    """Its perplexity against the reference's, as a fraction."""


async def _perplexity(engine, prompts) -> float:
    """The corpus's perplexity, from the prompt logprobs."""
    params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    total = 0.0
    count = 0
    for index, prompt in enumerate(prompts):
        async for out in engine.generate(prompt, params, f"ppl-{index}"):
            for entry in out.prompt_logprobs or []:
                if entry is None:
                    continue
                for value in entry.values():
                    total += value.logprob
                    count += 1
    return math.exp(-total / max(count, 1))


async def _arm_outputs(engine, prompts) -> list[list[int]]:
    return await server.greedy_outputs(engine, prompts, GREEDY_MAX_TOKENS)


def _run_arm(cfg: server.EngineConfig, prompts) -> tuple[list[list[int]], float]:
    """One arm's greedy outputs and its perplexity over the same prompts."""
    import asyncio

    async def run():
        engine = server.build_engine(cfg)
        try:
            outputs = await _arm_outputs(engine, prompts)
            ppl = await _perplexity(engine, prompts)
        finally:
            engine.shutdown()
        return outputs, ppl

    return asyncio.run(run())


def fidelity(cfg: server.EngineConfig, prompts, reference=None) -> FidelityResult:
    """Compare one arm against the reference arm.

    `reference` is the reference arm's (greedy outputs, perplexity). If it is
    None the reference arm is run here, which costs a second engine.
    """
    outputs, ppl = _run_arm(cfg, prompts)

    if reference is None:
        ref_cfg = server.EngineConfig(
            arm="gqa", max_model_len=cfg.max_model_len, block_size=cfg.block_size
        )
        reference = _run_arm(ref_cfg, prompts)
    ref_outputs, ref_ppl = reference

    divergences = [first_divergence(r, o) for r, o in zip(ref_outputs, outputs)]
    fractions = [
        (d / len(r)) if d is not None and r else 0.0
        for d, r in zip(divergences, ref_outputs)
    ]
    exact = sum(1 for d in divergences if d is None) / max(len(divergences), 1)

    ref_tokens = sum(len(r) for r in ref_outputs)
    same = sum(
        sum(1 for i in range(min(len(r), len(o))) if r[i] == o[i])
        for r, o in zip(ref_outputs, outputs)
    )
    agreement = same / max(ref_tokens, 1)

    return FidelityResult(
        exact_match=exact,
        token_agreement=agreement,
        first_divergence=divergences,
        divergence_fraction=fractions,
        ppl=ppl,
        rel_dppl=(ppl - ref_ppl) / ref_ppl if ref_ppl else float("nan"),
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="kvbench")
    sub = parser.add_subparsers(dest="command", required=True)

    fid = sub.add_parser("fidelity", help="compare an arm against the gqa arm")
    fid.add_argument("--arm", default="mla")
    fid.add_argument("--rank", type=int, default=512)
    fid.add_argument(
        "--reference-arm",
        default="gqa",
        choices=["gqa", "gqa_norope"],
        help="The arm the fidelity is measured against.",
    )
    fid.add_argument("--corpus", default=None)
    fid.add_argument("--prompts", type=int, default=24)
    fid.add_argument("--prompt-tokens", type=int, default=1024)
    fid.add_argument("--max-model-len", type=int, default=4096)
    fid.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    fid.add_argument("--kv-budget-bytes", type=int, default=config.KV_BUDGET_BYTES)
    fid.add_argument("--out", default=None)

    load = sub.add_parser("load", help="run a load test")
    load.add_argument("--arm", default="mla")
    load.add_argument("--rank", type=int, default=512)
    load.add_argument("--requests", type=int, default=64)
    load.add_argument("--concurrency", type=int, default=8)
    load.add_argument("--prompt-tokens", type=int, default=1024)
    load.add_argument("--output-tokens", type=int, default=64)
    load.add_argument("--max-model-len", type=int, default=4096)
    load.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    load.add_argument("--kv-budget-bytes", type=int, default=config.KV_BUDGET_BYTES)

    prep = sub.add_parser("prepare", help="build the MLA weights for each rank")
    prep.add_argument("--ranks", default=",".join(str(r) for r in config.RANKS))

    comp = sub.add_parser("compare", help="run both arms and measure them")
    comp.add_argument("--arms", default="gqa,mla")
    comp.add_argument("--ranks", default="512")
    comp.add_argument("--context-lens", default="4096")
    comp.add_argument("--repeats", type=int, default=1)
    comp.add_argument("--requests", type=int, default=64)
    comp.add_argument("--output-tokens", type=int, default=64)
    comp.add_argument("--prompt-tokens", type=int, default=1024)
    comp.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    comp.add_argument("--kv-budget-bytes", type=int, default=config.KV_BUDGET_BYTES)
    comp.add_argument(
        "--fidelity",
        action="store_true",
        help="also measure the arms' fidelity against the gqa arm",
    )
    comp.add_argument("--fidelity-prompts", type=int, default=8)
    comp.add_argument("--corpus", default=None)
    comp.add_argument("--out", default=None)

    sweep = sub.add_parser("sweep", help="run the matrix in a YAML file")
    sweep.add_argument("--config", default="experiments.yaml")

    rep = sub.add_parser("report", help="re-render the report from a run")
    rep.add_argument("--run", required=True, help="the results directory")

    args = parser.parse_args(argv)
    if args.command == "fidelity":
        _cli_fidelity(args)
    elif args.command == "load":
        _cli_load(args)
    elif args.command == "prepare":
        _cli_prepare(args)
    elif args.command == "compare":
        _cli_compare(args)
    elif args.command == "sweep":
        _cli_sweep(args)
    else:
        _cli_report(args)


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(server.base_model_dir()))


def _cli_fidelity(args) -> None:
    text = corrode(args.corpus)
    prompts = corpus_prompts(
        text, _tokenizer(), args.prompt_tokens, args.prompts
    )
    cfg = server.EngineConfig(
        arm=args.arm,
        rank=args.rank if args.arm == "mla" else None,
        max_model_len=args.max_model_len,
        kv_budget_bytes=args.kv_budget_bytes,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
    )
    ref_cfg = server.EngineConfig(
        arm=args.reference_arm,
        max_model_len=args.max_model_len,
        kv_budget_bytes=args.kv_budget_bytes,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
    )
    reference = _run_arm(ref_cfg, prompts)
    result = fidelity(cfg, prompts, reference=reference)

    print(json.dumps({
        "arm": args.arm,
        "rank": args.rank if args.arm == "mla" else None,
        "prompts": len(prompts),
        "exact_match": result.exact_match,
        "token_agreement": result.token_agreement,
        "ppl": result.ppl,
        "ref_ppl": reference[1],
        "rel_dppl": result.rel_dppl,
        "first_divergence": result.first_divergence,
    }, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(result.__dict__, indent=2))


def _cli_load(args) -> None:
    import asyncio

    body = "The quick brown fox jumps over the lazy dog. " * 40
    prompts = [f"Request {i}. {body}" for i in range(args.requests)]

    async def run():
        cfg = server.EngineConfig(
            arm=args.arm,
            rank=args.rank if args.arm == "mla" else None,
            max_model_len=args.max_model_len,
            kv_budget_bytes=args.kv_budget_bytes,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        engine = server.build_engine(cfg)
        try:
            requests = server.make_requests(prompts, args.output_tokens)
            return await server.run_load(engine, requests, args.concurrency)
        finally:
            engine.shutdown()

    result = asyncio.run(run())
    print(json.dumps({k: v for k, v in result.__dict__.items()}, indent=2, default=list))


def _cli_prepare(args) -> None:
    """Build the MLA checkpoint for each rank. Offline and on the CPU."""
    for rank in (int(r) for r in args.ranks.split(",") if r):
        out = server.mla_model_dir(rank)
        print(f"r={rank}: {out}")


def _percentile(values, q: float) -> float:
    """A percentile of a sample, by linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _measure_arm(
    args, arm: str, rank, context_lens: list[int], repeats: int, reference=None
) -> tuple[list[Row], dict[int, tuple[list[list[int]], float]]]:
    """One arm at one rank, at every context length and repeat.

    The engine is started once and every measurement is taken inside it:
    a vLLM start costs minutes, and the sweep has 8 arms x 2 context
    lengths x 3 repeats, so a start per measurement would cost hours.
    The engine's own max_model_len is the longest context, so the shorter
    ones fit inside it.

    The reference is the gqa arm's fidelity at the same context length:
    a dict of context length to (greedy outputs, perplexity).
    """
    import asyncio

    cfg = server.EngineConfig(
        arm=arm,
        rank=rank,
        max_model_len=max(context_lens),
        kv_budget_bytes=args.kv_budget_bytes,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    per_token = (
        config.NUM_KV_HEADS * config.HEAD_DIM * 2 * 2
        if arm == "gqa"
        else rank * 2
    )
    energy = None if arm == "gqa" else _sved_energy(rank)
    tokenizer = _tokenizer()

    async def run():
        engine = server.build_engine(cfg)
        rows = []
        fidelity: dict[int, tuple[list[list[int]], float]] = {}
        try:
            for context_len in context_lens:
                prompt_tokens = max(context_len - args.output_tokens, 1)
                prompts = server.prompts_of_length(
                    tokenizer, args.requests, prompt_tokens
                )
                corpus = corpus_prompts(
                    corrode(args.corpus),
                    tokenizer,
                    max(context_len - GREEDY_MAX_TOKENS, 1),
                    args.fidelity_prompts,
                )
                # the arithmetic ceiling: the most the budget could hold
                blocks = server.blocks_for(cfg)
                ceiling = (blocks * cfg.block_size) // context_len

                for repeat in range(repeats):
                    served = await server.find_requests_served(
                        engine, prompts, args.output_tokens, ceiling=ceiling
                    )
                    requests = server.make_requests(prompts, args.output_tokens)
                    result = await server.run_load(
                        engine, requests, served or 1
                    )

                    outputs = ppl = None
                    if args.fidelity:
                        outputs = await _arm_outputs(engine, corpus)
                        ppl = await _perplexity(engine, corpus)
                        if repeat == 0:
                            fidelity[context_len] = (outputs, ppl)

                    rows.append(
                        _row(
                            arm=arm,
                            rank=rank,
                            context_len=context_len,
                            repeat=repeat,
                            per_token=per_token,
                            budget=args.kv_budget_bytes,
                            blocks=blocks,
                            served=served,
                            tokens_per_request=prompt_tokens
                            + args.output_tokens,
                            result=result,
                            outputs=outputs,
                            ppl=ppl,
                            reference=None
                            if not isinstance(reference, dict)
                            else reference.get(context_len),
                            energy=energy,
                        )
                    )
                    print(
                        f"{rows[-1].label} @{context_len} "
                        f"#{repeat}: {served} served",
                        flush=True,
                    )
        finally:
            # The engine's core runs in its own process. If the driver is
            # killed, that process survives and keeps the card's memory,
            # so it is killed by name here rather than left to be found.
            engine.shutdown()
            server.kill_orphan_engines()
        return rows, fidelity

    return asyncio.run(run())


def _row(
    arm,
    rank,
    context_len,
    repeat,
    per_token,
    budget,
    blocks,
    served,
    tokens_per_request,
    result,
    outputs,
    ppl,
    reference,
    energy,
) -> Row:
    """One measured row, with the fidelity against the reference arm."""
    exact = agreement = rel_dppl = None
    if outputs is not None:
        if reference is None:
            # the gqa arm is the reference: it has nothing to be compared to
            exact = 1.0
            agreement = 1.0
            rel_dppl = 0.0
        else:
            ref_outputs, ref_ppl = reference
            divergences = [
                first_divergence(r, o) for r, o in zip(ref_outputs, outputs)
            ]
            exact = sum(1 for d in divergences if d is None) / max(
                len(divergences), 1
            )
            ref_tokens = sum(len(r) for r in ref_outputs)
            same = sum(
                sum(1 for i in range(min(len(r), len(o))) if r[i] == o[i])
                for r, o in zip(ref_outputs, outputs)
            )
            agreement = same / max(ref_tokens, 1)
            rel_dppl = (ppl - ref_ppl) / ref_ppl if ref_ppl else float("nan")

    return Row(
        arm=arm,
        rank=rank,
        context_len=context_len,
        repeat=repeat,
        kv_bytes_per_token=per_token * config.NUM_LAYERS,
        kv_budget_bytes=budget,
        blocks=blocks,
        requests_served=served,
        tokens_per_request=tokens_per_request,
        prefill_tok_per_s=result.prefill_tok_per_s,
        decode_tok_per_s=result.decode_tok_per_s,
        decode_tok_per_s_per_seq=_percentile(result.decode_tok_per_s_per_seq, 0.5),
        ttft_p50=_percentile(result.ttft_ms, 0.5),
        ttft_p95=_percentile(result.ttft_ms, 0.95),
        tpot_p50=_percentile(result.tpot_ms, 0.5),
        tpot_p95=_percentile(result.tpot_ms, 0.95),
        e2e_p50=_percentile(result.e2e_ms, 0.5),
        e2e_p95=_percentile(result.e2e_ms, 0.95),
        exact_match=exact,
        token_agreement=agreement,
        rel_dppl=rel_dppl,
        rel_err_k=energy["k"] if energy else None,
        rel_err_v=energy["v"] if energy else None,
    )


def _sved_energy(rank: int) -> dict[str, float] | None:
    """How much of the key and value subspace the rank-r latent kept.

    This is the fidelity of the cache itself, independent of the model:
    it says how much energy the truncation threw away.
    """
    from . import weights

    if rank in _ENERGY_CACHE:
        return _ENERGY_CACHE[rank]

    base = server.base_model_dir()
    state = weights.load_model_state(base)
    errors = {"k": [], "v": []}
    for index in weights._layer_indices(state):
        prefix = f"model.layers.{index}.self_attn."
        layer = {
            "k_proj.weight": state[prefix + "k_proj.weight"].double(),
            "v_proj.weight": state[prefix + "v_proj.weight"].double(),
        }
        sub = weights.joint_svd(layer, rank)
        errors["k"].append(sub.rel_err_k.max().item())
        errors["v"].append(sub.rel_err_v.max().item())
    energy = {"k": max(errors["k"]), "v": max(errors["v"])}
    _ENERGY_CACHE[rank] = energy
    return energy


def _cli_compare(args) -> None:
    """Both arms, measured, into a results directory with a report."""
    out = Path(args.out or config.RESULTS_DIR / _timestamp())
    out.mkdir(parents=True, exist_ok=True)

    arms = [a for a in args.arms.split(",") if a]
    ranks = [int(r) for r in args.ranks.split(",") if r]
    context_lens = [int(c) for c in args.context_lens.split(",") if c]

    rows: list[Row] = []
    # the reference is the gqa arm's fidelity, per context length
    reference: dict[int, tuple[list[list[int]], float]] = {}
    for arm in arms:
        # the gqa arm has no rank; the mla arm has one per rank
        arm_ranks = ranks if arm == "mla" else [None]
        for rank in arm_ranks:
            measured, fidelity = _measure_arm(
                args, arm, rank, context_lens, args.repeats, reference
            )
            rows.extend(measured)
            if arm == "gqa":
                # the gqa arm's own greedy output and perplexity, per context
                reference.update(fidelity)

    render_report(rows, out)
    print(f"wrote {out}")


def _timestamp() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def _cli_sweep(args) -> None:
    """The matrix from a YAML file."""
    import yaml

    matrix = yaml.safe_load(Path(args.config).read_text())
    args.arms = ",".join(matrix["arms"])
    args.ranks = ",".join(str(r) for r in matrix["ranks"])
    args.context_lens = ",".join(str(c) for c in matrix["context_lens"])
    args.repeats = int(matrix["repeats"])
    args.out = str(config.RESULTS_DIR / _timestamp())
    args.fidelity = True
    # the flags the YAML does not name keep the CLI's defaults
    for name, value in (
        ("requests", 64),
        ("output_tokens", 64),
        ("prompt_tokens", 1024),
        ("gpu_memory_utilization", 0.8),
        ("kv_budget_bytes", config.KV_BUDGET_BYTES),
        ("fidelity_prompts", 8),
        ("corpus", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)
    # the YAML may name them
    for name in (
        "requests",
        "output_tokens",
        "gpu_memory_utilization",
        "kv_budget_bytes",
        "fidelity_prompts",
    ):
        if name in matrix:
            setattr(args, name, matrix[name])
    _cli_compare(args)


def _cli_report(args) -> None:
    """Re-render the report from a run's rows.csv. No GPU."""
    run = Path(args.run)
    rows = read_rows(run / "rows.csv")
    path = render_report(rows, run)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
