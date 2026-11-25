# KV cache: MLA against grouped-query attention

Fidelity floor: `exact_match >= 95%` and `rel_dppl <= 1%`.

## The headline

| arm | rank | bytes/token | requests served | ideal | measured/ideal | decode tok/s | per-seq tok/s | prefill tok/s | ttft p50 ms | tpot p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| gqa | - | 28,672 | 4 | 4 | 1.00 | 32.0 | 16.1 | 3,712.0 | 1,959.8 | 65.3 |
| mla | 128 | 7,168 | 16 | 18 | 0.89 | 12.1 | 1.8 | 1,399.3 | 24,190.9 | 582.6 |
| mla | 256 | 14,336 | 8 | 9 | 0.89 | 9.9 | 2.0 | 1,152.4 | 9,314.9 | 526.5 |
| mla | 512 | 28,672 | 4 | 4 | 1.00 | 6.8 | 5.1 | 793.5 | 12,215.9 | 203.9 |

The requests-served number is the largest rung of the ladder that fits, and the ladder is powers of two: 1, 2, 4, 8, 16, 32, 64, 128. It is therefore a lower bound, and the true capacity lies between it and the rung above -- which is why a measured of 4 against an ideal of 6 is the engine holding 6 and the ladder having no rung for it.

## The trade curve

Rank against fidelity, with the floor drawn. A rank that misses the floor is not a candidate.

| arm | rank | exact_match | rel_dppl | requests served | ideal | measured/ideal | clears the floor |
|---|---|---|---|---|---|---|---|
| gqa | - | 100.0% | 0.0% | 4 | 4 | 1.00 | yes |
| mla | 128 | 0.0% | 20338947.0% | 16 | 18 | 0.89 | NO |
| mla | 256 | 0.0% | 650635.6% | 8 | 9 | 0.89 | NO |
| mla | 512 | 0.0% | 121771.2% | 4 | 4 | 1.00 | NO |

## The fidelity detail

| arm | rank | exact_match | token agreement | rel_dppl | rel_err_k | rel_err_v |
|---|---|---|---|---|---|---|
| gqa | - | 100.0% | 100.0% | 0.0% | - | - |
| mla | 128 | 0.0% | 1.8% | 20338947.0% | 85.4% | 92.1% |
| mla | 256 | 0.0% | 0.8% | 650635.6% | 74.3% | 71.4% |
| mla | 512 | 0.0% | 3.1% | 121771.2% | 0.0% | 0.0% |

Fidelity is a proxy for output quality, not a task-quality guarantee: it does not measure factuality, helpfulness or task success.
