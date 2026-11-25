# KV cache: MLA against grouped-query attention

Fidelity floor: `exact_match >= 95%` and `rel_dppl <= 1%`.

## The headline

| arm | rank | bytes/token | requests served | ideal | measured/ideal | decode tok/s | per-seq tok/s | prefill tok/s | ttft p50 ms | tpot p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| gqa | - | 28,672 | 4 | 4 | 1.00 | 30.8 | 14.6 | 3,574.0 | 2,013.4 | 71.2 |
| mla | 64 | 3,584 | 32 | 36 | 0.89 | 12.0 | 1.8 | 1,388.3 | 24,047.9 | 586.0 |
| mla | 96 | 5,376 | 16 | 24 | 0.67 | 11.8 | 1.7 | 1,365.1 | 24,811.1 | 596.5 |
| mla | 128 | 7,168 | 16 | 18 | 0.89 | 11.4 | 1.7 | 1,328.1 | 25,525.6 | 612.0 |
| mla | 192 | 10,752 | 8 | 12 | 0.67 | 9.8 | 1.9 | 1,135.4 | 9,273.6 | 534.5 |
| mla | 256 | 14,336 | 8 | 9 | 0.89 | 9.1 | 1.7 | 1,056.0 | 9,860.1 | 601.7 |
| mla | 384 | 21,504 | 4 | 6 | 0.67 | 7.6 | 5.5 | 877.6 | 11,141.7 | 189.2 |
| mla | 512 | 28,672 | 4 | 4 | 1.00 | 6.6 | 4.9 | 761.4 | 12,805.9 | 211.6 |

The requests-served number is the largest rung of the ladder that fits, and the ladder is powers of two: 1, 2, 4, 8, 16, 32, 64, 128. It is therefore a lower bound, and the true capacity lies between it and the rung above -- which is why a measured of 4 against an ideal of 6 is the engine holding 6 and the ladder having no rung for it.

## The trade curve

Rank against fidelity, with the floor drawn. A rank that misses the floor is not a candidate.

| arm | rank | exact_match | rel_dppl | requests served | ideal | measured/ideal | clears the floor |
|---|---|---|---|---|---|---|---|
| gqa | - | 100.0% | 0.0% | 4 | 4 | 1.00 | yes |
| mla | 64 | 0.0% | 1191430.7% | 32 | 36 | 0.89 | NO |
| mla | 96 | 0.0% | 16760391.1% | 16 | 24 | 0.67 | NO |
| mla | 128 | 0.0% | 20338947.0% | 16 | 18 | 0.89 | NO |
| mla | 192 | 0.0% | 529848.9% | 8 | 12 | 0.67 | NO |
| mla | 256 | 0.0% | 650635.6% | 8 | 9 | 0.89 | NO |
| mla | 384 | 0.0% | 105736.4% | 4 | 6 | 0.67 | NO |
| mla | 512 | 0.0% | 121771.2% | 4 | 4 | 1.00 | NO |

## The fidelity detail

| arm | rank | exact_match | token agreement | rel_dppl | rel_err_k | rel_err_v |
|---|---|---|---|---|---|---|
| gqa | - | 100.0% | 100.0% | 0.0% | - | - |
| mla | 64 | 0.0% | 0.0% | 1191430.7% | 90.2% | 96.9% |
| mla | 96 | 0.0% | 0.0% | 16760391.1% | 87.6% | 95.1% |
| mla | 128 | 0.0% | 2.0% | 20338947.0% | 85.4% | 92.1% |
| mla | 192 | 0.0% | 1.8% | 529848.9% | 80.8% | 82.5% |
| mla | 256 | 0.0% | 0.9% | 650635.6% | 74.3% | 71.4% |
| mla | 384 | 0.0% | 3.0% | 105736.4% | 44.3% | 42.3% |
| mla | 512 | 0.0% | 3.1% | 121771.2% | 0.0% | 0.0% |

Fidelity is a proxy for output quality, not a task-quality guarantee: it does not measure factuality, helpfulness or task success.
