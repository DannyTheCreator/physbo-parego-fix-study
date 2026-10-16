# ParEGO Implementation Fix and Fixed-Budget Performance

**Research question:** How does the PHYSBO ParEGO implementation fix affect optimization performance, and how does corrected ParEGO compare with HVPI, NDS and Random under a fixed evaluation budget?

*Local scope revision: 16 October 2026. Two VLMOP2 experiments.*

## Shared experimental setup
Both experiments use PHYSBO 3.2.1 and the fixed VLMOP2 two-objective function on a 101 x 101 grid over [-2,2]^2, with reference (-1,-1). Each trajectory has 50 total evaluations including ten shared random initial candidates. Seeds are 10-29; exact GP is used and retrained every ten probes. Hypervolume (HV) is the dominated trade-off area relative to the fixed reference; larger is better. Full-grid HV is approximately 0.334518. The 90% target is approximately 0.301066. The full grid is evaluated only for offline reporting.

## Experiment 1 - Effect of the ParEGO implementation fix
Experiment 1 changes only the weighted Chebyshev reduction inside PHYSBO 3.2.1: diagnostic max versus corrected min. Initial points, normalization, GP settings and an independently logged weight stream are paired. For weights (0.5,0.5), extreme points (-1,0) and (0,-1) score -0.525 under min and -0.025 under max; balanced point (-0.4,-0.4) scores -0.220 under both. The ranking changes, followed by different search trajectories.

| Controlled operator | Final raw HV mean [95% interval] |
| --- | --- |
| ParEGO-max | 0.1588 [0.1341, 0.1830] |
| ParEGO-min | 0.2762 [0.2629, 0.2876] |

At 50 evaluations, the paired raw-HV difference (min minus max) is 0.1174 [0.0891, 0.1469]. The corrected reduction improves mean final quality in this controlled experiment. The official fix is upstream PHYSBO work. This is an operator-only diagnostic within 3.2.1, not an end-to-end comparison of complete software releases.

![Controlled ParEGO operator comparison](../figures/upgrade/version_comparison.png)

## Experiment 2 - Corrected ParEGO under a fixed evaluation budget
Experiment 2 uses the corrected native library strategies. Native ParEGO has mean final raw HV 0.2713, compared with 0.3270 for HVPI, 0.2516 for NDS and 0.1922 for Random. The 90% target is reached in 4/20, 20/20, 6/20 and 0/20 runs, respectively. Thus the fix improves controlled ParEGO behavior, while native ParEGO still has lower observed mean final HV than HVPI at the same evaluation cap. ParEGO has a higher observed mean than NDS and Random; no new pairwise significance test is claimed.

| Strategy | Final raw HV mean [95% interval] | 90% target hits | Decision seconds |
| --- | --- | --- | --- |
| ParEGO | 0.2713 [0.2472, 0.2910] | 4/20 | 2.718 |
| HVPI | 0.3270 [0.3260, 0.3278] | 20/20 | 6.948 |
| NDS | 0.2516 [0.2195, 0.2804] | 6/20 | 2.650 |
| Random | 0.1922 [0.1795, 0.2043] | 0/20 | <0.001 |

The corrected diagnostic ParEGO-min mean in Experiment 1 is 0.2762, whereas native ParEGO in Experiment 2 is 0.2713. These are separate conditions with different weight mechanisms: the diagnostic uses an independent logged paired weight stream; native ParEGO uses the library mechanism. Their records must not be pooled or treated as the same branch.

![Native strategies at fixed query counts](../figures/upgrade/multi_quality.png)

### Existing query-budget summaries: mean raw HV
| Queries | ParEGO | HVPI | NDS | Random |
| --- | --- | --- | --- | --- |
| 10 | 0.0795 | 0.0795 | 0.0795 | 0.0795 |
| 20 | 0.1455 | 0.2555 | 0.1456 | 0.1290 |
| 50 | 0.2713 | 0.3270 | 0.2516 | 0.1922 |

### Existing target sensitivity: successes out of 20 at 50 evaluations
| Grid-HV target | ParEGO | HVPI | NDS | Random |
| --- | --- | --- | --- | --- |
| 80% | 15/20 | 20/20 | 13/20 | 0/20 |
| 90% | 4/20 | 20/20 | 6/20 | 0/20 |
| 95% | 0/20 | 20/20 | 0/20 | 0/20 |

## Interpretation and limits
The conclusions concern this VLMOP2 grid, 20 seeds, fixed model settings and evaluation cap. The existing 10/20/50-query summaries and 80%/90%/95% targets describe budget and target dependence within the same runs. Pointwise 95% quality intervals use 10,000 percentile seed-bootstrap draws with seed 20261015. Success denominators retain all runs. This evidence does not establish a universal ranking, a win in every seed, or physical experimental performance. Algorithm time is a separate local computation measurement.

## Provenance and reproduction
The active dataset is a documented subset of the earlier acquisition: 120 retained trajectories / 6,000 charged evaluations, comprising 40 controlled-operator and 80 native-strategy trajectories. Original retained per-run bytes and timestamps are unchanged. Scope metadata and derived tables were updated without rerunning these experiments. Original collector snapshots and historical validation are retained as provenance, not relabeled as a new acquisition or review. The official algorithms and fix belong to PHYSBO maintainers; project implementation and reporting were AI-assisted.

Verify with `python scripts/run_upgrade.py --verify-only`. For a new run use a fresh output directory. See [protocol](PROTOCOL.md), [reproduction](REPRODUCTION.md), `results/upgrade/statistics.json`, `summary.csv`, `paired_comparisons.csv` and `target_sensitivity.csv`.

## Sources
- [Official PHYSBO 3.2.1 correction](https://github.com/issp-center-dev/PHYSBO/releases/tag/v3.2.1)
- [PHYSBO update paper, arXiv v1](https://arxiv.org/html/2603.01349v1)
