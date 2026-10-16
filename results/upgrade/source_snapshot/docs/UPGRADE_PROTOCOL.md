# Historical protocol: English rendering

This is an English translation and reorganization of the protocol used for the
2026-10-15 observations. Administrative development notes have been omitted.
The experiment settings and results are unchanged. The original protocol is
retained in [v1.0.0](https://github.com/DannyTheCreator/physbo-budget-benchmark/blob/v1.0.0/results/upgrade/source_snapshot/docs/UPGRADE_PROTOCOL.md)
(Git blob 8b323eef87e15cc8ca0f7be1d9dc992f28f8175f). The three collector files in this snapshot
are unchanged; this Markdown translation is not claimed to preserve original bytes.


The confirmation study uses PHYSBO 3.2.1 with fixed candidate pools, budgets,
targets and seeds. Earlier exploratory runs used seeds 0-9; the published
confirmation results use seeds 10-29.

## Tasks and budgets

| Experiment | Design | Runs |
|---|---|---:|
| ParEGO operator diagnostic | 20 seeds, current `min` and diagnostic `max` | 40 |
| Native multiobjective search | 20 seeds, Random / HVPI / ParEGO / NDS | 80 |
| Copper initialization | Two pools, 20 seeds, random / maximin, Random / TS / EI | 240 |

Every run includes ten initial evaluations. VLMOP2 uses a fixed 101 x 101 grid
of 10,201 candidates, 50 total evaluations and an exact Gaussian process.
Copper uses two input-only random pools of 2,000 candidates from the same
17,982-row table, pool seeds 20261015 and 20261016, 100 total evaluations and
128 random features. Models are retrained every ten probes.

Policies receive standardized candidate inputs and only the rewards returned
by charged queries. Full-pool labels, optima, IQR and full-grid hypervolume are
used only by the oracle or after search for evaluation.

## ParEGO operator diagnostic

The official 3.2.1 correction changes the weighted Chebyshev reduction from
`max` to `min` for PHYSBO's maximization convention. It is the maintainers' fix.
The diagnostic runs both reductions inside 3.2.1, keeping initialization,
normalization, model settings and the weight stream identical. It does not
reproduce every part of PHYSBO 3.2.0.

After scaling observed objectives to [-1, 0], the current scalar score is:

$$
u = \min_j(w_j\tilde y_j) + 0.05\sum_j w_j\tilde y_j.
$$

For weights (0.5, 0.5), three already normalized objective vectors give:

| Point | Objectives | Current `min` score | Diagnostic `max` score |
|---|---|---:|---:|
| A | (-1, 0) | -0.525 | -0.025 |
| B | (0, -1) | -0.525 | -0.025 |
| C | (-0.4, -0.4) | -0.220 | -0.220 |

The ranking of observed points changes. Subsequent BO decisions still depend
on a fitted surrogate and expected improvement; unobserved rewards are unknown.

The two diagnostic operators share an independent weight generator seeded
with `SeedSequence([seed, 991])`. Each call normalizes two uniform random
numbers. Call index, observation count and weights are logged and checked.
Native ParEGO uses the library's own weight mechanism and is analyzed separately.

## Initialization control

Initial points use standardized inputs only. Maximin starts at a seed-controlled
random index, then repeatedly selects the unchosen candidate with the largest
minimum squared Euclidean distance to the selected set. Index order breaks
ties. The input-covering radius is calculated outside timed optimization.

Random, TS and EI share the same ordered initial points within a condition.
Both initialization conditions use a separate Random continuation ordering
from `SeedSequence([seed, 7341])`, with initial points filtered out. This
separates initial-point effects from subsequent strategy effects.

The primary initialization contrast is maximin minus random-initialization gap
at 20 evaluations. The paired change in each BO strategy's advantage over its
own Random control is also reported. The two pools are not independent materials.

## Quality targets

- Copper gap: `(best observed energy - pool minimum) / pool IQR`; lower is better.
  Success is finding an energy at or below the pool's `ceil(0.01 * N)`-th
  smallest energy, including ties.
- VLMOP2: hypervolume with fixed lower reference (-1, -1). The primary target
  is 90% of full-grid hypervolume; 80% and 95% are sensitivity checks.

All prespecified runs, including unsuccessful runs, remain in the analysis.

## Timing and evaluation costs

Runs are serial, with loaded BLAS libraries limited to one thread. Method order
rotates across seeds; the two- and four-method experiments balance positions
over 20 seeds. Imports are warmed up before measurement.

Cumulative decision overhead includes initial selection, policy setup, model
fitting, updates and selection. It excludes imports, shared input standardization,
oracle lookup/log construction, offline metrics, plotting and file I/O.
Oracle/logging time is measured separately and subtracted. Final model cleanup
is retained separately and excluded from the result-availability boundary used
in time-to-target analysis. These measurements do not separate fitting from
acquisition-function computation.

The ten initial observations return as one batch, so time analysis begins at
evaluation ten. For a fixed assumed serial evaluation cost:

$$
T(n;c) = H(n) + nc.
$$

Costs are 0, 0.001, 0.01, 0.1, 1, 10, 60 and 600 seconds per evaluation. All
objectives return together. No artificial sleep or physical relaxation timing
is used. Search remains capped at 50/100 evaluations; more time does not imply
additional queries. Non-hits remain censored and in success-rate denominators.

## Statistics and strategy maps

Bootstrap intervals resample seeds, with paired resampling for paired
comparisons: 10,000 samples and statistical seed 20261015. Pointwise intervals
are descriptive and do not imply simultaneous discovery across every setting.

A strategy-map cell is colored only if at least half the seeds reach the target
and the strategy's paired success-rate difference has a positive lower bound
against every alternative. Comparisons use per-cell Bonferroni-adjusted
percentile intervals. Otherwise the cell is labeled insufficient evidence.
This exploratory map does not provide simultaneous inference over the entire grid.

## Saved results

The collector saves a manifest, environment, source snapshot, per-run state and
all trajectories. It audits saved runs before reuse and rejects a changed
manifest, scientific dependency versions or source bytes. A changed collector
or protocol requires a new output directory. Historical collector files and
an English rendering of the original protocol are retained in the published
source snapshot. See the [reproduction guide](../../../../docs/REPRODUCTION.md).
