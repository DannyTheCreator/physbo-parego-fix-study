# VLMOP2 two-experiment protocol

## Research question

How does the PHYSBO ParEGO implementation fix affect optimization performance, and how does corrected ParEGO compare with HVPI, NDS and Random under a fixed evaluation budget?

This local scope revision retains two blocks of the original acquisition:
40 controlled ParEGO-operator trajectories and 80 native strategy trajectories.
The 120 retained runs contain 6,000 charged evaluations. Original per-run
observations, seeds, timings and thresholds remain unchanged. This is a documented
scope reduction, not a new acquisition or preregistration.

## Shared setup

- PHYSBO 3.2.1; VLMOP2 on [-2,2]^2; 101 points per axis, 10,201 candidates.
- Two negated minimization objectives for PHYSBO maximization.
- Fixed HV reference (-1,-1); full-grid HV approximately 0.334518.
- Seeds 10-29; ten identical ordered random initial candidates within each seed.
- 50 total evaluations per trajectory, including initialization; one new candidate per probe.
- Exact GP (`num_rand_basis=0`); retraining interval ten probes; no repeated queries.
- Standardization uses public candidate inputs only; unqueried objective values never enter selection.
- Full-grid labels, full-grid HV and target fractions are used only by offline evaluation.

## Experiment 1 - Effect of the ParEGO implementation fix

Run current min and diagnostic max inside 3.2.1. Initial candidates, normalization,
GP settings and weight stream are paired; only the Chebyshev reduction changes.
The official correction belongs to PHYSBO maintainers. This diagnostic does not
emulate an entire earlier release or certify the whole implementation correct.

The two operators share an independent generator `SeedSequence([seed, 991])`.
Each call normalizes two uniform random numbers; call index, observation count
and weights are logged and audited. The scalar score after observed-objective
scaling to [-1,0] is:

$$
u = \min_j(w_j\tilde y_j) + 0.05\sum_j w_j\tilde y_j.
$$

The diagnostic max condition changes only the minimum to a maximum. For weights
(0.5,0.5), A=(-1,0), B=(0,-1), C=(-0.4,-0.4), current min scores are
-0.525, -0.525, -0.220; diagnostic max scores are -0.025, -0.025, -0.220.
Primary outcome: paired final raw-HV difference, min minus max, at 50 evaluations.
There are 20 pairs / 40 trajectories.

## Experiment 2 - Corrected ParEGO under a fixed evaluation budget

Compare native ParEGO, HVPI, NDS and Random with the shared setup and 20 seeds
(80 trajectories). Native ParEGO uses library weights, `weight_min=1`,
`weight_sum=0.05`; NDS uses `rank_max=10`. HVPI uses the native multiobjective
policy. Random uses a separate continuation permutation from
`SeedSequence([seed, 7341])` with initial candidates filtered out.

Native ParEGO and Experiment 1's diagnostic min have different weight mechanisms;
their records are never pooled. The primary endpoint is raw HV at 50 evaluations.
Also report the existing 10/20/50-evaluation summaries and target attainment at
80%, 90% and 95% of fixed-grid HV. All runs, including non-hits, remain in denominators.

## Statistics and timing

Continuous quality and paired differences use percentile seed bootstrap:
10,000 draws with statistical seed 20261015, paired resampling where applicable.
Binary success intervals use pointwise 95% Clopper-Pearson exact intervals.
These are descriptive, pointwise summaries; no new simultaneous or pairwise
significance claim is added by the local revision.

Runs are serial with both loaded BLAS libraries limited to one thread. Method
positions rotate across seeds. Decision overhead includes initial selection,
policy construction, fitting, updating and selection; imports, shared input
preprocessing, oracle/log construction, offline metrics and output I/O are excluded.
Final cleanup is stored separately and excluded from the result-availability
clock in summaries. The ten initial observations are available together at query ten.
Algorithm time is separate from quality and is not converted to physical evaluation time.

## Provenance and resuming

The active manifest is the VLMOP2 subset of the earlier campaign; a scope-projection
note records the selection. Original collector snapshots are retained unmodified
as historical provenance. They are not the current runnable source.

New runs use the current source and protocol and must write to a fresh output
directory. Resume validates job records, collector bytes, package versions and
CPU/Python/OS/BLAS/thread identity. Legacy `pool_seed`/`pool_size` job keys are
retained solely for compatibility with original trace identity; VLMOP2 candidates
are always defined by `grid_points ** 2`, and these legacy keys select no data.

Historical repeat and isolated-smoke metadata record the six retained VLMOP2
conditions from the earlier checks; they are not presented as newly executed checks.
