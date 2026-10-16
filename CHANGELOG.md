# Change log

## Unreleased - ParEGO fix and fixed-budget VLMOP2 study

Study the controlled effect of the official ParEGO reduction correction and
compare corrected native ParEGO with HVPI, NDS and Random at 50 evaluations.
Retain 120 original VLMOP2 trajectories / 6,000 evaluations with explicit
scope-projection provenance; keep the paired diagnostic and native weight
mechanisms separate. Remove the former material and initialization tasks,
simplified experiment and time-budget analysis from the active study.

Rebuild the report, two figures, PDF, evidence notebook, protocol and CI for
this scope. Verify actual historical repeat and isolation files as well as
all retained numerical exports. New repeat commands write separate outputs.
Update citation metadata to the current title; use commit-specific citation
until a new version is released. See docs/VALIDATION.md for current checks.

## 1.0.3 - 2026-10-15

Recompute and verify every report JSON statistic and all CSV exports by
unique keys, with strict numeric tolerance for cross-platform grid rounding.
Isolate default figure outputs inside each result directory
and add an explicit `--figure-dir` for publication rebuilds. Use exact
binomial success intervals and two-sided exact paired McNemar tests with
the existing per-cell Bonferroni threshold. Retain legacy bootstrap map
decisions as sensitivity evidence: seven cells lose support (five Random,
two EI), while the 170 HVPI cells remain. Reject resume after timing
environment changes, including CPU/Python/OS/BLAS/thread limits.
Rebuild derived tables, figures and reports from frozen trajectories;
the original 360 runs, timings, metadata and collector snapshot are preserved.

## 1.0.2 - 2026-10-15

Clarify all five strategy evidence maps: separate discrete cost rows, label
supported methods directly, show only present colors in the legend, and name
methods with no supported cells. Correct the half-cell time-coordinate offset
by centering tiles on the tested logarithmic budgets. The VLMOP2 map correctly
has only HVPI-colored cells (170), alongside 342 insufficient-evidence cells.
Rebuild the report, PDF and executed notebook; verify all 2,560 map decisions
from the original trajectories. Scientific results and timings are unchanged.

## 1.0.1 - 2026-10-15

Use English throughout the documentation and executed demo. Consolidate the
project around the report, protocol, reproduction guide and data description;
remove internal review transcripts and duplicated publication notes. Original
experimental observations and repeat evidence are unchanged.

## 1.0.0 — 2026-10-15

First public research snapshot: 360 confirmation trajectories, version-aware
ParEGO diagnostic, VLMOP2 strategy comparison, two-pool copper initialization
controls, cumulative timing and cost-scenario maps. Includes the English report,
executed notebook, raw evidence, and qualified integrity review.

Publication fixes prepare data before tests, add separate smoke/verification/
full-run modes, include review and repeat evidence inside the repository, and
add Windows/Linux CI. The original measured experiments are preserved.

