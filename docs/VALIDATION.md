# Final two-experiment validation

This record concerns the current ParEGO correction-effect and fixed-budget
VLMOP2 study. Original reported observations are the retained 120 trajectories /
6,000 evaluations. New replay checks are separate evidence and do not replace
the original decision-time measurements.

## Installation and runtime

On 16 October 2026 a new Python 3.11 virtual environment, with system site
packages disabled, installed all pinned experiment and optional PDF-report
dependencies from official PyPI. `pip check` reported no broken requirements.
The existing historical experiment environment was not used as proof of clean
installation. Local execution is Windows CPU; Linux execution is assigned to
the repository CI and is only confirmed when that workflow succeeds remotely.

## Deterministic evidence and replay

The frozen-result verifier reconstructs raw HV and target attainment from all
120 trajectories, validates the one paired correction effect, all 18 summary
rows and 18 sensitivity rows, and compares every JSON/CSV statistic. It audits
actual historical repeat files and isolated smoke pairs, not only metadata
success flags. Failure cases are covered by regression tests.

Final clean-checkout, full-replay and test counts are recorded in
[`evidence/release-validation.json`](../evidence/release-validation.json).
The current and exact staged-tree export each passed **57 tests**. A complete
fresh replay matched all **120 trajectories / 6,000 evaluations** in job identity,
initialization, candidate sequence, rewards and weights. The clean export also
passed a six-trajectory smoke pipeline, frozen-evidence verification, public-file
checks, four executed notebook cells and PDF rebuilding. New replay files and
raw command logs are retained locally; the repository contains the comparison
summary and commands required to repeat the check. Original reported clocks
were not replaced by replay clocks.
To repeat the main checks:

~~~bash
python -m pytest -q
python scripts/run_upgrade.py --verify-only
python scripts/run_upgrade.py --smoke-only --output .cache/my-release-smoke
python scripts/check_publication.py
~~~

A full rerun uses `--full` and a fresh output path. Compare it with
`scripts/verify_replay.py`; the comparison includes job definitions,
initialization, source-row identities, all selected candidates, rewards and
scalarization weights. It intentionally excludes clocks and timestamps.

## Semantic review status

A final-scope fresh-context GPT-6-Astra ultra audit was attempted. The reviewer
hit a usage limit before returning an overall verdict. It is **not a PASS**.
The partial finding about historical-evidence verification was fixed and tested;
citation metadata was also updated. There was no completed semantic recheck.
See the [audit record](../evidence/review/EXPERIMENT_AUDIT.md).

Deterministic validation and full trajectory agreement establish the reported
software checks within the recorded scope. They do not certify scientific
correctness, cross-platform trajectory identity or general strategy superiority.
