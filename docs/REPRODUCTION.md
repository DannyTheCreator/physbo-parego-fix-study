# Reproduce the two VLMOP2 experiments

Use Python 3.11 and the pinned dependencies. No material dataset download is needed.

~~~bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_upgrade.py --verify-only
python scripts/check_publication.py
~~~

Verification audits 120 retained trajectories / 6,000 evaluations against analytic
VLMOP2, checking unique queries, budgets, initial IDs, rewards, scalarization
weights and clocks. It reconstructs all current JSON/CSV statistics: 18 summary
rows, one controlled paired comparison and 18 target-sensitivity rows. It fits no
model and collects no timing observations.

It also audits the actual six historical repeat files against seed 25 and the
two historical six-run smoke evidence sets against each other. Counts and
metadata success flags alone are insufficient: missing, extra or corrupted
records cause verification to fail.

## New interface check or full campaign

~~~bash
python scripts/run_upgrade.py --smoke-only --output .cache/two-study-smoke
python scripts/run_upgrade.py --full --output .cache/two-study-full
~~~

Smoke executes six small trajectories; a full campaign executes 120. Both contain
only the controlled-operator and native-strategy blocks. Choose a fresh directory
after changes to code, protocol, dependencies or machine. Original observations
are not overwritten or resumed with the revised collector.

## Rebuild current statistics and presentation

~~~bash
python src/upgrade_report.py --results results/upgrade --figure-dir figures/upgrade
python scripts/build_upgrade_demo.py
python -m pip install -r requirements-report.txt
python scripts/build_upgrade_report.py
~~~

The notebook executes four evidence-reading cells without optimization. The report
command rebuilds [Markdown](REPORT.md) and the [PDF brief](../output/pdf/physbo_upgrade_report.pdf).
The two quality figures are controlled operator behavior and native strategy quality.

## Repeat evidence and original snapshots

`python scripts/verify_upgrade_repro.py` repeats prespecified seed 25 in six
conditions into a new `.cache/repeat-seed25` directory and compares selected IDs, rewards,
initialization and weights. Timing equality is not required. Existing retained
repeat records in `evidence/repeats` and the two six-record smoke evidence sets
come from historical execution, not from the present documentation change.
The repeat script writes its new comparison inside the selected output directory;
it never replaces the original historical evidence metadata.

After a full fresh run, compare all observations without comparing clocks:

~~~bash
python scripts/verify_replay.py --replay .cache/two-study-full --summary .cache/two-study-full/comparison.json
~~~

`results/upgrade/source_snapshot` preserves the original acquisition collector;
it may describe the larger former scope. The current active manifest, results
and code implement only the two VLMOP2 experiments. Published version metadata
and earlier reviews refer to historical scope. This revision has no new tagged
release yet; [validation](VALIDATION.md) records its final checks and review limits.
