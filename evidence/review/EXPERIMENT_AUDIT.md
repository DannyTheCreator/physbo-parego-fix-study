# Final-scope experiment audit attempt

**Date:** 16 October 2026. **Status: ERROR / no overall verdict.**

A fresh-context GPT-6-Astra ultra reviewer was asked to audit reference provenance,
normalization, actual result files, execution paths, claim scope and evaluation
type from primary code and evidence. This route is same-family and provisional,
not independent scientific replication. The reviewer hit a usage limit before
returning an overall judgment. Partial observations are not an audit PASS.

The partial review identified a concrete verification gap: the old verifier
checked historical repeat/isolation counts and success flags without comparing
the evidence files. `scripts/verify_results.py` now audits actual records, exact
condition membership, candidate sequences, rewards, initialization and weights.
Seven regression cases exercise valid evidence and corrupt/missing/extra records.
The reviewer also noted stale citation metadata, which has been updated.

These fixes and deterministic checks are executor work; there was no completed
fresh-reviewer recheck. See [validation](../../docs/VALIDATION.md) for executed
installation, trace, replay and clean-checkout checks, whose scope is separate
from semantic review. The analytic VLMOP2 objective supplies reference values;
it is a mathematical benchmark, not a model-generated reference or a physical test.
