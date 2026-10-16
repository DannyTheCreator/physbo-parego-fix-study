"""Audit saved trajectories and every derived report statistic without optimizer reruns."""
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from upgrade_report import (load_runs, measure, group_runs, summary_tables,
                            statistics)
from upgrade_study import audit


TABLE_KEYS = {
    'summary': ('block', 'pool_seed', 'initialization', 'method', 'evaluations', 'metric'),
    'paired_comparisons': ('comparison', 'pool_seed', 'method', 'evaluations'),
    'target_sensitivity': ('block', 'method', 'target'),
}


def compare_value(actual, expected, label, from_csv=False):
    """Compare finite numeric fields and exact structure, with CSV-aware scalar parsing."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'{label}: fields differ')
        for key in expected:
            compare_value(actual[key], expected[key], f'{label}.{key}', from_csv)
    elif expected is None:
        if actual != ('' if from_csv else None):
            raise ValueError(f'{label}: expected a missing value')
    elif isinstance(expected, bool):
        if (actual != str(expected) if from_csv else type(actual) is not bool or actual != expected):
            raise ValueError(f'{label}: boolean differs')
    elif isinstance(expected, (int, float)):
        if isinstance(actual, bool) or (not from_csv and not isinstance(actual, (int, float))):
            raise ValueError(f'{label}: expected a number')
        number = float(actual)
        if not math.isfinite(number):
            raise ValueError(f'{label}: nonfinite number')
        np.testing.assert_allclose(number, expected, rtol=1e-10, atol=1e-12, err_msg=label)
    elif actual != expected:
        raise ValueError(f'{label}: {actual!r} != {expected!r}')


def verify_rows(actual, expected, keys, label, from_csv=False):
    # NumPy logarithmic grids can differ by a few ULPs across platforms.
    # Match numeric coordinates within tolerance, retaining a unique identity
    # for duplicate/missing-row detection and checking every field afterwards.
    float_keys = [k for k in keys if any(isinstance(row[k], float) for row in expected)]
    fixed_keys = [k for k in keys if k not in float_keys]
    def fixed_key(row):
        return tuple('' if row[k] is None else str(row[k]) for k in fixed_keys)
    derived, candidates = {}, {}
    for row in expected:
        fixed = fixed_key(row)
        key = fixed, tuple(row[k] for k in float_keys)
        if key in derived:
            raise ValueError(f'{label}: duplicate expected key {key}')
        derived[key] = row
        candidates.setdefault(fixed, []).append(key)
    saved = {}
    for row in actual:
        numbers = tuple(float(row[k]) for k in float_keys)
        matches = [key for key in candidates.get(fixed_key(row), [])
                   if all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
                          for a, b in zip(numbers, key[1]))]
        if len(matches) != 1:
            raise ValueError(f'{label}: unexpected or ambiguous row coordinates')
        key = matches[0]
        if key in saved:
            raise ValueError(f'{label}: duplicate key {key}')
        saved[key] = row
    if set(saved) != set(derived):
        raise ValueError(f'{label}: missing or unexpected rows')
    for key, row in derived.items():
        compare_value(saved[key], row, f'{label}[{key}]', from_csv)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def verify_statistics(groups, out, runs_audited):
    """Rebuild all report JSON fields and exported tables from measured trajectories."""
    tables = summary_tables(groups)
    expected = statistics(runs_audited, tables)
    saved = json.loads((out / 'statistics.json').read_text(encoding='utf-8'))
    if set(saved) != set(expected):
        raise ValueError('statistics.json: missing or unexpected fields')
    for name, value in expected.items():
        if name in TABLE_KEYS:
            verify_rows(saved[name], value, TABLE_KEYS[name], f'statistics.json.{name}')
        else:
            compare_value(saved[name], value, f'statistics.json.{name}')
    for name, rows in tables.items():
        verify_rows(read_csv(out / (name + '.csv')), rows, TABLE_KEYS[name], name + '.csv', from_csv=True)
    return saved


def scientific_record(record):
    """Deterministic observations, excluding runtime-dependent clocks and timestamps."""
    return {key: record[key] for key in ['job', 'initial_ids', 'source_rows', 'weights']} | {
        'observations': [{'step': row['step'], 'candidate_id': row['candidate_id'], 'reward': row['reward']}
                         for row in record['rows']]}


def evidence_set(directory, jobs):
    expected = {job['id'] + '.json': job for job in jobs}
    paths = {path.name: path for path in directory.glob('*.json')}
    if set(paths) != set(expected):
        raise ValueError(f'{directory.name}: missing or unexpected evidence records')
    records = {}
    for name, job in expected.items():
        record = json.loads(paths[name].read_text(encoding='utf-8'))
        if record['job'] != job:
            raise ValueError(f'{name}: evidence job differs')
        audit(record)
        records[job['id']] = record
    return records


def verify_saved_evidence(runs, out, root=ROOT):
    """Audit the actual historical repeat and isolated-smoke files, not only flags."""
    from upgrade_study import manifest
    repeat = json.loads((out / 'reproducibility.json').read_text(encoding='utf-8'))
    isolated = json.loads((out / 'isolation_check.json').read_text(encoding='utf-8'))
    selected_seed = int(np.random.default_rng(20261015).choice(np.arange(10, 30)))
    originals = {r['job']['id']: r for r in runs if r['job']['seed'] == selected_seed}
    if len(originals) != 6 or repeat['selected_seed'] != selected_seed:
        raise ValueError('historical repeat seed differs')
    if (repeat['runs_checked'] != 6 or set(repeat['conditions']) != set(originals)
            or len(repeat['conditions']) != 6
            or not repeat['candidate_ids_rewards_initialization_weights_match']):
        raise ValueError('historical repeat metadata differs')
    repeats = evidence_set(root / 'evidence/repeats', [r['job'] for r in originals.values()])
    for identity in originals:
        compare_value(scientific_record(repeats[identity]), scientific_record(originals[identity]), identity)
    if (isolated['runs_compared'] != 6 or isolated['smoke_runs_passed'] != 6
            or not isolated['same_ids_rewards_initialization_weights']):
        raise ValueError('historical isolation metadata differs')
    working = evidence_set(root / 'evidence/smoke-working/runs', manifest(True))
    other = evidence_set(root / 'evidence/smoke-isolated/runs', manifest(True))
    for identity in working:
        compare_value(scientific_record(working[identity]), scientific_record(other[identity]), identity)
    return {'repeat_runs_audited': len(repeats), 'isolated_pairs_audited': len(working),
            'selected_seed': selected_seed, 'timing_equality_required': False}


def main():
    out = ROOT / 'results/upgrade'
    runs = [measure(r) for r in load_runs(out)]
    if len(runs) != 120 or sum(len(r['rows']) for r in runs) != 6000:
        raise ValueError('formal study must contain 120 trajectories / 6000 queries')
    groups = group_runs(runs)
    verify_statistics(groups, out, len(runs))
    evidence = verify_saved_evidence(runs, out)
    print(f'PASS: {len(runs)} frozen trajectories, 6,000 charged queries; all JSON statistics and CSV tables recomputed')
    print(f'PASS: {evidence["repeat_runs_audited"]} repeat runs and {evidence["isolated_pairs_audited"]} isolated smoke pairs audited from actual files')
    print('No optimizer rerun; frozen Windows timings remain historical observations.')


if __name__ == '__main__':
    main()
