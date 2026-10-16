"""Reject orphaned evidence, invalid oracle requests and impossible report clocks."""
import copy
import json
import shutil
import time

import numpy as np
import pytest

from benchmark import ROOT
import upgrade_report as report
import upgrade_study as study


def directory_state(directory):
    return {p.relative_to(directory): p.read_bytes() if p.is_file() else None
            for p in directory.rglob('*')}


def test_missing_manifest_never_relabels_existing_runs(tmp_path, monkeypatch):
    shutil.copytree(ROOT / 'evidence/smoke-working/runs', tmp_path / 'runs')
    monkeypatch.setattr(study, 'environment_metadata', lambda: {
        'cpu': 'different CPU', 'python': '3.11', 'os': 'different OS',
        'package_versions': {}, 'threadpools': []})
    monkeypatch.setattr(study, 'run_job', lambda job: pytest.fail('must not run optimizer'))
    before = directory_state(tmp_path)
    with pytest.raises(ValueError, match='new output directory'):
        study.execute(tmp_path, smoke=True)
    assert directory_state(tmp_path) == before


@pytest.mark.parametrize('artifact', ['metadata.json', 'queue_state.json',
                                      'source_snapshot/old.py', 'runs/partial_failed.json'])
def test_orphaned_campaign_artifacts_are_rejected_without_writes(tmp_path, artifact, monkeypatch):
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('old campaign', encoding='utf-8')
    monkeypatch.setattr(study, 'environment_metadata', lambda: pytest.fail('must reject before setup'))
    before = directory_state(tmp_path)
    with pytest.raises(ValueError, match='new output directory'):
        study.execute(tmp_path, smoke=True)
    assert directory_state(tmp_path) == before


@pytest.mark.parametrize('missing', ['metadata.json', 'source_snapshot/src/upgrade_study.py'])
def test_incomplete_resume_is_rejected_without_writes(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(study, 'run_job', lambda job: pytest.fail('must not run optimizer'))
    study.execute(tmp_path, smoke=True, blocks=['none'])
    (tmp_path / missing).unlink()
    before = directory_state(tmp_path)
    with pytest.raises(ValueError, match='new output directory'):
        study.execute(tmp_path, smoke=True)
    assert directory_state(tmp_path) == before


@pytest.fixture
def timed_record():
    path = ROOT / 'results/upgrade/runs/multi_0_random_HVPI_10.json'
    record = json.loads(path.read_text())
    # Exact values make the allowed cleanup boundary independent of local clocks.
    for step, row in enumerate(record['rows'], 1):
        row['decision_s'] = float(step)
    record['optimizer_tail_s'] = .25
    record['optimizer_total_s'] = 50.25
    record['rows'][-1]['decision_s'] = 50.25
    return record


@pytest.mark.parametrize('consumer', [study.audit, report.measure])
@pytest.mark.parametrize('tail', [50.25, 1.5, -1., float('nan'), float('inf')])
def test_impossible_cleanup_fails_in_audit_and_report(timed_record, consumer, tail):
    timed_record['optimizer_tail_s'] = tail
    with pytest.raises(ValueError, match='time'):
        consumer(timed_record)


@pytest.mark.parametrize('tail,expected_final', [(0., 50.25), (.25, 50.), (1.25, 49.)])
def test_valid_cleanup_produces_monotonic_availability_times(timed_record, tail, expected_final):
    timed_record['optimizer_tail_s'] = tail
    before = copy.deepcopy(timed_record['rows'])
    assert study.audit(timed_record)
    measured = report.measure(timed_record)
    assert measured['h'][-1] == expected_final
    assert np.all(np.diff(measured['h']) >= 0)
    assert timed_record['rows'] == before


def test_cleanup_subtraction_roundoff_is_clamped(timed_record):
    timed_record['optimizer_tail_s'] = 1.25 + 1e-14
    assert study.audit(timed_record)
    assert report.measure(timed_record)['h'][-1] == 49.


@pytest.mark.parametrize('consumer', [study.audit, report.measure])
@pytest.mark.parametrize('mutation', ['nan-total', 'wrong-total', 'nan-row', 'decreasing-row'])
def test_corrupt_raw_clocks_are_rejected(timed_record, consumer, mutation):
    if mutation == 'nan-total':
        timed_record['optimizer_total_s'] = float('nan')
    elif mutation == 'wrong-total':
        timed_record['optimizer_total_s'] += 1
    elif mutation == 'nan-row':
        timed_record['rows'][0]['decision_s'] = float('nan')
    else:
        timed_record['rows'][1]['decision_s'] = 0
    with pytest.raises(ValueError, match='time'):
        consumer(timed_record)


@pytest.mark.parametrize('actions,error', [
    ([1.9], ValueError), ([1.0], ValueError), ([], ValueError),
    ([[1]], ValueError), ([True], ValueError), (['1'], ValueError),
    ([float('nan')], ValueError), ([-1], IndexError), ([5], IndexError),
    (np.array([2**64 - 1], dtype=np.uint64), IndexError),
])
def test_invalid_candidate_requests_never_query_or_charge(actions, error):
    lookups = []

    def lookup(ids):
        lookups.append(ids.copy())
        return np.zeros((len(ids), 2))

    oracle = study.TimedOracle(lookup, 3, time.perf_counter(), n_candidates=5)
    with pytest.raises(error):
        oracle(actions)
    assert lookups == []
    assert oracle.rows == [] and oracle.seen == set() and oracle.query_s == 0


def test_repeated_and_overbudget_requests_leave_valid_history_unchanged():
    lookups = []

    def lookup(ids):
        lookups.append(ids.tolist())
        return np.zeros((len(ids), 2))

    oracle = study.TimedOracle(lookup, 3, time.perf_counter(), n_candidates=5)
    oracle(np.array([0, 1], dtype=np.int64))
    before = copy.deepcopy(oracle.rows), oracle.seen.copy(), oracle.query_s
    for actions in [[2, 2], [1], [2, 3]]:
        with pytest.raises(ValueError):
            oracle(actions)
        assert (oracle.rows, oracle.seen, oracle.query_s) == before
    oracle(np.int64(2))
    assert lookups == [[0, 1], [2]]
    assert [row['candidate_id'] for row in oracle.rows] == [0, 1, 2]
