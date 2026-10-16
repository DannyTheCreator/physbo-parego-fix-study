"""Regression checks for report evidence, binary uncertainty and safe campaign reuse."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import upgrade_report as report
import upgrade_study as study
from benchmark import ROOT, write_json


def verifier():
    spec = importlib.util.spec_from_file_location('result_verifier', ROOT / 'scripts/verify_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def measured_groups():
    runs = []
    for block, pool, init, names in [
        ('version', None, 'random', ['ParEGO-min', 'ParEGO-max']),
        ('multi', None, 'random', ['Random', 'HVPI', 'ParEGO', 'NDS']),
    ]:
        for i, method in enumerate(names):
            for seed in [10, 11]:
                cap = 50
                quality = np.linspace(.1, 1.1, cap)
                runs.append({'job': {'block': block, 'pool_seed': pool, 'init': init,
                                     'method': method, 'seed': seed, 'budget': cap, 'n_initial': 10},
                             'quality': quality, 'quality_target': 1., 'h': np.linspace(.001, .1+i*.01, cap),
                             'hv': quality * (1+i*.01), 'gap': 1/quality, 'hit': cap-4,
                             'initial_cover_radius': .2, 'initial_best': .3, 'initial_selection_s': .001})
    return report.group_runs(runs)


@pytest.fixture
def saved_evidence(tmp_path, measured_groups):
    tables = report.summary_tables(measured_groups)
    stats = report.statistics(12, tables)
    write_json(tmp_path / 'statistics.json', stats)
    for name, rows in tables.items():
        report.table(tmp_path / (name + '.csv'), rows)
    return tmp_path, stats


def test_all_report_tables_and_reordered_json_are_verified(saved_evidence, measured_groups):
    out, stats = saved_evidence
    for name in ['summary', 'paired_comparisons', 'target_sensitivity']:
        stats[name].reverse()
    write_json(out / 'statistics.json', stats)
    verifier().verify_statistics(measured_groups, out, 12)


@pytest.mark.parametrize('section,field,value', [
    ('summary', 'mean', -999), ('summary', 'low', -999), ('summary', 'high', -999),
    ('summary', 'n_seeds', 123), ('summary', 'target_hit_rate', -1),
    ('summary', 'target_hit_low', -1), ('summary', 'target_hit_high', -1),
    ('summary', 'mean_decision_s', -1), ('paired_comparisons', 'mean', 123.456),
    ('paired_comparisons', 'low', -999), ('paired_comparisons', 'high', -999),
    ('target_sensitivity', 'total', 123),
])
def test_corrupt_report_json_fails(saved_evidence, measured_groups, section, field, value):
    out, stats = saved_evidence
    stats[section][0][field] = value
    write_json(out / 'statistics.json', stats)
    with pytest.raises((ValueError, AssertionError)):
        verifier().verify_statistics(measured_groups, out, 12)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'extra-field', 'wrong-key', 'nan', 'methods'])
def test_statistical_schema_corruption_fails(saved_evidence, measured_groups, mutation):
    out, stats = saved_evidence
    if mutation == 'duplicate':
        stats['summary'].append(stats['summary'][0])
    elif mutation == 'missing':
        stats['paired_comparisons'].pop()
    elif mutation == 'extra-field':
        stats['summary'][0]['stale'] = 1
    elif mutation == 'wrong-key':
        stats['summary'][0]['method'] = 'other'
    elif mutation == 'nan':
        stats['summary'][0]['mean'] = float('nan')
    else:
        stats['analysis_methods']['success_intervals'] = 'wrong method'
    (out / 'statistics.json').write_text(json.dumps(stats), encoding='utf-8')
    with pytest.raises((ValueError, AssertionError)):
        verifier().verify_statistics(measured_groups, out, 12)


@pytest.mark.parametrize('name', ['summary', 'paired_comparisons', 'target_sensitivity'])
def test_missing_exported_row_fails(saved_evidence, measured_groups, name):
    out, _ = saved_evidence
    path = out / (name + '.csv')
    lines = path.read_text(encoding='utf-8').splitlines()
    path.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    with pytest.raises(ValueError):
        verifier().verify_statistics(measured_groups, out, 12)


def test_exact_binary_uncertainty():
    all_success = report.binomial_interval(np.ones(20))
    all_failure = report.binomial_interval(np.zeros(20))
    assert all_success['low'] == pytest.approx(.8315665290169146)
    assert all_success['high'] == 1
    assert all_failure['low'] == 0
    assert all_failure['high'] == pytest.approx(1-all_success['low'])
def test_equal_result_basenames_have_distinct_figure_outputs(tmp_path, monkeypatch, measured_groups):
    destinations = []
    monkeypatch.setattr(report, 'load_runs', lambda out: [])
    monkeypatch.setattr(report, 'group_runs', lambda runs: measured_groups)
    monkeypatch.setattr(report, 'figures_quality', lambda groups, figures: destinations.append(figures))
    a, b = tmp_path / 'results/upgrade', tmp_path / '.cache/upgrade'
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    report.generate(a)
    report.generate(b)
    explicit = tmp_path / 'selected-figures'
    report.generate(b, figure_dir=explicit)
    assert destinations == [a / 'figures', b / 'figures', explicit]
    assert len(set(destinations)) == 3


@pytest.fixture
def environment():
    return {'cpu': 'CPU-A', 'python': '3.11.7', 'os': 'OS-A', 'package_versions': {'numpy': '1'},
            'threadpools': [{'internal_api': 'openblas', 'version': '1', 'architecture': 'SkylakeX',
                             'prefix': 'libblas', 'user_api': 'blas', 'num_threads': 1},
                            {'internal_api': 'openblas', 'version': '2', 'num_threads': 1}]}


@pytest.fixture
def empty_campaign(tmp_path, monkeypatch, environment):
    monkeypatch.setattr(study, 'environment_metadata', lambda: copy.deepcopy(environment))
    monkeypatch.setattr(study, 'run_job', lambda job: pytest.fail('unexpected optimizer execution'))
    study.execute(tmp_path, smoke=True, blocks=['none'])
    return tmp_path


@pytest.mark.parametrize('field', ['cpu', 'python', 'os', 'blas', 'threads', 'architecture',
                                 'missing', 'missing-blas', 'packages', 'fingerprint'])
def test_resume_rejects_timing_environment_changes(empty_campaign, environment, field):
    meta_path = empty_campaign / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    if field in ['cpu', 'python', 'os']:
        meta['environment'][field] = 'OTHER'
    elif field in ['blas', 'threads', 'architecture']:
        key = {'blas': 'version', 'threads': 'num_threads', 'architecture': 'architecture'}[field]
        meta['environment']['threadpools'][0][key] = 'OTHER'
    elif field == 'missing':
        del meta['environment']['python']
    elif field == 'missing-blas':
        meta['environment']['threadpools'].pop()
    elif field == 'packages':
        meta['environment']['package_versions']['numpy'] = 'OTHER'
    else:
        meta['timing_environment_fingerprint'] = 'OTHER'
    write_json(meta_path, meta)
    before = {p.relative_to(empty_campaign): p.read_bytes() for p in empty_campaign.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='new output directory'):
        study.execute(empty_campaign, smoke=True, blocks=['none'])
    assert before == {p.relative_to(empty_campaign): p.read_bytes() for p in empty_campaign.rglob('*') if p.is_file()}


def test_resume_accepts_same_identity_and_ignores_discovery_order_and_paths(empty_campaign, environment):
    meta_path = empty_campaign / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    meta['environment']['threadpools'].reverse()
    for pool in meta['environment']['threadpools']:
        pool['filepath'] = '/another/install/path'
    meta['environment']['git_commit_at_run'] = 'different-unrelated-commit'
    write_json(meta_path, meta)
    study.execute(empty_campaign, smoke=True, blocks=['none'])
