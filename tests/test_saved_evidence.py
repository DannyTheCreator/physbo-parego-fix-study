"""Historical evidence must match actual audited traces, not merely true metadata flags."""
import importlib.util
import json
import shutil

import pytest

from benchmark import ROOT


def verifier():
    spec = importlib.util.spec_from_file_location('evidence_verifier', ROOT / 'scripts/verify_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved_evidence(tmp_path):
    shutil.copytree(ROOT / 'evidence/repeats', tmp_path / 'evidence/repeats')
    for name in ['smoke-working', 'smoke-isolated']:
        shutil.copytree(ROOT / 'evidence' / name / 'runs', tmp_path / 'evidence' / name / 'runs')
    out = tmp_path / 'results'
    out.mkdir()
    for name in ['reproducibility.json', 'isolation_check.json']:
        shutil.copyfile(ROOT / 'results/upgrade' / name, out / name)
    runs = [json.loads(path.read_text()) for path in (ROOT / 'results/upgrade/runs').glob('*_25.json')]
    return runs, out, tmp_path


def test_actual_saved_evidence_matches(saved_evidence):
    runs, out, root = saved_evidence
    result = verifier().verify_saved_evidence(runs, out, root)
    assert result['repeat_runs_audited'] == result['isolated_pairs_audited'] == 6


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'repeat-reward', 'repeat-id', 'smoke-reward', 'conditions'])
def test_corrupt_actual_evidence_fails_even_when_metadata_flags_are_true(saved_evidence, mutation):
    runs, out, root = saved_evidence
    path = sorted((root / 'evidence/repeats').glob('*.json'))[0]
    if mutation == 'missing':
        path.unlink()
    elif mutation == 'extra':
        shutil.copyfile(path, path.with_name('unexpected.json'))
    else:
        if mutation == 'smoke-reward':
            path = sorted((root / 'evidence/smoke-isolated/runs').glob('*.json'))[0]
        elif mutation == 'conditions':
            path = out / 'reproducibility.json'
        data = json.loads(path.read_text())
        if mutation == 'conditions':
            data['conditions'][-1] = data['conditions'][0]
        elif mutation == 'repeat-id':
            data['rows'][-1]['candidate_id'] = data['rows'][0]['candidate_id']
        else:
            data['rows'][-1]['reward'][0] += .1
        path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises((ValueError, AssertionError)):
        verifier().verify_saved_evidence(runs, out, root)
