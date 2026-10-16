"""Rerunning the repeat command must leave published historical metadata untouched."""
import importlib.util
import json
import sys

import pytest

from benchmark import ROOT


def launcher():
    spec = importlib.util.spec_from_file_location('repeat_launcher', ROOT / 'scripts/verify_upgrade_repro.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repeat_writes_only_to_its_new_output(monkeypatch, tmp_path):
    module = launcher()
    original_meta = ROOT / 'results/upgrade/reproducibility.json'
    before = original_meta.read_bytes()
    monkeypatch.setattr(module, 'run_job', lambda job: json.loads(
        (ROOT / 'results/upgrade/runs' / (job['id'] + '.json')).read_text()))
    out = tmp_path / 'fresh-repeat'
    monkeypatch.setattr(sys, 'argv', ['verify_upgrade_repro.py', '--output', str(out)])
    module.main()
    assert original_meta.read_bytes() == before
    result = json.loads((out / 'comparison.json').read_text())
    assert result['runs_checked'] == 6 and result['selected_seed'] == 25
    assert len(list(out.glob('*_25.json'))) == 6


def test_repeat_rejects_nonempty_output_before_running_optimizer(monkeypatch, tmp_path):
    module = launcher()
    (tmp_path / 'existing.txt').write_text('preserve')
    monkeypatch.setattr(module, 'run_job', lambda job: pytest.fail('unexpected optimizer execution'))
    monkeypatch.setattr(sys, 'argv', ['verify_upgrade_repro.py', '--output', str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert (tmp_path / 'existing.txt').read_text() == 'preserve'
