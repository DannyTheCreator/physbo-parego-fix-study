"""Exercise first-run orchestration without downloading data or training models."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location('release_launcher', ROOT / 'scripts/run_upgrade.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('arguments,full', [([], False), (['--smoke-only'], False), (['--full'], True)])
def test_tests_precede_vlmop2_runs_and_results_are_separate(monkeypatch, arguments, full):
    module = launcher()
    calls = []
    monkeypatch.setattr(sys, 'argv', ['run_upgrade.py', *arguments])
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: calls.append(command[1:]))
    module.main()
    assert calls[0] == ['-m', 'pytest', '-q']
    assert ('--smoke' not in calls[1]) == full
    assert 'results/upgrade' not in calls[1]


def test_verification_never_runs_optimizer(monkeypatch):
    module = launcher()
    calls = []
    monkeypatch.setattr(sys, 'argv', ['run_upgrade.py', '--verify-only'])
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: calls.append(command[1:]))
    module.main()
    assert calls == [['scripts/verify_results.py']]
