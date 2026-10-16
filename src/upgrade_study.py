"""ParEGO fix effect and fixed-budget VLMOP2 strategy study; local serial resumable queue."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import physbo
from threadpoolctl import threadpool_limits

from benchmark import (ROOT, environment_metadata, initial_ids, standardize_inputs, utc_now, write_json)
from multiobjective import candidate_grid, vlmop2

PAREGO = importlib.import_module(physbo.search.unify.ParEGO.__module__)
DEFAULT_OUT = ROOT / 'results' / 'upgrade'


def initialize(x, n, seed, kind):
    if kind != 'random':
        raise ValueError('this study uses shared random initialization only')
    return initial_ids(seed, len(x), n)


class PairedParEGO:
    """Only the reduction changes; weight randomness is independent of the GP."""
    def __init__(self, seed, operator):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
        self.operator, self.calls = operator, []

    def __call__(self, t):
        weights = self.rng.random(2)
        weights /= weights.sum()
        self.calls.append({'call': len(self.calls), 'observations': len(t),
                           'weights': weights.tolist()})
        current = physbo.search.unify.ParEGO(2, weights=weights, weight_min=1., weight_sum=.05)
        if self.operator == 'min':
            return current(t)
        weighted = PAREGO.min_max_scaling(np.asarray(t).reshape(-1, 2), -1., 0.) * current.weights
        return (np.max(weighted, axis=1) + .05 * weighted.sum(axis=1)).reshape(-1, 1)


def three_points():
    t = np.array([[-1., 0.], [0., -1.], [-.4, -.4]])
    w = np.array([.5, .5])
    cur = physbo.search.unify.ParEGO(2, weights=w)(t).ravel()
    z = PAREGO.min_max_scaling(t, -1., 0.) * w
    old = z.max(axis=1) + .05 * z.sum(axis=1)
    np.testing.assert_allclose(cur, [-.525, -.525, -.22])
    np.testing.assert_allclose(old, [-.025, -.025, -.22])
    return {'points': t.tolist(), 'weights': w.tolist(), 'current_min': cur.tolist(),
            'legacy_max_diagnostic': old.tolist()}


def manifest(smoke=False):
    seeds = [104] if smoke else list(range(10, 30))
    jobs = []
    # Legacy pool keys below preserve original trace identity; the grid defines all candidates.
    def add(block, methods, init, seed, pool=None):
        shift = seed % len(methods)
        order = methods[shift:] + methods[:shift]
        for method in order:
            jobs.append({'id': f'{block}_{pool or 0}_{init}_{method}_{seed}',
                         'block': block, 'method': method, 'init': init, 'seed': seed,
                         'pool_seed': pool, 'budget': (14 if smoke else 50),
                         'n_initial': 10, 'pool_size': 256 if smoke else 2000,
                         'grid_points': 15 if smoke else 101,
                         'num_rand_basis': 0, 'interval': 10})
    for seed in seeds:
        add('version', ['ParEGO-min', 'ParEGO-max'], 'random', seed)
    for seed in seeds:
        add('multi', ['Random', 'HVPI', 'ParEGO', 'NDS'], 'random', seed)
    return jobs


class TimedOracle:
    def __init__(self, lookup, budget, started):
        self._lookup, self.budget, self.started = lookup, budget, started
        self.query_s, self.rows, self.seen = 0., [], set()

    def __call__(self, actions):
        tick = time.perf_counter()
        h = tick - self.started - self.query_s
        ids = np.atleast_1d(actions).astype(int)
        if len(self.rows) + len(ids) > self.budget or len(set(ids.tolist())) != len(ids):
            raise ValueError('invalid budget or duplicate batch')
        if any(int(i) in self.seen for i in ids):
            raise ValueError('repeated query')
        values = np.asarray(self._lookup(ids))
        for i, value in zip(ids, values):
            self.seen.add(int(i))
            self.rows.append({'step': len(self.rows) + 1, 'candidate_id': int(i),
                              'reward': np.atleast_1d(value).tolist(), 'decision_s': h})
        self.query_s += time.perf_counter() - tick
        return values


def inputs(job):
    if job['block'] not in {'version', 'multi'}:
        raise ValueError('unsupported study block')
    raw = candidate_grid(job['grid_points'])
    x, mean, scale = standardize_inputs(raw)
    # Full multiobjective values are not prepared here. Oracle evaluates requested points.
    return x, np.arange(len(raw)), None, {'mean': mean.tolist(), 'scale': scale.tolist()}


def policy_search(job, x, oracle, initial):
    """Receives queried rewards only; private labels never enter policy setup."""
    seed, method = job['seed'], job['method']
    rewards = oracle(initial)
    probes = job['budget'] - len(initial)
    unify = None
    if method == 'Random':
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7341]))
        priority = rng.permutation(len(x))
        rest = priority[~np.isin(priority, initial)][:probes]
        for candidate in rest:
            oracle(np.array([candidate]))
    else:
        cls = physbo.search.discrete_multi.Policy if method == 'HVPI' else physbo.search.discrete_unified.Policy
        policy = cls(test_X=x, num_objectives=2, initial_data=(initial, rewards))
        score = 'HVPI' if method == 'HVPI' else 'EI'
        if job['block'] == 'version':
            unify = PairedParEGO(seed, 'min' if method == 'ParEGO-min' else 'max')
        elif method == 'ParEGO':
            unify = physbo.search.unify.ParEGO(2, weight_min=1., weight_sum=.05)
        elif method == 'NDS':
            unify = physbo.search.unify.NDS(2, rank_max=10)
        policy.set_seed(seed)
        extra = {'unify_method': unify} if unify is not None else {}
        policy.bayes_search(max_num_probes=probes, num_search_each_probe=1, simulator=oracle,
                            score=score, interval=job['interval'], num_rand_basis=job['num_rand_basis'],
                            is_disp=False, **extra)
    return unify.calls if isinstance(unify, PairedParEGO) else []


def run_job(job):
    x, source, labels, preprocessing = inputs(job)
    raw = candidate_grid(job['grid_points'])
    lookup = (lambda ids: vlmop2(raw[ids]))
    started = time.perf_counter()
    initial = initialize(x, job['n_initial'], job['seed'], job['init'])
    init_s = time.perf_counter() - started
    oracle = TimedOracle(lookup, job['budget'], started)
    try:
        weights = policy_search(job, x, oracle, initial)
    except Exception as exc:
        exc.partial_trace = oracle.rows
        raise
    total = time.perf_counter() - started - oracle.query_s
    tail_s = total - oracle.rows[-1]['decision_s']
    oracle.rows[-1]['decision_s'] = total
    result = {'status': 'complete', 'job': job, 'rows': oracle.rows,
              'initial_ids': initial.tolist(), 'source_rows': source.tolist(),
              'input_preprocessing': preprocessing, 'initial_selection_s': init_s,
              'optimizer_total_s': total,
              'optimizer_tail_s': tail_s, 'oracle_and_logging_s': oracle.query_s,
              'weights': weights, 'finished_at_utc': utc_now()}
    audit(result)
    return result


def audit(record):
    job, rows = record['job'], record['rows']
    if record['status'] != 'complete' or len(rows) != job['budget']:
        raise ValueError('incomplete record')
    x, source, labels, _ = inputs(job)
    ids = np.array([r['candidate_id'] for r in rows])
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(x)):
        raise ValueError('invalid candidate IDs')
    initial = initialize(x, job['n_initial'], job['seed'], job['init'])
    np.testing.assert_array_equal(ids[:len(initial)], initial)
    np.testing.assert_array_equal(source, record['source_rows'])
    np.testing.assert_array_equal(initial, record['initial_ids'])
    expected = vlmop2(candidate_grid(job['grid_points'])[ids])
    np.testing.assert_allclose([r['reward'] for r in rows], expected, rtol=1e-12, atol=1e-12)
    h = np.array([r['decision_s'] for r in rows])
    if not np.isfinite(h).all() or np.any(h < 0) or np.any(np.diff(h) < 0):
        raise ValueError('invalid cumulative decision time')
    if [r['step'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('invalid step sequence')
    np.testing.assert_allclose(h[-1], record['optimizer_total_s'])
    if not 0 <= record['optimizer_tail_s'] <= h[-1]:
        raise ValueError('invalid final tail time')
    if job['block'] == 'version':
        calls = record['weights']
        expected_count = job['budget'] - job['n_initial'] + 1
        if len(calls) != expected_count:
            raise ValueError('missing scalarization weight calls')
        rng = np.random.default_rng(np.random.SeedSequence([job['seed'], 991]))
        for k, call in enumerate(calls):
            weights = rng.random(2)
            weights /= weights.sum()
            if call['call'] != k or call['observations'] != job['n_initial'] + k:
                raise ValueError('incorrect weight call schedule')
            np.testing.assert_array_equal(weights, call['weights'])
    return True


def timing_environment_fingerprint(environment):
    """Canonical timing identity; library discovery order and install paths are irrelevant."""
    fields = ['cpu', 'python', 'os', 'package_versions', 'threadpools']
    if any(k not in environment or environment[k] is None for k in fields):
        raise ValueError('missing timing environment identity; use a new output directory')
    identity = {k: environment[k] for k in fields}
    identity['threadpools'] = sorted(
        [{k: v for k, v in pool.items() if k != 'filepath'} for pool in environment['threadpools']],
        key=lambda pool: json.dumps(pool, sort_keys=True))
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()


def execute(out, smoke=False, blocks=None):
    out = Path(out)
    jobs = manifest(smoke)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'runs').mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        env = environment_metadata()
        fingerprint = timing_environment_fingerprint(env)
        manifest_path = out / 'manifest.json'
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old != jobs:
                raise ValueError('manifest changed; use a new output directory')
            metadata = json.loads((out / 'metadata.json').read_text())
            if metadata['environment']['package_versions'] != env['package_versions']:
                raise ValueError('package versions changed; use a new output directory')
            recorded = timing_environment_fingerprint(metadata['environment'])
            if (recorded != fingerprint
                    or metadata.get('timing_environment_fingerprint', recorded) != recorded):
                raise ValueError('timing environment changed (CPU/Python/OS/BLAS/threads); use a new output directory')
            for rel in ['src/upgrade_study.py', 'src/benchmark.py', 'src/multiobjective.py', 'docs/PROTOCOL.md']:
                if (out / 'source_snapshot' / rel).read_bytes() != (ROOT / rel).read_bytes():
                    raise ValueError('implementation changed; use a new output directory')
        else:
            for rel in ['src/upgrade_study.py', 'src/benchmark.py', 'src/multiobjective.py', 'docs/PROTOCOL.md']:
                dest = out / 'source_snapshot' / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, dest)
            write_json(manifest_path, jobs)
            write_json(out / 'metadata.json', {'created_at_utc': utc_now(), 'environment': env,
                       'timing_environment_fingerprint': fingerprint,
                       'task': 'VLMOP2', 'protocol': 'docs/PROTOCOL.md',
                       'timing': 'cumulative decision overhead, oracle/logging excluded; final tail charged',
                       'smoke': smoke})
        write_json(out / 'three_points.json', three_points())
        state = {}
        if (out / 'queue_state.json').exists():
            state = json.loads((out / 'queue_state.json').read_text())
        for job in jobs:
            if blocks and job['block'] not in blocks:
                continue
            path = out / 'runs' / (job['id'] + '.json')
            if path.exists():
                rec = json.loads(path.read_text())
                if rec['job'] != job:
                    raise ValueError('saved job changed')
                audit(rec)
                state[job['id']] = 'complete'
                continue
            state[job['id']] = 'running'
            write_json(out / 'queue_state.json', state)
            captured = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured):
                    record = run_job(job)
                write_json(path, record)
                state[job['id']] = 'complete'
                print(f"DONE {job['id']}: {record['optimizer_total_s']:.3f}s", flush=True)
            except Exception:
                state[job['id']] = 'failed'
                write_json(out / 'runs' / (job['id'] + '_failed.json'),
                           {'job': job, 'traceback': traceback.format_exc(), 'stdout': captured.getvalue(),
                            'partial_trace': getattr(__import__('sys').exception(), 'partial_trace', [])})
                raise
            finally:
                write_json(out / 'queue_state.json', state)
    print(f"QUEUE {sum(v == 'complete' for v in state.values())}/{len(jobs)} complete", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--blocks', nargs='+', choices=['version', 'multi'])
    args = p.parse_args()
    execute(args.output, args.smoke, args.blocks)


if __name__ == '__main__':
    main()
