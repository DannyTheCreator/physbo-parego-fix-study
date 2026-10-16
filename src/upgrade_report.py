"""Recompute the upgrade study's quality metrics and paired intervals."""
from __future__ import annotations

import argparse
import csv
import json
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from benchmark import ROOT, write_json
from multiobjective import candidate_grid, vlmop2
from multiobjective_report import hypervolume_2d, pareto_front_2d
from upgrade_study import audit, decision_times, DEFAULT_OUT, manifest

COLORS = {'Random': '#777777', 'HVPI': '#0072B2', 'ParEGO': '#D55E00', 'NDS': '#009E73',
          'ParEGO-min': '#0072B2', 'ParEGO-max': '#CC79A7'}


def table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bootstrap(values, alpha=.05):
    values = np.asarray(values)
    indices = np.random.default_rng(20261015).integers(len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {'mean': float(values.mean()), 'low': float(lo), 'high': float(hi)}


def binary_values(values):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError('expected a nonempty vector of binary seed outcomes')
    return values.astype(int)


@lru_cache(maxsize=None)
def _binomial_interval(successes, total, alpha):
    ci = binomtest(successes, total).proportion_ci(confidence_level=1-alpha, method='exact')
    return {'mean': successes / total, 'low': float(ci.low), 'high': float(ci.high)}


def binomial_interval(values, alpha=.05):
    """Pointwise Clopper-Pearson interval, including all-success/all-failure samples."""
    values = binary_values(values)
    return dict(_binomial_interval(int(values.sum()), len(values), alpha))


def first_hit(quality, threshold, n_initial=10):
    hits = np.flatnonzero(np.asarray(quality)[n_initial-1:] >= threshold)
    return int(hits[0] + n_initial) if len(hits) else None


def load_runs(out):
    meta = json.loads((out / 'metadata.json').read_text())
    jobs = manifest(meta['smoke'])
    if json.loads((out / 'manifest.json').read_text()) != jobs:
        raise ValueError('manifest mismatch')
    runs = []
    for job in jobs:
        path = out / 'runs' / (job['id'] + '.json')
        r = json.loads(path.read_text())
        if r['job'] != job:
            raise ValueError('job mismatch')
        audit(r)
        runs.append(r)
    for seed in {r['job']['seed'] for r in runs}:
        pair = [r for r in runs if r['job']['block'] == 'version' and r['job']['seed'] == seed]
        if pair[0]['weights'] != pair[1]['weights']:
            raise ValueError('version weight stream or call schedule differs')
    return runs


def measure(r):
    h = decision_times(r)
    job = r['job']
    rewards = np.array([row['reward'] for row in r['rows']])
    full_hv = hypervolume_2d(vlmop2(candidate_grid(job['grid_points'])))
    hv = np.array([hypervolume_2d(rewards[:i]) for i in range(1, len(rewards) + 1)])
    r['quality'] = hv / full_hv
    r['hv'] = hv
    r['quality_target'] = .9
    r['full_grid_hv'] = full_hv
    r['h'] = h
    r['hit'] = first_hit(r['quality'], r['quality_target'], job['n_initial'])
    return r


def group_runs(runs):
    groups = {}
    for r in runs:
        j = r['job']
        key = (j['block'], j['pool_seed'], j['init'])
        groups.setdefault(key, {}).setdefault(j['method'], []).append(r)
    for methods in groups.values():
        for rs in methods.values():
            rs.sort(key=lambda r: r['job']['seed'])
    return groups


def paired_difference(a, b, field, step):
    assert [r['job']['seed'] for r in a] == [r['job']['seed'] for r in b]
    return bootstrap(np.array([r[field][step-1] for r in a]) - np.array([r[field][step-1] for r in b]))


def plot_save(fig, figures, name):
    fig.savefig(figures / (name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(figures / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def figures_quality(groups, figures):
    methods = groups['multi', None, 'random']
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for method, rs in methods.items():
        values = np.array([r['quality'] for r in rs])
        cis = [bootstrap(values[:, i]) for i in range(values.shape[1])]
        n = np.arange(1, values.shape[1]+1)
        ax.plot(n, values.mean(axis=0), label=method, color=COLORS[method])
        ax.fill_between(n, [c['low'] for c in cis], [c['high'] for c in cis], color=COLORS[method], alpha=.13)
    ax.axhline(.9, color='black', linestyle=':', label='Predeclared 90% target')
    count = len(next(iter(methods.values())))
    ax.set(xlabel='Evaluations (including 10 initial)', ylabel='HV / full-grid HV', title=f'Current PHYSBO 3.2.1 | {count} seeds')
    ax.legend(fontsize=8)
    plot_save(fig, figures, 'multi_quality')
    version = groups['version', None, 'random']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for method, rs in version.items():
        v = np.array([r['quality'] for r in rs])
        axes[0].plot(np.arange(1, v.shape[1]+1), v.mean(axis=0), label=method, color=COLORS[method])
    full = vlmop2(candidate_grid(next(iter(version.values()))[0]['job']['grid_points']))
    front = pareto_front_2d(full)
    for ax, (method, rs) in zip(axes[1:], version.items()):
        r = rs[0]
        points = np.array([row['reward'] for row in r['rows']])
        found = pareto_front_2d(points)
        ax.plot(front[:, 0], front[:, 1], color='#bbbbbb', label='Full grid')
        ax.scatter(points[:, 0], points[:, 1], color='#bbbbbb', s=12)
        ax.scatter(found[:, 0], found[:, 1], color=COLORS[method], s=22)
        ax.set(title=f'{method} | fixed seed {r["job"]["seed"]}', xlabel='Reward 1', ylabel='Reward 2')
    axes[0].set(xlabel='Evaluations', ylabel='HV / full-grid HV', title='Controlled operator comparison')
    axes[0].legend(fontsize=8)
    fig.suptitle('Legacy max is a diagnostic, not a full PHYSBO 3.2.0 reproduction')
    plot_save(fig, figures, 'version_comparison')
def summary_tables(groups):
    summary, pairing, sensitivity = [], [], []
    for key, methods in groups.items():
        for method, runs in methods.items():
            for n in [10, 20, 50]:
                if n > runs[0]['job']['budget']:
                    continue
                ci = bootstrap([r['hv'][n-1] for r in runs])
                hit = binomial_interval([r['quality'][n-1] >= .9 for r in runs])
                summary.append({'block': key[0], 'pool_seed': key[1], 'initialization': key[2],
                                'method': method, 'evaluations': n, 'metric': 'hv', **ci,
                                'target_hit_rate': hit['mean'], 'target_hit_low': hit['low'],
                                'target_hit_high': hit['high'],
                                'mean_decision_s': float(np.mean([r['h'][n-1] for r in runs])),
                                'n_seeds': len(runs)})
            for target in [.8, .9, .95]:
                hits = [first_hit(r['quality'], target) for r in runs]
                sensitivity.append({'block': key[0], 'method': method, 'target': target,
                                    'successes': sum(h is not None for h in hits), 'total': len(hits)})
        if key[0] == 'version':
            n = next(iter(methods.values()))[0]['job']['budget']
            pairing.append({'comparison': 'current min minus legacy max', 'pool_seed': None,
                            'method': 'ParEGO', 'evaluations': n,
                            **paired_difference(methods['ParEGO-min'], methods['ParEGO-max'], 'hv', n)})
    return {'summary': summary, 'paired_comparisons': pairing, 'target_sensitivity': sensitivity}


ANALYSIS_METHODS = {
    'continuous_intervals': 'paired/ordinary percentile seed bootstrap; 10000 draws; seed 20261015',
    'success_intervals': 'pointwise 95% Clopper-Pearson exact binomial',
}


def statistics(runs_audited, tables):
    return {'runs_audited': runs_audited,
            **{k: tables[k] for k in ['summary', 'paired_comparisons', 'target_sensitivity']},
            'analysis_methods': dict(ANALYSIS_METHODS),
            'note': 'Continuous metrics use seed bootstrap; binary rates use exact binomial intervals. '
                    'Decision overhead is a separate local measurement. '
                    'Quality comparisons use fixed evaluation budgets.'}


def generate(out, figure_dir=None):
    out = Path(out)
    figures = Path(figure_dir) if figure_dir is not None else out / 'figures'
    runs = [measure(r) for r in load_runs(out)]
    groups = group_runs(runs)
    tables = summary_tables(groups)
    figures.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        table(out / (name + '.csv'), rows)
    figures_quality(groups, figures)
    stats = statistics(len(runs), tables)
    write_json(out / 'statistics.json', stats)
    print(json.dumps({'runs_audited': len(runs), 'figures': str(figures)}, indent=2))
    return stats


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=DEFAULT_OUT)
    p.add_argument('--figure-dir', type=Path, help='explicit figure destination (default: RESULTS/figures)')
    args = p.parse_args()
    generate(args.results, args.figure_dir)
