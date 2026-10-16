"""Repeat one seed selected without inspecting outcomes across all 6 VLMOP2 conditions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from benchmark import write_json
from upgrade_study import manifest, run_job, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.cache/repeat-seed25'),
                        help='new output directory; historical evidence is never overwritten')
    args = parser.parse_args()
    seed = int(np.random.default_rng(20261015).choice(np.arange(10, 30)))
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists() and any(out.iterdir()):
        parser.error('use a new empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    checked = []
    with threadpool_limits(limits=1):
        for job in manifest():
            if job['seed'] != seed:
                continue
            original = json.loads((ROOT / 'results/upgrade/runs' / (job['id'] + '.json')).read_text())
            repeated = run_job(job)
            audit(repeated)
            assert original['initial_ids'] == repeated['initial_ids']
            assert original['source_rows'] == repeated['source_rows']
            assert original['weights'] == repeated['weights']
            for a, b in zip(original['rows'], repeated['rows']):
                assert a['candidate_id'] == b['candidate_id']
                np.testing.assert_array_equal(a['reward'], b['reward'])
            write_json(out / (job['id'] + '.json'), repeated)
            checked.append(job['id'])
            print('MATCH ' + job['id'], flush=True)
    write_json(out / 'comparison.json',
               {'seed_selection_rng': 20261015, 'selected_seed': seed, 'runs_checked': len(checked),
                'conditions': checked, 'candidate_ids_rewards_initialization_weights_match': True,
                'timing_match_required': False,
                'source_evidence': 'results/upgrade/runs', 'repeated_runs': '.'})
    print(f'{len(checked)} matched repeated runs for seed {seed}', flush=True)


if __name__ == '__main__':
    main()
