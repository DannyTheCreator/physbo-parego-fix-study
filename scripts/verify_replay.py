"""Compare a complete fresh campaign with retained original scientific observations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from benchmark import write_json
from upgrade_report import load_runs
from verify_results import compare_value, scientific_record


def verify(original, replay):
    originals = {r['job']['id']: r for r in load_runs(original)}
    repeated = {r['job']['id']: r for r in load_runs(replay)}
    if set(originals) != set(repeated):
        raise ValueError('campaign conditions differ')
    for identity in originals:
        compare_value(scientific_record(repeated[identity]), scientific_record(originals[identity]), identity)
    return {'runs_compared': len(originals), 'evaluations_compared': sum(len(r['rows']) for r in originals.values()),
            'job_initialization_candidate_ids_rewards_weights_match': True,
            'timing_equality_required': False,
            'note': 'Fresh replay verifies scientific observations only. Original timings remain the reported measurements.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--original', type=Path, default=ROOT / 'results/upgrade')
    parser.add_argument('--summary', type=Path, help='optional output JSON comparison summary')
    args = parser.parse_args()
    result = verify(args.original, args.replay)
    if args.summary:
        write_json(args.summary, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
