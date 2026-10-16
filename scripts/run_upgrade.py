"""Smoke-test, verify frozen results, or run a fresh campaign."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--smoke-only', action='store_true', help='run 6 small VLMOP2 trajectories (the default)')
    mode.add_argument('--verify-only', action='store_true', help='audit the retained 120 VLMOP2 trajectories')
    mode.add_argument('--full', action='store_true', help='run all 120 VLMOP2 trajectories in a separate directory')
    p.add_argument('--output', type=Path, help='new or resumable output directory for smoke/full runs')
    args = p.parse_args()
    if args.verify_only and args.output:
        p.error('--output applies to smoke/full runs; retained results are verified in place')
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(ROOT / '.cache' / 'matplotlib')
    commands = []
    if args.verify_only:
        commands.append(['scripts/verify_results.py'])
    else:
        out = args.output or Path('.cache/vlmop2-full' if args.full else '.cache/vlmop2-smoke')
        commands += [['-m', 'pytest', '-q'],
                     ['src/upgrade_study.py', '--output', str(out)] + ([] if args.full else ['--smoke']),
                     ['src/upgrade_report.py', '--results', str(out)]]
    for command in commands:
        print('RUN ' + ' '.join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
