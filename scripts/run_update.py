"""Run checks, a small v3 benchmark and the paper-aligned VLMOP2 configuration."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    commands = [["-m", "pytest", "-q"]]
    names = ["vlmop2_smoke"] if args.smoke_only else ["vlmop2_smoke", "vlmop2"]
    for name in names:
        commands.extend([
            ["src/multiobjective.py", "--config", f"configs/{name}.json", "--resume"],
            ["src/multiobjective_report.py", "--results", f"results/{name}"],
        ])
    for command in commands:
        print("RUN python " + " ".join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
