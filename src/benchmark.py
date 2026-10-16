"""Shared runtime and input helpers for the VLMOP2 study."""
import csv
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def standardize_inputs(x):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (x - mean) / scale, mean, scale


def initial_ids(seed, n_pool, n_initial):
    return np.random.default_rng(seed).choice(n_pool, n_initial, replace=False)


def read_trace(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def environment_metadata():
    packages = ("physbo", "numpy", "scipy", "matplotlib", "threadpoolctl", "odat-se")
    versions = {p: importlib.metadata.version(p) for p in packages}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True)
    return {
        "python": platform.python_version(), "os": platform.platform(),
        "cpu": platform.processor(), "package_versions": versions, "code_version": VERSION,
        "git_commit_at_run": git.stdout.strip() if git.returncode == 0 else None,
        "threadpools": [{k: v for k, v in p.items() if k != "filepath"}
                        for p in threadpool_info()],
    }
