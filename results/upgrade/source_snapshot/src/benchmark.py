"""Budgeted offline search. Hidden targets are only exposed by LookupOracle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
SOURCE_COMMIT = "f44c9eab6ef63a3dd1689506b5cc445350a1f8ec"
DATA_URL = (
    "https://raw.githubusercontent.com/issp-center-dev/PHYSBO/"
    f"{SOURCE_COMMIT}/examples/grain_bound/data/s5-210.csv"
)
DATA_SHA256 = "18dc9afba5adee69ee58472b63829237384d641a1305c4e49b7566c8b56a02c8"
HEADER = "dx,gb,dz,gb_energy"
FIELDS = ["method", "seed", "step", "candidate_id", "source_row", "energy",
          "best_energy", "phase", "elapsed_s"]


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


def load_config(path):
    cfg = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    required = {"name", "pool_size", "pool_seed", "seeds", "methods", "n_initial",
                "budget", "num_rand_basis", "interval", "threads", "data_path", "output_dir"}
    if set(cfg) != required:
        raise ValueError(f"configuration keys must be exactly {sorted(required)}")
    for key in ("pool_size", "pool_seed", "n_initial", "budget", "num_rand_basis",
                "interval", "threads"):
        if type(cfg[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if not 2 <= cfg["n_initial"] < cfg["budget"] <= cfg["pool_size"]:
        raise ValueError("require 2 <= n_initial < budget <= pool_size")
    if cfg["num_rand_basis"] < 1 or cfg["interval"] < 0 or cfg["threads"] < 1:
        raise ValueError("invalid model or thread configuration")
    seeds = cfg["seeds"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("seeds must be distinct nonnegative 32-bit integers")
    if not cfg["methods"] or len(set(cfg["methods"])) != len(cfg["methods"]):
        raise ValueError("methods must be nonempty and distinct")
    if not set(cfg["methods"]) <= {"Random", "TS", "EI"}:
        raise ValueError("unsupported method")
    return cfg


def load_dataset(path):
    path = resolve_path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("dataset checksum mismatch; run scripts/download_data.py")
    if raw.decode("utf-8").splitlines()[0] != HEADER:
        raise ValueError("unexpected dataset columns")
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    if a.shape != (17982, 4) or not np.isfinite(a).all():
        raise ValueError("unexpected dataset shape or nonfinite values")
    if len(np.unique(a[:, :3], axis=0)) != len(a):
        raise ValueError("duplicate candidate parameter vectors")
    return a[:, :3], a[:, 3]


def pool_rows(n_rows, pool_size, seed):
    if not 1 <= pool_size <= n_rows:
        raise ValueError("pool size outside dataset")
    if pool_size == n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, pool_size, replace=False))


def standardize_inputs(x):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (x - mean) / scale, mean, scale


def initial_ids(seed, n_pool, n_initial):
    return np.random.default_rng(seed).choice(n_pool, n_initial, replace=False)


class LookupOracle:
    """A query is the only route from a strategy to the hidden energy column."""
    def __init__(self, energies, source_rows, budget, method, seed):
        self._energies = np.array(energies, dtype=float, copy=True)
        self._source_rows = np.asarray(source_rows, dtype=int)
        if (self._energies.ndim != 1 or len(self._source_rows) != len(self._energies)
                or not np.isfinite(self._energies).all()):
            raise ValueError("invalid oracle data")
        self.budget = budget
        self.method = method
        self.seed = seed
        self.phase = "initial"
        self.trace = []
        self.queried = set()
        self._best = float("inf")
        self.started = time.perf_counter()

    def __call__(self, actions):
        ids = np.atleast_1d(actions)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) == 0:
            raise ValueError("candidate IDs must be a nonempty integer vector")
        if np.any(ids < 0) or np.any(ids >= len(self._energies)):
            raise IndexError("candidate ID outside pool")
        if len(self.trace) + len(ids) > self.budget:
            raise RuntimeError("evaluation budget exceeded")
        if len(np.unique(ids)) != len(ids) or any(int(i) in self.queried for i in ids):
            raise RuntimeError("repeated candidate query")
        energies = self._energies[ids]
        for candidate, energy in zip(ids, energies):
            self._best = min(self._best, float(energy))
            self.queried.add(int(candidate))
            self.trace.append({
                "method": self.method, "seed": self.seed, "step": len(self.trace) + 1,
                "candidate_id": int(candidate), "source_row": int(self._source_rows[candidate]),
                "energy": float(energy), "best_energy": self._best, "phase": self.phase,
                "elapsed_s": time.perf_counter() - self.started,
            })
        return -energies


def run_method(method, seed, x, oracle, initial, cfg):
    """Strategies receive X and an opaque oracle, never the full target array."""
    initial_reward = oracle(initial)
    oracle.phase = "search"
    remaining_budget = cfg["budget"] - cfg["n_initial"]
    if method == "Random":
        # Continue the same RNG stream that generated the shared initial points.
        rng = np.random.default_rng(seed)
        assert np.array_equal(rng.choice(len(x), len(initial), replace=False), initial)
        remaining = np.setdiff1d(np.arange(len(x)), initial)
        for candidate in rng.choice(remaining, remaining_budget, replace=False):
            oracle(np.array([candidate], dtype=int))
    else:
        import physbo
        policy = physbo.search.discrete.Policy(test_X=x, initial_data=(initial, initial_reward))
        policy.set_seed(seed)
        policy.bayes_search(
            max_num_probes=remaining_budget, num_search_each_probe=1, simulator=oracle,
            score=method, interval=cfg["interval"], num_rand_basis=cfg["num_rand_basis"],
            is_disp=False,
        )
    if len(oracle.trace) != cfg["budget"]:
        raise RuntimeError("strategy did not consume the declared evaluation budget")
    return oracle.trace


def validate_trace(rows, cfg, method, seed, energies, source_rows):
    """Offline audit against dataset ground truth, including initialization."""
    if len(rows) != cfg["budget"]:
        raise ValueError("trace length differs from total evaluation budget")
    ids = np.array([int(r["candidate_id"]) for r in rows])
    if np.any(ids < 0) or np.any(ids >= len(energies)) or len(np.unique(ids)) != len(ids):
        raise ValueError("out-of-range or repeated candidate in trace")
    expected_initial = initial_ids(seed, len(energies), cfg["n_initial"])
    if not np.array_equal(ids[:cfg["n_initial"]], expected_initial):
        raise ValueError("shared initialization differs from protocol")
    measured = np.array([float(r["energy"]) for r in rows])
    best = np.array([float(r["best_energy"]) for r in rows])
    elapsed = np.array([float(r["elapsed_s"]) for r in rows])
    if not np.isfinite(np.concatenate([measured, best, elapsed])).all():
        raise ValueError("nonfinite logged data")
    if not np.allclose(measured, energies[ids], rtol=1e-12, atol=1e-12):
        raise ValueError("logged energy differs from dataset ground truth")
    if not np.allclose(best, np.minimum.accumulate(measured), rtol=1e-12, atol=1e-12):
        raise ValueError("incorrect best-so-far energy")
    if np.any(elapsed < 0) or np.any(np.diff(elapsed) < 0):
        raise ValueError("invalid elapsed time")
    for step, r in enumerate(rows, 1):
        expected_phase = "initial" if step <= cfg["n_initial"] else "search"
        if (int(r["step"]) != step or int(r["seed"]) != seed or r["method"] != method
                or int(r["source_row"]) != int(source_rows[ids[step - 1]])
                or r["phase"] != expected_phase):
            raise ValueError("trace metadata or source-row mapping mismatch")


def read_trace(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_trace(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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


def execute(cfg, output=None, seeds=None, resume=False):
    x, targets = load_dataset(cfg["data_path"])
    source_rows = pool_rows(len(x), cfg["pool_size"], cfg["pool_seed"])
    x_scaled, mean, scale = standardize_inputs(x[source_rows])
    pool_energy = targets[source_rows]
    selected = cfg["seeds"] if seeds is None else seeds
    if len(set(selected)) != len(selected) or not set(selected) <= set(cfg["seeds"]):
        raise ValueError("requested seeds must be a distinct subset of configured seeds")
    out = resolve_path(output or cfg["output_dir"])
    meta_path = out / "metadata.json"
    # Load SciPy's BLAS before applying limits; imports are outside per-run timers.
    if any(method in {"TS", "EI"} for method in cfg["methods"]):
        import physbo
    with threadpool_limits(limits=cfg["threads"]):
        env = environment_metadata()
        if meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if not resume:
                raise FileExistsError("result directory exists; use --resume or a new --output")
            if previous["config"] != cfg or previous["environment"]["package_versions"] != env["package_versions"]:
                raise ValueError("resume requires the identical configuration and package versions")
            if not np.array_equal(np.load(out / "pool_indices.npy"), source_rows):
                raise ValueError("saved candidate pool differs from protocol")
        else:
            if out.exists() and any(out.iterdir()):
                raise FileExistsError("nonempty result directory without metadata")
            out.mkdir(parents=True, exist_ok=True)
            (out / "runs").mkdir()
            np.save(out / "pool_indices.npy", source_rows)
            write_json(meta_path, {
                "created_at_utc": utc_now(), "config": cfg, "environment": env,
                "dataset": {"url": DATA_URL, "commit": SOURCE_COMMIT, "sha256": DATA_SHA256,
                            "total_candidates": len(x), "columns": HEADER.split(","), "units": "unverified"},
                "input_preprocessing": {"mean": mean.tolist(), "scale": scale.tolist(),
                                        "scope": "public candidate inputs only"},
                "protocol": "all evaluations include initialization; hidden labels used only by oracle/evaluator",
            })
        completed = 0
        for seed in selected:
            initial = initial_ids(seed, len(source_rows), cfg["n_initial"])
            for method in cfg["methods"]:
                stem = f"{method}_seed{seed}"
                csv_path = out / "runs" / f"{stem}.csv"
                run_path = out / "runs" / f"{stem}.json"
                if resume and csv_path.exists() and run_path.exists():
                    run_meta = json.loads(run_path.read_text(encoding="utf-8"))
                    if run_meta["status"] != "complete":
                        raise ValueError(f"invalid completion metadata: {stem}")
                    validate_trace(read_trace(csv_path), cfg, method, seed, pool_energy, source_rows)
                    print(f"SKIP {stem}: completed trace verified", flush=True)
                    completed += 1
                    continue
                oracle = LookupOracle(pool_energy, source_rows, cfg["budget"], method, seed)
                try:
                    rows = run_method(method, seed, x_scaled, oracle, initial, cfg)
                    duration = time.perf_counter() - oracle.started
                    validate_trace(rows, cfg, method, seed, pool_energy, source_rows)
                    save_trace(csv_path, rows)
                    write_json(run_path, {
                        "status": "complete", "method": method, "seed": seed,
                        "evaluations": len(rows), "runtime_s": duration,
                        "shared_initial_ids": initial.tolist(), "finished_at_utc": utc_now(),
                    })
                    completed += 1
                    print(f"DONE {stem}: {len(rows)} evaluations, {duration:.3f}s", flush=True)
                except Exception:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    save_trace(out / "runs" / f"{stem}_failed_{stamp}.csv", oracle.trace)
                    write_json(out / "runs" / f"{stem}_failed_{stamp}.json", {
                        "status": "failed", "method": method, "seed": seed,
                        "traceback": traceback.format_exc(), "finished_at_utc": utc_now(),
                    })
                    raise
        print(f"Verified {completed} runs in {out.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--output", help="new output directory (paths relative to repository)")
    parser.add_argument("--seeds", nargs="+", type=int, help="subset of configured seeds")
    parser.add_argument("--resume", action="store_true", help="verify and skip completed runs")
    args = parser.parse_args()
    execute(load_config(args.config), args.output, args.seeds, args.resume)


if __name__ == "__main__":
    main()
