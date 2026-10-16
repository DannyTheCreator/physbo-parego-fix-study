"""PHYSBO v3 multi-objective practice on the update paper's VLMOP2 function."""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

from benchmark import (environment_metadata, initial_ids, read_trace, resolve_path,
                       standardize_inputs, utc_now, write_json)

PAPER_URL = "https://arxiv.org/html/2603.01349v1#S4.SS4.SSS1"
FIELDS = ["method", "seed", "step", "candidate_id", "x1", "x2",
          "reward1", "reward2", "phase", "elapsed_s"]
METHODS = {"Random", "HVPI", "ParEGO", "NDS"}


def load_config(path):
    cfg = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    required = {"name", "grid_points", "seeds", "methods", "n_initial", "budget",
                "num_rand_basis", "interval", "threads", "parego_weight_min",
                "parego_weight_sum", "nds_rank_max", "reference_point", "output_dir"}
    if set(cfg) != required:
        raise ValueError(f"configuration keys must be exactly {sorted(required)}")
    for key in ("grid_points", "n_initial", "budget", "num_rand_basis", "interval",
                "threads", "nds_rank_max"):
        if type(cfg[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if cfg["grid_points"] < 2 or not 2 <= cfg["n_initial"] < cfg["budget"] <= cfg["grid_points"]**2:
        raise ValueError("require 2 <= n_initial < budget <= grid_points squared")
    if cfg["num_rand_basis"] < 0 or cfg["interval"] < 0 or cfg["threads"] < 1 or cfg["nds_rank_max"] < 1:
        raise ValueError("invalid model or thread settings")
    seeds, methods = cfg["seeds"], cfg["methods"]
    if (not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct nonnegative 32-bit integers")
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= METHODS:
        raise ValueError("unsupported or repeated methods")
    for key in ("parego_weight_min", "parego_weight_sum"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError("invalid ParEGO coefficients")
    # This task's rewards are in (-1, 0]; the fixed lower reference is preregistered.
    if cfg["reference_point"] != [-1.0, -1.0]:
        raise ValueError("VLMOP2 requires the declared reference point (-1,-1)")
    return cfg


def candidate_grid(points):
    axis = np.linspace(-2.0, 2.0, points)
    x1, x2 = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack((x1.ravel(), x2.ravel()))


def vlmop2(x):
    """Return negative minimization objectives: larger rewards are better."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("VLMOP2 inputs must have finite shape (n,2)")
    offset = 1.0 / np.sqrt(2.0)
    return np.column_stack((np.expm1(-np.sum((x - offset)**2, axis=1)),
                            np.expm1(-np.sum((x + offset)**2, axis=1))))


class FunctionOracle:
    """Evaluate only requested points, count initialization, reject duplicates."""

    def __init__(self, x, budget, method, seed):
        self._x = np.array(x, dtype=float, copy=True)
        self.budget, self.method, self.seed = budget, method, seed
        self.phase = "initial"
        self.trace, self.queried = [], set()
        self.started = time.perf_counter()

    def __call__(self, actions):
        ids = np.atleast_1d(actions)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids):
            raise ValueError("candidate IDs must be a nonempty integer vector")
        if np.any(ids < 0) or np.any(ids >= len(self._x)):
            raise IndexError("candidate ID outside pool")
        if len(self.trace) + len(ids) > self.budget:
            raise RuntimeError("evaluation budget exceeded")
        if len(np.unique(ids)) != len(ids) or any(int(i) in self.queried for i in ids):
            raise RuntimeError("repeated candidate query")
        rewards = vlmop2(self._x[ids])
        for candidate, reward in zip(ids, rewards):
            self.queried.add(int(candidate))
            self.trace.append({
                "method": self.method, "seed": self.seed, "step": len(self.trace) + 1,
                "candidate_id": int(candidate), "x1": float(self._x[candidate, 0]),
                "x2": float(self._x[candidate, 1]), "reward1": float(reward[0]),
                "reward2": float(reward[1]), "phase": self.phase,
                "elapsed_s": time.perf_counter() - self.started,
            })
        return rewards


def run_method(method, seed, x_scaled, oracle, initial, cfg):
    """Policies see public inputs and queried rewards only; no full-label array."""
    import physbo  # execute() warms this before thread limits and per-run timing.
    rewards = oracle(initial)
    oracle.phase = "search"
    started = time.perf_counter()
    probes = cfg["budget"] - len(initial)
    if method == "Random":
        rng = np.random.default_rng(seed)
        if not np.array_equal(rng.choice(len(x_scaled), len(initial), replace=False), initial):
            raise ValueError("initialization differs from protocol")
        remaining = np.setdiff1d(np.arange(len(x_scaled)), initial)
        for candidate in rng.choice(remaining, probes, replace=False):
            oracle(np.array([candidate], dtype=int))
    else:
        policy_class = (physbo.search.discrete_multi.Policy if method == "HVPI"
                        else physbo.search.discrete_unified.Policy)
        policy = policy_class(test_X=x_scaled, num_objectives=2,
                              initial_data=(initial, rewards))
        policy.set_seed(seed)
        extra = {}
        if method == "ParEGO":
            extra["unify_method"] = physbo.search.unify.ParEGO(
                num_objectives=2, weight_min=cfg["parego_weight_min"],
                weight_sum=cfg["parego_weight_sum"])
        elif method == "NDS":
            extra["unify_method"] = physbo.search.unify.NDS(
                num_objectives=2, rank_max=cfg["nds_rank_max"])
        elif method != "HVPI":
            raise ValueError("unsupported method")
        # One call preserves PHYSBO's internal hyperparameter retraining interval.
        policy.bayes_search(max_num_probes=probes, num_search_each_probe=1,
                            simulator=oracle, score="HVPI" if method == "HVPI" else "EI",
                            interval=cfg["interval"], num_rand_basis=cfg["num_rand_basis"],
                            is_disp=False, **extra)
    duration = time.perf_counter() - started
    if len(oracle.trace) != cfg["budget"]:
        raise RuntimeError("strategy did not consume the declared evaluation budget")
    return oracle.trace, duration


def validate_trace(rows, cfg, method, seed, x):
    """Recompute objective values only in the offline audit."""
    if len(rows) != cfg["budget"]:
        raise ValueError("trace length differs from total evaluation budget")
    ids = np.array([int(r["candidate_id"]) for r in rows])
    if np.any(ids < 0) or np.any(ids >= len(x)) or len(np.unique(ids)) != len(ids):
        raise ValueError("out-of-range or repeated candidate in trace")
    initial = initial_ids(seed, len(x), cfg["n_initial"])
    if not np.array_equal(ids[:len(initial)], initial):
        raise ValueError("shared initialization differs from protocol")
    measured = np.array([[float(r["reward1"]), float(r["reward2"])] for r in rows])
    coords = np.array([[float(r["x1"]), float(r["x2"])] for r in rows])
    elapsed = np.array([float(r["elapsed_s"]) for r in rows])
    if not np.isfinite(np.concatenate((measured.ravel(), coords.ravel(), elapsed))).all():
        raise ValueError("nonfinite logged data")
    if not np.allclose(coords, x[ids], rtol=1e-12, atol=1e-12):
        raise ValueError("logged inputs differ from candidate grid")
    if not np.allclose(measured, vlmop2(x[ids]), rtol=1e-12, atol=1e-12):
        raise ValueError("logged rewards differ from ground truth")
    if np.any(elapsed < 0) or np.any(np.diff(elapsed) < 0):
        raise ValueError("invalid elapsed time")
    for step, r in enumerate(rows, 1):
        phase = "initial" if step <= cfg["n_initial"] else "search"
        if (int(r["step"]) != step or int(r["seed"]) != seed
                or r["method"] != method or r["phase"] != phase):
            raise ValueError("trace metadata mismatch")


def save_trace(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_completion(record, cfg, method, seed, x):
    if (record["status"] != "complete" or record["method"] != method or record["seed"] != seed
            or record["evaluations"] != cfg["budget"]
            or record["shared_initial_ids"] != initial_ids(seed, len(x), cfg["n_initial"]).tolist()
            or not np.isfinite(record["search_runtime_s"]) or record["search_runtime_s"] < 0):
        raise ValueError("invalid completion record")


def execute(cfg, output=None, resume=False):
    import physbo  # Load NumPy and SciPy BLAS before limiting threads/timing.
    x = candidate_grid(cfg["grid_points"])
    x_scaled, mean, scale = standardize_inputs(x)
    out = resolve_path(output or cfg["output_dir"])
    with threadpool_limits(limits=cfg["threads"]):
        env = environment_metadata()
        env["extension_version"] = "0.2.0"
        meta_path = out / "metadata.json"
        if meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if not resume:
                raise FileExistsError("result directory exists; use --resume or a new --output")
            if previous["config"] != cfg or previous["environment"]["package_versions"] != env["package_versions"]:
                raise ValueError("resume requires identical configuration and package versions")
            if not np.array_equal(np.load(out / "candidates.npy"), x):
                raise ValueError("saved candidate grid differs from protocol")
        else:
            if out.exists() and any(out.iterdir()):
                raise FileExistsError("nonempty result directory without metadata")
            (out / "runs").mkdir(parents=True)
            np.save(out / "candidates.npy", x)
            write_json(meta_path, {
                "created_at_utc": utc_now(), "config": cfg, "environment": env,
                "task": {"name": "VLMOP2", "source": PAPER_URL, "type": "synthetic_function",
                         "domain": [[-2.0, 2.0], [-2.0, 2.0]], "candidates": len(x),
                         "objective_direction": "maximize negative minimization objectives",
                         "units": "dimensionless"},
                "input_preprocessing": {"mean": mean.tolist(), "scale": scale.tolist(),
                                        "scope": "public candidate inputs only"},
                "timing": "post-initialization search; imports, I/O and offline metrics excluded",
            })
        for seed in cfg["seeds"]:
            initial = initial_ids(seed, len(x), cfg["n_initial"])
            for method in cfg["methods"]:
                stem = f"{method}_seed{seed}"
                csv_path, run_path = out / "runs" / f"{stem}.csv", out / "runs" / f"{stem}.json"
                if resume and csv_path.exists() and run_path.exists():
                    record = json.loads(run_path.read_text(encoding="utf-8"))
                    validate_completion(record, cfg, method, seed, x)
                    validate_trace(read_trace(csv_path), cfg, method, seed, x)
                    print(f"SKIP {stem}: completed trace verified", flush=True)
                    continue
                oracle = FunctionOracle(x, cfg["budget"], method, seed)
                try:
                    rows, duration = run_method(method, seed, x_scaled, oracle, initial, cfg)
                    validate_trace(rows, cfg, method, seed, x)
                    save_trace(csv_path, rows)
                    write_json(run_path, {
                        "status": "complete", "method": method, "seed": seed,
                        "evaluations": len(rows), "search_runtime_s": duration,
                        "shared_initial_ids": initial.tolist(), "finished_at_utc": utc_now(),
                    })
                    print(f"DONE {stem}: {len(rows)} evaluations, search {duration:.3f}s", flush=True)
                except Exception:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    save_trace(out / "runs" / f"{stem}_failed_{stamp}.csv", oracle.trace)
                    write_json(out / "runs" / f"{stem}_failed_{stamp}.json", {
                        "status": "failed", "traceback": traceback.format_exc(),
                        "finished_at_utc": utc_now(),
                    })
                    raise
    print(f"Verified {len(cfg['methods']) * len(cfg['seeds'])} VLMOP2 runs in {out.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vlmop2.json")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    execute(load_config(args.config), args.output, args.resume)


if __name__ == "__main__":
    main()
