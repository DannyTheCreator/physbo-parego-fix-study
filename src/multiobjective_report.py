"""Audit VLMOP2 trajectories; compare dominated area and search overhead."""
from __future__ import annotations

import argparse
import json

import numpy as np

from benchmark import ROOT, read_trace, resolve_path, write_json
from multiobjective import (candidate_grid, validate_completion, validate_trace, vlmop2)
from report import archive_existing, plt, save_figure, write_table

COLORS = {"Random": "#666666", "HVPI": "#0072B2", "ParEGO": "#D55E00", "NDS": "#009E73"}


def pareto_front_2d(rewards):
    """Unique non-dominated rewards, maximizing both axes, sorted by axis 1."""
    rewards = np.asarray(rewards, dtype=float).reshape(-1, 2)
    order = np.lexsort((-rewards[:, 1], -rewards[:, 0]))
    front, best_y = [], -np.inf
    for point in rewards[order]:
        if point[1] > best_y:
            front.append(point)
            best_y = point[1]
    return np.array(front[::-1], dtype=float).reshape(-1, 2)


def hypervolume_2d(rewards, reference=(-1.0, -1.0)):
    """Exact union area of rectangles from a fixed lower reference to rewards."""
    rewards = np.asarray(rewards, dtype=float).reshape(-1, 2)
    reference = np.asarray(reference, dtype=float)
    if not np.isfinite(rewards).all() or not np.isfinite(reference).all():
        raise ValueError("hypervolume requires finite inputs")
    front = pareto_front_2d(rewards[np.all(rewards > reference, axis=1)])
    if not len(front):
        return 0.0
    widths = np.diff(np.r_[reference[0], front[:, 0]])
    return float(np.dot(widths, front[:, 1] - reference[1]))


def generate(results, figure_dir=None):
    results = resolve_path(results)
    meta = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    x = candidate_grid(cfg["grid_points"])
    if not np.array_equal(np.load(results / "candidates.npy"), x):
        raise ValueError("saved candidate grid differs from declared grid")
    # Full rewards enter offline evaluation only, never search or its timers.
    pool_rewards = vlmop2(x)
    ref = cfg["reference_point"]
    grid_hv = hypervolume_2d(pool_rewards, ref)
    records, curves, all_rewards = [], {}, {}
    for method in cfg["methods"]:
        curves[method] = []
        for seed in cfg["seeds"]:
            stem = f"{method}_seed{seed}"
            rows = read_trace(results / "runs" / f"{stem}.csv")
            record = json.loads((results / "runs" / f"{stem}.json").read_text(encoding="utf-8"))
            validate_completion(record, cfg, method, seed, x)
            validate_trace(rows, cfg, method, seed, x)
            rewards = np.array([[float(r["reward1"]), float(r["reward2"])] for r in rows])
            hv = np.array([hypervolume_2d(rewards[:t], ref) for t in range(1, len(rows) + 1)])
            if np.any(np.diff(hv) < -1e-12) or np.any(hv > grid_hv + 1e-12):
                raise ValueError("hypervolume trajectory inconsistent with candidate pool")
            curves[method].append(hv)
            all_rewards[method, seed] = rewards
            records.append({"method": method, "seed": seed, "evaluations": len(rows),
                            "final_hypervolume": float(hv[-1]),
                            "normalized_hv_gap": float(max(0., (grid_hv - hv[-1]) / grid_hv)),
                            "search_runtime_s": record["search_runtime_s"]})
    n, summary = len(cfg["seeds"]), []
    ddof = 1 if n > 1 else 0
    cuts = sorted({t for t in (cfg["n_initial"], 20, 50, cfg["budget"]) if t <= cfg["budget"]})
    for method in cfg["methods"]:
        values = np.array(curves[method])
        runtimes = [r["search_runtime_s"] for r in records if r["method"] == method]
        for cut in cuts:
            hv = values[:, cut - 1]
            summary.append({
                "method": method, "budget": cut, "seeds": n,
                "mean_hypervolume": float(np.mean(hv)),
                "std_hypervolume": float(np.std(hv, ddof=ddof)),
                "mean_normalized_hv_gap": float(np.mean(np.maximum(0., (grid_hv - hv) / grid_hv))),
                "mean_full_search_runtime_s": float(np.mean(runtimes)),
                "std_full_search_runtime_s": float(np.std(runtimes, ddof=ddof)),
            })
    paired = []
    if "Random" in cfg["methods"]:
        for method in cfg["methods"]:
            if method == "Random":
                continue
            difference = np.array(curves[method])[:, -1] - np.array(curves["Random"])[:, -1]
            paired.append({"method": method, "baseline": "Random", "seeds": n,
                           "budget": cfg["budget"],
                           "mean_paired_hv_difference": float(np.mean(difference)),
                           "std_paired_hv_difference": float(np.std(difference, ddof=ddof)),
                           "better_seed_count": int(np.sum(difference > 1e-12))})
    write_table(results / "summary.csv", summary)
    write_table(results / "run_summary.csv", records)
    if paired:
        write_table(results / "paired_comparison.csv", paired)
    stats = {
        "task": "VLMOP2 (synthetic, dimensionless)", "candidate_count": len(x),
        "budget_including_initial": cfg["budget"], "n_initial": cfg["n_initial"],
        "seed_count": n, "completed_runs": len(records), "reference_point": ref,
        "full_grid_hypervolume": grid_hv, "summary": summary, "paired_comparison": paired,
        "timing": meta["timing"], "pareto_plot_seed": cfg["seeds"][0],
        "audit": "all configured trajectories, shared initial IDs, budgets, inputs and objectives verified",
        "limitations": "one synthetic problem; no materials performance or old/new version speed claim",
    }
    archive_existing(results / "statistics.json")
    write_json(results / "statistics.json", stats)
    figures = resolve_path(figure_dir) if figure_dir else ROOT / "figures" / cfg["name"]
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "axes.grid": True, "grid.alpha": .15,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    steps = np.arange(1, cfg["budget"] + 1)
    for method in cfg["methods"]:
        values = np.array(curves[method])
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=ddof)
        axes[0].plot(steps, mean, label=method, color=COLORS[method], linewidth=1.8)
        axes[0].fill_between(steps, np.maximum(0., mean - std), np.minimum(grid_hv, mean + std),
                             color=COLORS[method], alpha=.12)
        final = next(r for r in summary if r["method"] == method and r["budget"] == cfg["budget"])
        axes[1].errorbar(final["mean_full_search_runtime_s"], final["mean_hypervolume"],
                         yerr=final["std_hypervolume"], fmt="o", color=COLORS[method], capsize=3)
        axes[1].annotate(method, (final["mean_full_search_runtime_s"], final["mean_hypervolume"]),
                         xytext=(5, 7), textcoords="offset points", color=COLORS[method])
    axes[0].axhline(grid_hv, linestyle=":", color="#888888", label="Full grid (offline only)")
    axes[0].axvline(cfg["n_initial"], linestyle=":", color="#999999", linewidth=.8)
    axes[0].set(xlabel="Evaluations (including initialization)", ylabel="Hypervolume (higher is better)",
                title=f"VLMOP2 | {len(x):,} candidates | {n} seeds", xlim=(1, cfg["budget"]))
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xscale("log")
    axes[1].margins(x=.25, y=.25)
    axes[1].set(xlabel="Mean search overhead (seconds, log scale)", ylabel="Final hypervolume",
                title="Quality versus computation cost")
    fig.supxlabel("Bands/bars: one sample SD across seeds; bands clipped to feasible HV range", fontsize=8)
    save_figure(fig, figures, "quality_cost")

    count = len(cfg["methods"])
    fig, axes = plt.subplots(1, count, figsize=(3.1 * count, 3.4), layout="constrained", squeeze=False)
    # A preregistered first configured seed, not a best-looking run selected post hoc.
    seed = cfg["seeds"][0]
    grid_front = pareto_front_2d(pool_rewards)
    for ax, method in zip(axes[0], cfg["methods"]):
        points = all_rewards[method, seed]
        front = pareto_front_2d(points)
        ax.plot(grid_front[:, 0], grid_front[:, 1], color="#bbbbbb", linewidth=1, label="Full grid")
        ax.scatter(points[:, 0], points[:, 1], s=12, color="#999999", alpha=.55, label="Evaluated")
        ax.scatter(front[:, 0], front[:, 1], s=22, color=COLORS[method], label="Observed Pareto")
        ax.set(xlabel="Reward 1 (higher is better)", ylabel="Reward 2 (higher is better)",
               title=f"{method} | seed {seed}", xlim=(-1.02, .02), ylim=(-1.02, .02))
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"VLMOP2 trade-offs after {cfg['budget']} evaluations (fixed illustrative seed)")
    save_figure(fig, figures, "pareto_fronts")
    print(f"Audited {len(records)} VLMOP2 trajectories; full-grid HV={grid_hv:.6f}")
    for row in summary:
        if row["budget"] == cfg["budget"]:
            print(f"{row['method']}: HV {row['mean_hypervolume']:.6f} +/- "
                  f"{row['std_hypervolume']:.6f}; search {row['mean_full_search_runtime_s']:.3f}s")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/vlmop2")
    parser.add_argument("--figure-dir")
    args = parser.parse_args()
    generate(args.results, args.figure_dir)


if __name__ == "__main__":
    main()
