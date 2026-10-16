import copy
import json

import numpy as np
import pytest

import multiobjective as mo
from benchmark import initial_ids, read_trace, standardize_inputs


def test_vlmop2_direction_and_known_optima():
    a = 1 / np.sqrt(2)
    y = mo.vlmop2(np.array([[a, a], [-a, -a], [0., 0.]]))
    assert np.allclose(y[0], [0., np.exp(-4.) - 1])
    assert np.allclose(y[1], [np.exp(-4.) - 1, 0.])
    assert np.allclose(y[2], [np.exp(-1.) - 1] * 2)


def test_function_oracle_invalid_batches_do_not_charge_budget():
    o = mo.FunctionOracle(mo.candidate_grid(3), 3, "Random", 0)
    with pytest.raises(RuntimeError, match="repeated"):
        o(np.array([0, 0]))
    with pytest.raises(ValueError):
        o(np.array([0.5]))
    with pytest.raises(IndexError):
        o(np.array([9]))
    assert not o.trace
    o(np.array([0, 1]))
    with pytest.raises(RuntimeError, match="budget"):
        o(np.array([2, 3]))
    with pytest.raises(RuntimeError, match="repeated"):
        o(np.array([0]))
    assert len(o.trace) == 2


def test_actual_v3_policies_share_initial_points_and_exact_budget(tmp_path):
    cfg = mo.load_config("configs/vlmop2_smoke.json")
    cfg.update(grid_points=7, n_initial=4, budget=8)
    out = tmp_path / "v3"
    mo.execute(cfg, out)
    x = mo.candidate_grid(7)
    traces = {}
    for method in cfg["methods"]:
        traces[method] = read_trace(out / "runs" / f"{method}_seed0.csv")
        mo.validate_trace(traces[method], cfg, method, 0, x)
        assert len(traces[method]) == 8
    shared = [r["candidate_id"] for r in traces["Random"][:4]]
    assert all([r["candidate_id"] for r in trace[:4]] == shared for trace in traces.values())
    meta = json.loads((out / "metadata.json").read_text())
    pools = [p for p in meta["environment"]["threadpools"] if p["user_api"] == "blas"]
    assert len(pools) >= 2 and all(p["num_threads"] == 1 for p in pools)
    # Resume validates records and preserves completed trajectories.
    raw = (out / "runs" / "NDS_seed0.csv").read_bytes()
    mo.execute(cfg, out, resume=True)
    assert (out / "runs" / "NDS_seed0.csv").read_bytes() == raw
    broken = copy.deepcopy(traces["ParEGO"])
    broken[-1]["reward1"] = str(float(broken[-1]["reward1"]) + .1)
    with pytest.raises(ValueError, match="ground truth"):
        mo.validate_trace(broken, cfg, "ParEGO", 0, x)
    (out / "runs" / "ParEGO_seed0.csv").write_text("missing", encoding="utf-8")
    with pytest.raises(ValueError, match="trace length"):
        mo.execute(cfg, out, resume=True)


def test_random_multiobjective_trace_repeats():
    cfg = mo.load_config("configs/vlmop2_smoke.json")
    cfg.update(grid_points=5, n_initial=3, budget=10)
    x = mo.candidate_grid(5)
    z, _, _ = standardize_inputs(x)
    initial = initial_ids(3, len(x), 3)
    paths = []
    for _ in range(2):
        oracle = mo.FunctionOracle(x, 10, "Random", 3)
        rows, _ = mo.run_method("Random", 3, z, oracle, initial, cfg)
        paths.append([r["candidate_id"] for r in rows])
    assert paths[0] == paths[1]


def test_hypervolume_matches_known_rectangle_union_and_ignores_dominated_points():
    from multiobjective_report import hypervolume_2d, pareto_front_2d
    points = np.array([[.5, .8], [.8, .5], [.2, .2], [.5, .8], [-.1, .9]])
    assert hypervolume_2d(points, (0., 0.)) == pytest.approx(.55)
    assert hypervolume_2d(np.empty((0, 2))) == 0.
    assert hypervolume_2d([[-1., 0.], [0., -1.]]) == 0.
    assert hypervolume_2d([[-.5, -.2], [-.2, -.5]]) == pytest.approx(.55)
    assert len(pareto_front_2d(points)) == 3  # one below reference remains non-dominated


def test_report_audits_all_seeds_before_aggregating(tmp_path):
    from multiobjective_report import generate, hypervolume_2d
    cfg = mo.load_config("configs/vlmop2_smoke.json")
    cfg.update(grid_points=5, n_initial=3, budget=6, methods=["Random"], seeds=[0, 1])
    out = tmp_path / "report-check"
    mo.execute(cfg, out)
    stats = generate(out, tmp_path / "figures")
    assert stats["completed_runs"] == 2
    for seed in cfg["seeds"]:
        rows = read_trace(out / "runs" / f"Random_seed{seed}.csv")
        ids = np.array([int(r["candidate_id"]) for r in rows])
        expected = hypervolume_2d(mo.vlmop2(mo.candidate_grid(5)[ids]))
        run = read_trace(out / "run_summary.csv")[seed]
        assert float(run["final_hypervolume"]) == pytest.approx(expected)
    assert (tmp_path / "figures" / "quality_cost.pdf").exists()
    (out / "runs" / "Random_seed1.json").unlink()
    with pytest.raises(FileNotFoundError):
        generate(out, tmp_path / "figures")
