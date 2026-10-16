import numpy as np

from upgrade_study import (PairedParEGO, initialize, manifest, three_points,
                           run_job, audit)


def test_fixed_points_and_paired_weight_stream():
    three_points()
    a, b = PairedParEGO(11, 'min'), PairedParEGO(11, 'max')
    t = np.array([[-1., 0.], [0., -1.], [-.4, -.4]])
    for _ in range(4):
        a(t)
        b(t)
    assert a.calls == b.calls


def test_manifest_and_oracle_audit():
    jobs = manifest()
    assert len(jobs) == 120 and len({j['id'] for j in jobs}) == 120
    assert {j['block'] for j in jobs} == {'version', 'multi'}
    assert len(manifest(True)) == 6
    assert {j['seed'] for j in jobs} == set(range(10, 30))
    for block, count in [('version', 2), ('multi', 4)]:
        by_seed = [[j['method'] for j in jobs if j['block'] == block and j['seed'] == seed] for seed in range(10, 30)]
        for method in set(by_seed[0]):
            assert [sum(row[pos] == method for row in by_seed) for pos in range(count)] == [20 // count] * count
    job = next(j for j in manifest(True) if j['block'] == 'multi' and j['method'] == 'Random')
    record = run_job(job)
    assert audit(record)
    record['rows'][0]['reward'][0] += 1
    with np.testing.assert_raises(AssertionError):
        audit(record)


def test_target_censoring_and_initial_batch():
    from upgrade_report import first_hit, bootstrap
    assert first_hit([0.] * 50, .9) is None
    assert first_hit([1.] * 50, .9) == 10
    assert first_hit([0.] * 19 + [1.] * 31, .9) == 20
    assert bootstrap(np.zeros(20)) == {'mean': 0., 'low': 0., 'high': 0.}
