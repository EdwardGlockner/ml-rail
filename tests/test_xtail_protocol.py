import numpy as np
import torch

from evaluation import xtail_protocol as xp


def _rail_metrics(table):
    """RAIL's own code (primal_RAIL.py / dual_RAIL.py, end of file)."""
    upper = np.triu(table, k=1)
    masked = np.ma.masked_equal(upper, 0)
    return float(np.mean(np.mean(masked, axis=0))), float(np.mean(np.mean(table, axis=0))), float(np.mean(table[-1, :]))


def test_metrics_match_rail_code():
    rng = np.random.default_rng(0)
    table = rng.uniform(1, 100, size=(10, 10))
    m = xp.metrics(table)
    t, a, last = _rail_metrics(table)
    assert np.isclose(m["transfer"], t) and np.isclose(m["average"], a) and np.isclose(m["last"], last)


def _toy_data(n_domains=3, n_cls=4, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    data, offset = {}, 0
    for k in range(n_domains):
        centers = rng.normal(size=(n_cls, dim))
        text = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        def sample(n):
            y = np.repeat(np.arange(n_cls), n)
            x = centers[y] + 0.3 * rng.normal(size=(len(y), dim))
            return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), y
        tr_x, tr_y = sample(20)
        te_x, te_y = sample(10)
        data[f"d{k}"] = {"text": text.astype(np.float32), "train_global": tr_x, "train_labels": tr_y,
                         "test_global": te_x, "test_labels": te_y, "classnames": np.arange(n_cls),
                         "offset": offset, "n_classes": n_cls}
        offset += n_cls
    return data


def test_fusion_preserves_zero_shot_on_unseen_domains():
    """Zero-shot accuracy on a domain not yet learned must be unchanged by RAIL's fusion rule."""
    data = _toy_data()
    zs_table = xp.run_stream(data, xp.ZeroShot(), seed=0)
    for method in (xp.PrimalRAIL(width=256, in_dim=16), xp.DualRAIL()):
        table = xp.run_stream(data, method, seed=0)
        for t in range(3):
            for d in range(t + 1, 3):
                assert np.isclose(table[t, d], zs_table[t, d])


def test_sum_kernel_with_one_block_equals_dual_and_calibration_gives_probabilities():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(40, 8)).astype(np.float32), rng.integers(0, 4, 40)
    Xq = rng.normal(size=(10, 8)).astype(np.float32)
    a, b = xp.DualRAIL(gamma=0.3, lam=1e-2, dtype=torch.float64), xp.SumKernelDual([(0, 8)], gamma=0.3, lam=1e-2, dtype=torch.float64)
    a.add(X, y)
    b.add(X, y)
    assert torch.allclose(a.scores(Xq), b.scores(Xq), atol=1e-5)
    c = xp.Calibrated(xp.DualRAIL(gamma=0.3, lam=1e-2), temperature=0.05)
    c.add(X, y)
    p = c.scores(Xq)
    assert torch.allclose(p.sum(1), torch.ones(10), atol=1e-5) and (p >= 0).all()


def test_rff_ridge_approaches_dual_kernel_ridge():
    """With many random Fourier features, the bounded-memory model's scores approach DualRAIL's."""
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(60, 6)).astype(np.float32), rng.integers(0, 3, 60)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xq = rng.normal(size=(20, 6)).astype(np.float32)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    d = xp.DualRAIL(gamma=0.5, lam=1e-1, dtype=torch.float64)
    d.add(X[:30], y[:30])
    d.add(X[30:], y[30:])
    r = xp.RFFRidge(6, width=20000, gamma=0.5, lam=1e-1, seed=0)
    r.add(X[:30], y[:30])  # added in two parts: accumulation must equal fitting all at once
    r.add(X[30:], y[30:])
    sd, sr = d.scores(Xq), r.scores(Xq)
    assert torch.corrcoef(torch.stack([sd.flatten(), sr.flatten()]))[0, 1] > 0.98
    assert (sd.argmax(1) == sr.argmax(1)).float().mean() > 0.9


def test_krr_loo_matches_refitting_without_each_point():
    rng = np.random.default_rng(4)
    X = torch.as_tensor(rng.normal(size=(25, 5)))
    Y = torch.nn.functional.one_hot(torch.as_tensor(rng.integers(0, 3, 25)), 3).double()
    K = torch.exp(-0.3 * torch.cdist(X, X).pow(2))
    evals, Q = torch.linalg.eigh(K)
    loo, alpha = xp.krr_loo(evals, Q, Y, 0.05)
    assert torch.allclose(alpha, torch.linalg.solve(K + 0.05 * torch.eye(25, dtype=torch.float64), Y), atol=1e-8)
    for i in range(25):
        keep = torch.arange(25) != i
        a = torch.linalg.solve(K[keep][:, keep] + 0.05 * torch.eye(24, dtype=torch.float64), Y[keep])
        assert torch.allclose(loo[i], K[i, keep] @ a, atol=1e-8)


def test_per_domain_krr_equals_separate_dual_models_and_oracle_gate_runs():
    """With one fixed setting, PerDomainKRR's scores are the per-domain DualRAIL models' scores side by side."""
    rng = np.random.default_rng(5)
    Xa, Xb = rng.normal(size=(30, 8)).astype(np.float32), rng.normal(size=(20, 8)).astype(np.float32)
    ya, yb = rng.integers(0, 3, 30), 3 + rng.integers(0, 2, 20)
    ya[:3], yb[:2] = [0, 1, 2], [3, 4]  # every class present
    Xq = rng.normal(size=(7, 8)).astype(np.float32)
    m = xp.PerDomainKRR([(0, 4), (4, 8)], gammas=(0.3,), lams=(1e-2,), weights=(0.5, 0.8))
    m.set_text(rng.normal(size=(5, 8)))
    m.add(Xa, ya)
    m.add(Xb, yb)
    s = m.scores(Xq)
    for X, y, cols in ((Xa, ya, slice(0, 3)), (Xb, yb - 3, slice(3, 5))):
        d = xp.DualRAIL(gamma=0.3, lam=1e-2, dtype=torch.float64)
        d.add(X, y)
        assert torch.allclose(s[:, cols], d.scores(Xq), atol=1e-4)
    assert m.fusion_weight_for(np.array([0, 4])).shape == (2,)
    data = _toy_data()
    for gate in ("oracle", "oracle_domain"):
        table = xp.run_stream(data, xp.DualRAIL(), seed=0, gate=gate)
        assert table.shape == (3, 3) and (table >= 0).all()


def test_joint_loo_weight_equals_dual_scores():
    """A1w's model is ML-RAIL's joint kernel ridge; only the fusion weight is chosen per domain."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y = np.concatenate([rng.integers(0, 3, 20), 3 + rng.integers(0, 2, 20)])
    y[:3], y[20:22] = [0, 1, 2], [3, 4]
    m, d = xp.JointLOOWeight(gamma=0.3, lam=1e-2), xp.DualRAIL(gamma=0.3, lam=1e-2)
    for sl, k in ((slice(0, 20), 3), (slice(20, 40), 5)):
        m.set_text(rng.normal(size=(k, 8)))
        m.add(X[sl], y[sl])
        d.add(X[sl], y[sl])
    Xq = rng.normal(size=(6, 8)).astype(np.float32)
    assert torch.allclose(m.scores(Xq), d.scores(Xq).float(), atol=1e-4)
    assert len(m.chosen) == 2 and m.fusion_weight_for(np.arange(5)).shape == (5,)


def test_looser_gates_reduce_to_rail_gate():
    data = _toy_data()
    rail = xp.run_stream(data, xp.DualRAIL(), seed=0)
    for gate in ("mass:1.01", "topk:1"):
        assert np.allclose(xp.run_stream(data, xp.DualRAIL(), seed=0, gate=gate), rail)
    loose = xp.run_stream(data, xp.DualRAIL(), seed=0, gate="mass:0.0")
    assert np.allclose(loose[-1], rail[-1])  # every class learned at the last step: identical


def test_conformal_gate_extends_rail_gate_and_route_runs():
    data = _toy_data()
    rail = xp.run_stream(data, xp.DualRAIL(), seed=0)
    same = xp.run_stream(data, xp.DualRAIL(), seed=0, gate=xp.ConformalGate(0.5, min_mass=2.0))  # rescues nothing
    assert np.allclose(same, rail)
    loose = xp.run_stream(data, xp.DualRAIL(), seed=0, gate=xp.ConformalGate(0.5))
    assert np.allclose(loose[-1], rail[-1])
    routed = xp.run_stream(data, xp.DualRAIL(), seed=0, route=True)
    assert np.allclose(np.triu(routed, 1), np.triu(rail, 1))  # unseen-domain cells: unchanged


def test_cascade_with_self_as_expert_keeps_the_routers_top_class():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y = np.concatenate([rng.integers(0, 3, 20), 3 + rng.integers(0, 2, 20)])
    y[:3], y[20:22] = [0, 1, 2], [3, 4]
    m = xp.DualRAIL(gamma=0.3, lam=1e-2)
    c = xp.Cascade(m, m)
    c.add(X[:20], y[:20])
    c.add(X[20:], y[20:])
    Xq = rng.normal(size=(9, 8)).astype(np.float32)
    s, r = c.scores(Xq), m.scores(Xq)
    assert (s.argmax(1) == r.argmax(1)).all()
    routed_cols = (s > -10).sum(1)
    assert set(routed_cols.tolist()) <= {3, 2}


def test_primal_rail_equals_direct_ridge():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(50, 16)).astype(np.float32), rng.integers(0, 5, 50)
    m = xp.PrimalRAIL(width=64, reg=0.1, in_dim=16)
    m.add(X[:25], y[:25])
    m.add(X[25:], y[25:])
    H = torch.relu(torch.from_numpy(X).to(xp.DEV) @ m.W).double()
    Y = torch.nn.functional.one_hot(torch.as_tensor(y, device=xp.DEV), int(y.max()) + 1).double()
    beta = torch.linalg.solve(H.T @ H + 0.1 * torch.eye(64, device=xp.DEV, dtype=torch.float64), H.T @ Y)
    assert torch.allclose(m.beta.double(), beta, atol=1e-4)
