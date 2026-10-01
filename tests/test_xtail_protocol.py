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
