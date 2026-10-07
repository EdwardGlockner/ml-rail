import numpy as np

from scripts.experiments import run_xtail_spatial as sp


def test_grid_regions_tile_the_patch_grid():
    for n in (2, 3):
        cover = np.zeros((sp.GRID, sp.GRID), dtype=int)
        for a, b, c, e in sp.regions(n):
            cover[a:b, c:e] += 1
        assert (cover == 1).all()


def test_kmeans_and_part_blocks():
    rng = np.random.default_rng(0)
    centers = np.eye(4, 8, dtype=np.float32) * 5
    X = (centers[rng.integers(0, 4, 2000)] + 0.1 * rng.normal(size=(2000, 8))).astype(np.float32)
    C = sp.kmeans(X, 4, seed=0).cpu().numpy()
    assert all(np.min(np.linalg.norm(C - c, axis=1)) < 0.5 for c in centers)  # each true center found
    P = (centers[rng.integers(0, 4, (3, sp.GRID * sp.GRID))] + 0.1 * rng.normal(size=(3, sp.GRID * sp.GRID, 8))).astype(np.float16)
    means, (hist,) = sp.part_blocks(sp.torch.from_numpy(C).to(sp.xp.DEV), P)
    assert len(means) == 4 and means[0].shape == (3, 8)
    assert np.allclose(hist.sum(1), 1.0, atol=1e-5)


def test_poolings_shapes_and_mean_grid_consistency():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(3, 8)).astype(np.float32)
    P = rng.normal(size=(3, sp.GRID * sp.GRID, 8)).astype(np.float16)
    for kind in sp.POOLINGS:
        X = sp.features(kind, g, P)
        assert X.shape[0] == 3 and np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-4)
    # the 2x2 grid regions are equal-sized here, so their average equals the global patch mean
    grid = np.stack(sp.pool("grid2", g, P)).mean(0)
    assert np.allclose(grid, sp.pool("mean", g, P)[0], atol=1e-4)
