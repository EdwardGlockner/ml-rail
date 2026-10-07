import numpy as np
import torch

from evaluation import xtail_protocol as xp
from scripts.experiments.xtail_lora import ExpertCascade, LoRALinear, Strip
from tests.test_xtail_protocol import _toy_data


def _with_ids(data):
    names = list(data)

    def feat(d, split, idx=None):
        x = d[f"{split}_global"] if idx is None else d[f"{split}_global"][idx]
        pos = np.arange(len(x)) if idx is None else np.asarray(idx)
        j = names.index(next(n for n in names if data[n] is d))
        return np.concatenate([x, np.full((len(x), 1), j, np.float32), pos[:, None].astype(np.float32)], 1)
    return feat


def test_strip_is_transparent_and_expert_cascade_uses_expert_probs():
    data = _toy_data()
    plain = xp.run_stream(data, xp.DualRAIL(), seed=0)
    stripped = xp.run_stream(data, Strip(xp.DualRAIL()), seed=0, feat=_with_ids(data))
    assert np.allclose(plain, stripped)
    # perfect "experts" (one-hot on the true class) for every (domain, image): the last row reaches the
    # accuracy of routing alone, never lower than ML-RAIL's own routed prediction
    probs = {}
    for k, n in enumerate(data):
        for j, m in enumerate(data):
            y = data[m]["test_labels"]
            P = np.zeros((len(y), data[n]["n_classes"]), np.float32)
            if j == k:
                P[np.arange(len(y)), y] = 1
            probs[(k, j)] = (np.arange(len(y)), P)
    c = ExpertCascade(Strip(xp.DualRAIL()), probs, "E")
    table = xp.run_stream(data, c, seed=0, feat=_with_ids(data))
    assert c.misses == 0 and table[-1].mean() >= plain[-1].mean() - 1e-9


def test_lora_starts_as_identity():
    base = torch.nn.Linear(8, 4)
    l = LoRALinear(base)
    x = torch.randn(3, 8)
    assert torch.allclose(l(x), base(x))
