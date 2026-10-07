# Running RAIL's official code

The replication of RAIL (Xu et al., NeurIPS 2024) used the authors' code,
[linghan1997/Regression-based-Analytic-Incremental-Learning](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning)
(master branch, fetched 2026-10-01). That repository has no licence, so its code is not redistributed here.
These are the changes needed to run it on a current software stack and to log every X-TAIL metric. None
of them changes the arithmetic of the method.

| File | Change | Why |
|:--|:--|:--|
| `configs/analytic_clip.yaml` | `eval_adapter: 1` → `0` | Evaluate every domain at every step; Transfer and Average need the full table. |
| `primal_RAIL.py`, `utils.py` (`fit` of the kernel model) | `np.mat(A).I` → `np.linalg.inv(A)` | `np.mat` was removed in NumPy 2. |
| `utils.py` (`accuracy`) | `float(x.sum(0, keepdim=True).cpu().numpy())` → `float(x.sum().item())` | NumPy 2 refuses `float()` of a one-element array. |
| `utils.py` (`kernel`) | Fill a preallocated kernel matrix instead of growing it with `torch.cat` | Same values; the repeated concatenation ran out of GPU memory at step 10. |
| `dual_RAIL.py`, `primal_RAIL.py` (test loop) | Encode each test set once and reuse the features at every step | CLIP is frozen and test preprocessing is fixed, so the features are identical; this only saves time. |
| `scenario_datasets/collections.py` (MNIST) | Point `location` to your MNIST folder (here `$XTAIL_ROOT/datasets/mnist`) | The default is the authors' local path. |
| `scenario_datasets/utils.py`, `scenario_datasets/collections.py` | `num_workers` → `0` | On Windows, worker processes re-import RAIL's top-level scripts. Results are unaffected. |

With these changes, the official code gives 61.5 / 71.2 / 82.3 (Transfer / Average / Last) for Dual-RAIL
on X-TAIL order I, 16-shot. The RAIL paper (arXiv:2406.18868v3, Table 1) reports 62.4 / 71.9 / 82.4.
