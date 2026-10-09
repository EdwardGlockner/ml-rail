# ML-RAIL: Training-Free Continual Learning for CLIP with Per-Domain Gating

[![tests](https://github.com/EdwardGlockner/ml-rail/actions/workflows/tests.yml/badge.svg)](https://github.com/EdwardGlockner/ml-rail/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A frozen CLIP model learns ten image domains in sequence from 16 examples per class, without gradient
training and without forgetting. It extends RAIL (NeurIPS 2024) [1] with the same inputs: the same
frozen backbone, shots, class names and prompt. On held-out seeds of the X-TAIL benchmark it reaches
**62.16 Transfer, 73.61 Average and 84.52 Last accuracy**, against 61.55 / 71.63 / 82.84 for RAIL in the
same harness. Transfer is the accuracy on domains not yet learned, and here it ends up above zero-shot
CLIP's.

The method has three parts:

- **ML-RAIL** gives RAIL's closed-form kernel ridge adapter a multi-layer view of each image, following
  LayUP [4].
- **A rescue gate** admits test images that RAIL's zero-shot router wrongly keeps away from the adapter.
- **A reverse gate** takes back images that the router wrongly sends to the adapter, which can only name
  classes already learned.

Both gates compare an image's similarity to each learned domain against thresholds that the domain sets
from the spread of its own training images.

Two caveats matter for reading the numbers:

- The hyperparameters were chosen on validation data from all ten domains; RAIL chooses its own on the
  first domain only. Under RAIL's full protocol, with the feature set, γ and λ all chosen on the first
  domain, ML-RAIL + rescue gate gains only +0.18 Average and −0.43 Last over RAIL. Keeping the five feature
  blocks and choosing only γ and λ on the first domain gives +1.25 / +0.33 (Table 3).
- The rescue gate alone costs Transfer, and at 5 shots it misses its own 0.2-point tolerance (−0.23 on the
  final seeds). The reverse gate more than recovers that cost in every setting.

<p align="center"><img src="docs/figures/xtail_method.png" width="92%"></p>
<p align="center"><sub><b>Figure 1.</b> Overview. Highlighted components are added to RAIL; the backbone is never trained.</sub></p>

## Results

X-TAIL (cross-domain task-agnostic incremental learning) [1] learns ten domains in sequence. After every
step, the model classifies test images from all domains among ~1,100 class names, with no domain label.

- **Transfer:** accuracy on domains not yet learned.
- **Average:** accuracy over all steps.
- **Last:** accuracy after the final step.

All numbers below are on seeds 11–15, run once after every setting was fixed. Seeds 6–10, an earlier
held-out run of everything except the reverse gate, give the same numbers to within 0.2 (in
[results/xtail_results.json](results/xtail_results.json)).

<p align="center"><img src="docs/figures/xtail_table_main.png" width="92%"></p>
<p align="center"><sub><b>Table 1.</b> X-TAIL, order I, 16 shots per class, CLIP ViT-B/16. Top: numbers as published (RAIL from arXiv:2406.18868v3; LADA quotes Dual-RAIL as 71.3 Average and reports Transfer under a different definition). Bottom: this repository's harness, mean ± std over seeds 11–15; Δ is the gain over RAIL run in the same harness. Rankings (<b>bold</b>: best, <u>underlined</u>: second) are within each block, since the two blocks come from different code. Transfer has no spread where unseen domains are classified by zero-shot CLIP alone.</sub></p>

- These numbers are above the published ones we found as of October 2026 (MoRAM, 63.3 Transfer; MoRAM
  and LADA, 72.7 Average; LADA, 83.1 Last), except for MoRAM's Transfer.
- That comparison crosses harnesses. Our run of Dual-RAIL gives 61.55 / 71.63 / 82.84, against 62.4 / 71.9
  / 82.4 published. RAIL's official code, with small compatibility fixes, gives 61.5 / 71.2 / 82.3.
- R²A²-MoE (2025), an analytic method that could compete, was not accessible to check.

<p align="center"><img src="docs/figures/xtail_table_settings.png" width="88%"></p>
<p align="center"><sub><b>Table 2.</b> The other task order and a 5-shot, 11-domain variant (CIFAR-100 added), with every setting kept from order I. Mean ± std over seeds 11–15. Published numbers exist for order II (RAIL, LADA) and for a 5-shot variant (arXiv:2601.22828), but they come from other code, so only the same-harness differences are compared here.</sub></p>

<p align="center"><img src="docs/figures/xtail_steps.png" width="52%"></p>
<p align="center"><sub><b>Figure 2.</b> Accuracy over all ten domains after each learning step (order I, mean of seeds 11–15; shaded: ±1 std).</sub></p>

### Effect of the tuning protocol

<p align="center"><img src="docs/figures/xtail_table_tuning.png" width="96%"></p>
<p align="center"><sub><b>Table 3.</b> ML-RAIL with hyperparameters chosen on all domains' validation data (as above) or on the first domain's only (RAIL's protocol, by validation regression error). Order I, seeds 6–10; this check was run before the reverse gate existed. The rescue gate's two settings stay at the values chosen on all domains, because measuring their Transfer cost needs domains not yet learned.</sub></p>

- Selecting the feature set on the first domain alone picks RAIL's final embedding with a wider kernel. That
  is *worse* than RAIL's published setting. The validation errors of all three feature sets are within 4%
  of each other on that one domain, so the selection is close to noise. The rescue gate also costs a full
  Transfer point with this kernel, because the wider kernel makes the thresholds less selective.
- With ML-RAIL's five blocks kept and only γ and λ chosen on the first domain, ML-RAIL gains +0.35 Average
  and +0.3 Last over RAIL. The rescue gate's gain on top of it is unchanged (+0.9).

## Method

**ML-RAIL.** Each image is encoded once by frozen CLIP ViT-B/16. Five blocks are taken from that single
forward pass: the final embedding, the layer-10 class token, the mean patch token, and GeM-pooled patch
tokens of layers 8 and 10. Each block is $\ell_2$-normalized and the blocks are concatenated into $x$. With
$X$ the stored training features of all learned domains and $Y$ their one-hot labels, the adapter is RAIL's
kernel ridge regression with an RBF kernel:

$$f(x) = k(x, X)\,(K + \lambda I)^{-1} Y, \qquad k(x, x') = \exp(-\gamma \lVert x - x' \rVert^2).$$

The solution is exact and does not depend on the order in which domains arrive, so nothing is forgotten.
The joint solve over all domains matters, because the other domains act as negatives. Separate per-domain
models collapse (Last 84.9 → 61.4). The multi-layer idea is LayUP's [4], proposed for
class-incremental learning with random-projection classifiers; here it is applied to RAIL's dual adapter.

**Per-domain thresholds.** For each learned domain $d$, take every training image's similarity to its
nearest neighbour among $d$'s other training images, with $s_d(x) = \max_{x' \in X_d} k(x, x')$. Two
percentiles of these values serve as thresholds: $\tau_d$ (25th) and $\rho_d$ (1st).

**Rescue gate.** RAIL applies the adapter only when zero-shot CLIP's top-1 class belongs to a learned
domain. With an oracle gate, Average would rise by 1.5 points, all from learned-domain images that zero-shot
CLIP confuses with a class not yet learned. The rescue gate also admits an image if:

- zero-shot CLIP puts at least $m = 0.1$ probability on learned classes, and
- $s_d(x) \geq \tau_d$ for some learned domain $d$.

**Reverse gate.** An admitted image of a domain not yet learned is wrong for certain, because the adapter
only scores learned classes. This happens to 4–5% of such images. An admitted image is taken back if
$s_d(x) < \rho_d$ for *every* learned domain $d$. It is then predicted as zero-shot CLIP's best class among
those not yet learned.

**Prediction.** Admitted images are scored as $0.2\,p_{zs} + 0.8\,f(x)$, as in RAIL; rejected images keep
$p_{zs}$. Last is unchanged by construction: after the final step every class is learned, so every image is
admitted and nothing can be taken back.

Neither gate carries a coverage guarantee. As a check, 76% of each domain's own test images clear its
$\tau_d$ (nominal 75%; per domain 71–79%; seeds 6–10).

## Analysis

**Why per-domain thresholds.** A single global similarity threshold only works when it is tuned on the
domains it is tested on. Tuned on half the domains and tested on the other half, it changes Average by
−1.4 to +0.65. Domains differ in how tightly their images cluster (Figure 3). Thresholds set per domain from
its own training images put all domains on one scale, and in the same split test they gave +0.55 to +0.86.
This check stays within X-TAIL's ten domains; a different benchmark could still behave differently.

<p align="center"><img src="docs/figures/xtail_thresholds.png" width="92%"></p>
<p align="center"><sub><b>Figure 3.</b> Nearest-neighbour similarity among each domain's training images, seed 11 (box: quartiles; whiskers: 5th–95th percentile). Diamonds: rescue thresholds τ<sub>d</sub>; triangles: reverse thresholds ρ<sub>d</sub>. Dashed: the best single global threshold, tuned on all ten domains' validation data.</sub></p>

**Effect of the gates** (Figure 4).

- **Rescue gate:** raises Average on every seed (+0.80 to +1.28 on the mean) at a Transfer cost. The cost
  is small with 16 shots (−0.03 and −0.13) but larger with 5 shots (−0.16 to −0.23 across seed sets). A
  plausible cause, not tested, is that each threshold is estimated from fewer images.
- **Reverse gate:** on top of the rescue gate, adds +0.73, +0.52 and +0.64 Transfer, and +0.09, +0.11 and
  +0.00 Average. ΔTransfer is positive on every seed in all three settings.
- **Both gates together:** above ML-RAIL on both metrics everywhere, with Transfer +0.41 to +0.61 and
  Average +0.92 to +1.28.
- **Sensitivity:** the rescue gate's settings are not sharp. Over $q \in \{0.25, 0.5\}$ and
  $m \in \{0.05, 0.1\}$ the Average gain is +0.69 to +0.92, and the Transfer cost −0.09 to −0.21.

**Headroom of the reverse gate.** An oracle that takes back exactly the admitted images of unseen domains
would add +1.9 Transfer on order I (development seeds). The reverse gate recovers about 40% of that on order
I and a quarter to a third in the other settings. Looser thresholds recover more Transfer, but start taking
back learned-domain images, and Average then drops: −0.6 at the 5th percentile, −1.9 at the 10th.

<p align="center"><img src="docs/figures/xtail_gate_per_seed.png" width="92%"></p>
<p align="center"><sub><b>Figure 4.</b> Change over ML-RAIL from the rescue gate alone and from both gates, per held-out seed 11–15 (bar: mean). (a) Average. (b) Transfer; dashed: the rescue gate's tolerance for its mean Transfer cost, set before its run.</sub></p>

## How the result was reached

- **Rules first:** before each experiment, its settings and a pass/fail rule were fixed in a short note
  (the notes are not part of this repository). Hyperparameters were chosen on CoOp validation splits.
- **Rescue gate, development seeds 0–5:** tested in two rounds.
  - The first version ($m = 0.05$) cost 0.27 Transfer points in the 5-shot setting and failed its rule.
  - The second round raised $m$ to 0.1 and still exceeded the 0.2 tolerance at 5 shots (−0.214).
  - Seed 1 was also used during gate selection, so the result was rerun on fresh seeds 6–10. There the
    5-shot cost was −0.164, but on seeds 11–15 it is −0.234. The rescue gate alone does not reliably meet
    its tolerance at 5 shots.
- **Reverse gate:**
  - It was motivated by an oracle measurement of where Transfer is lost.
  - $\rho_d$'s percentile (1st) is the loosest in the grid (0.1st to 10th) that kept Average from falling
    on validation.
  - It passed its rule (mean ΔTransfer ≥ +0.3, ΔAverage ≥ 0, ΔTransfer > 0 on every seed) on development
    seeds 1–5 in all three settings, and again on fresh seeds 11–15.
- **Replication:** RAIL was reproduced with its official code (61.5 / 71.2 / 82.3) and with this
  repository's independent harness before any comparison. The changes needed to run the official code are
  listed in [third_party/RAIL_CHANGES.md](third_party/RAIL_CHANGES.md).
- **Fair baselines:** any change that is not part of the method, such as prompts or features, was also
  applied to the baselines.
- **Precision:** the reported runs solve in float32. A float64 run of seed 6 (without the reverse gate) matches within
  0.05 points.

## Related work

- **RAIL** [1] introduced X-TAIL and the closed-form adapter with zero-shot routing that this work extends.
  **MoRAM** [2] and **LADA** [3] are the strongest published X-TAIL results, both gradient-trained.
- **LayUP** [4] showed that multi-layer features help rehearsal-free continual learning. **RanPAC** [5] and
  analytic learning (ACIL, GKEAL) [6] are the closed-form classifier family RAIL belongs to.
- **DIKI** [7] is the closest prior work to the rescue gate. It fits a Gaussian per task on frozen features
  and uses the likelihood to weight task knowledge against frozen CLIP. It is trained, evaluated on MTIL,
  and reports no X-TAIL numbers. The rescue gate differs in being training-free, nonparametric, and
  calibrated per domain from nearest-neighbour similarities.
- **LADA** [3] already observed that learned knowledge can improve Transfer by separating learned from
  unseen classes, and its own Transfer measure exceeds zero-shot CLIP's. It does so with trained modules.
  The reverse gate is a training-free, explicit version of that idea inside RAIL's fusion. MoRAM [2] also
  exceeds zero-shot Transfer, through continual adaptation.
- **kNN-based OOD detection** [8] scores test images by nearest-neighbour distance to training features.
  Kim et al. [9] link task-incremental prediction to out-of-distribution detection. Per-group quantile
  thresholds resemble **Mondrian conformal prediction** [10], but no coverage guarantee is claimed here.

## Limitations

- One backbone (CLIP ViT-B/16) and one benchmark family.
- The headline gain depends on tuning with all domains' validation data (Table 3), and that check predates
  the reverse gate.
- The rescue gate alone is unreliable at 5 shots; the combined method is not.
- The dual adapter stores all training features, as RAIL's does. A fixed-memory version with random
  features comes within 0.3 Average and 0.5 Last.
- The rescue gate recovers about 60% of its oracle's Average headroom; the reverse gate, a quarter to two
  fifths of its oracle's Transfer headroom.

## Future work

- **More Transfer.** The reverse gate stops at the point where it starts taking back learned-domain images.
  A better signal than nearest-neighbour similarity, for example one that also uses the zero-shot
  distribution, could recover more of the +1.9 ceiling. Beyond that ceiling, Transfer is CLIP's own error
  on unseen classes. A prior learned from past domains, tried earlier, only tied CLIP's own class-name
  embeddings ([research log](docs/RESEARCH_LOG.md)).
- **Thresholds with few shots.** Shrinking per-domain thresholds toward a pooled estimate may reduce the
  rescue gate's 5-shot Transfer cost.
- **Tuning on the first domain.** A selection criterion that is less noisy on a single domain than
  regression error.
- **Other backbones and longer streams.** CLIP ViT-L/14 or SigLIP, and streams of 20 or more domains for
  the fixed-memory adapter.

## Project history

The project began as *Plastic Primitive Networks*: a gradient-free dictionary of visual primitives that
creates, merges, prunes and splits its own entries on a frozen backbone. It beat fixed and online
dictionaries but lost to properly tuned published continual learners, so it was stopped by its own kill
criteria; its code is not part of this repository. The work then moved to X-TAIL.

- [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md): the full path, including negative results.
- [docs/REPORT_XTAIL.md](docs/REPORT_XTAIL.md): the technical report.

## Reproducing

```bash
pip install -r requirements.txt      # Python 3.11; install a GPU build of torch first if you have one
export XTAIL_ROOT=/path/to/storage   # data and feature caches (default: ./xtail_data, ~50 GB)

python scripts/reproduce.py                  # every step below, in order; resumes if interrupted
python scripts/reproduce.py --from evaluate  # once the data and features are cached
```

| Step | Script |
|:--|:--|
| Download and verify the datasets | `scripts/setup_xtail_data.py` |
| Cache CLIP ViT-B/16 features, then the intermediate layers | `scripts/xtail_encode.py`, `scripts/xtail_encode_extra.py` |
| Tables 1–2 on seeds 11–15, and the seeds 6–10 replication | `scripts/evaluate.py` |
| Table 3 and the rescue gate's checks | `scripts/experiments/run_xtail_first_domain_tuning.py`, `run_xtail_gate_sensitivity.py` |
| Reverse gate: headroom and development runs | `scripts/experiments/run_xtail_transfer_headroom.py` |
| Figures, tables, `results/xtail_results.json` | `scripts/plot_xtail_figures.py` |

The per-seed metrics and accuracy tables behind Tables 1–3 are in
[results/xtail_results.json](results/xtail_results.json), so the reported numbers can be checked without the
datasets. Run the tests with `python -m pytest tests/ -q`.

```text
evaluation/ml_rail.py          ML-RAIL: feature blocks and the adapter
evaluation/xtail_protocol.py   X-TAIL protocol and metrics, RAIL adapters, DomainThresholdGate (both gates), run_stream
evaluation/xtail_variants.py   variants that were tested and did not help
evaluation/xtail_prompts.py    zero-shot text heads
data/real/xtail.py             dataset loaders and XTAIL_ROOT
scripts/                       the pipeline above; run_xtail.py runs single methods (RAIL replication)
scripts/experiments/           development experiments (table below)
results/                       per-seed results
tests/                         unit tests
third_party/                   CLIP's prompt templates (MIT) and the RAIL code changes
```

Development scripts keep the labels their cached results use: `Aprime` is ML-RAIL, `conformal` is the
rescue gate (at $m = 0.05$ in `xtail_lora.py`), and `C0`, `A1`, … name variants in `run_xtail_analysis.py`.

| Development experiment (seeds 0–5 and validation) | Script (in `scripts/experiments/`) |
|:--|:--|
| Hyperparameter and feature selection on validation | `run_xtail_tune.py`, `run_xtail_tune_val4.py`, `run_xtail_improve.py` |
| Zero-shot text heads | `run_xtail_zeroshot.py` |
| Oracle gate headroom, per-domain variants, looser gates | `run_xtail_analysis.py` |
| Signals for gating and the split-domain check | `run_xtail_gatesep.py`, `run_xtail_gatesep_explore.py` |
| Rescue gate selection on development seeds | `../run_xtail_routegate.py` |
| Transfer headroom and the reverse gate | `run_xtail_transfer_headroom.py` |
| Gate-free residual model | `run_xtail_residual.py` |
| Spatial readouts for fine-grained domains | `run_xtail_spatial.py`, `xtail_encode_patches.py`, `xtail_encode_layerpatch.py`, `xtail_encode_tiles.py` |
| Fixed-memory adapter (random features) | `run_xtail_bounded.py` |
| Per-domain LoRA experts | `xtail_lora.py` |

## References

1. Xu et al. *Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models* (RAIL). NeurIPS 2024; arXiv:2406.18868.
2. *Little by Little: Continual Learning via Incremental Mixture of Rank-1 Associative Memory Experts* (MoRAM). arXiv:2506.21035, 2025.
3. *LADA: Scalable Label-Specific CLIP Adapter for Continual Learning*. ICML 2025; arXiv:2505.23271.
4. Ahrens et al. *Read Between the Layers: Leveraging Multi-Layer Representations for Rehearsal-Free Continual Learning with Pre-Trained Models* (LayUP). arXiv:2312.08888.
5. McDonnell et al. *RanPAC: Random Projections and Pre-trained Models for Continual Learning*. NeurIPS 2023.
6. Zhuang et al. *ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection*. NeurIPS 2022; *GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task*. CVPR 2023.
7. Tang et al. *Mind the Interference: Retaining Pre-trained Knowledge in Parameter Efficient Continual Learning of Vision-Language Models* (DIKI). ECCV 2024; arXiv:2407.05342.
8. Sun et al. *Out-of-Distribution Detection with Deep Nearest Neighbors*. ICML 2022.
9. Kim et al. *A Theoretical Study on Solving Continual Learning*. NeurIPS 2022.
10. Vovk et al. *Mondrian Confidence Machine*. 2003; see Vovk, Gammerman and Shafer, *Algorithmic Learning in a Random World*.

## License

MIT; see [LICENSE](LICENSE). Third-party notices are in [third_party/README.md](third_party/README.md).
