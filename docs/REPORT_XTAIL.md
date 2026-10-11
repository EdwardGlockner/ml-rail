# ML-RAIL with per-domain gating: training-free continual learning for CLIP on X-TAIL

*Technical report, 2026-10-12. The path to this result, including the negative results, is in
[RESEARCH_LOG.md](RESEARCH_LOG.md). Per-seed numbers are in
[results/xtail_results.json](../results/xtail_results.json).*

## Summary

X-TAIL is cross-domain task-agnostic incremental learning, here with CLIP ViT-B/16, 10 domains and 16 shots
per class. On this benchmark, a training-free extension of RAIL uses the same inputs as RAIL: one frozen
backbone, the shots, the class names, and RAIL's single prompt "a photo of a {}.". The hyperparameters,
however, were tuned on all domains' validation data (see qualification 1). On held-out seeds 16–20 it
reaches **Transfer 62.75 / Average 73.76 / Last 84.53**. In the same harness, RAIL's Dual-RAIL gives
61.55 / 71.71 / 82.94, and zero-shot CLIP's Transfer is 61.55. The best published numbers we found are
63.3 (MoRAM) / 72.7 (MoRAM, LADA) / 83.1 (LADA). Those come from other code, so the comparison is
indicative only.

The method has three parts:

1. **ML-RAIL** runs RAIL's dual adapter on multi-layer frozen features: +0.98 Average, +1.59 Last.
2. **A rescue gate** admits images that RAIL's zero-shot gate rejects but that lie close to a learned
   domain: +0.89 Average, −0.13 Transfer.
3. **A reverse gate** takes back admitted images whose layer-10 class token is far from every learned
   domain, and predicts them among the classes not yet learned: +1.32 Transfer and +0.18 Average on top of
   the rescue gate. That is about 70% of what a perfect take-back could add.

Last is unchanged by both gates.

Three qualifications:

1. **Tuning protocol.** The hyperparameters were chosen on validation data from all ten domains; RAIL tunes
   on the first domain only (check on seeds 6–10).
   - Under RAIL's protocol with the five feature blocks kept, the full method gives 62.75 / 73.04 / 83.13.
     That is +1.20 Transfer, +1.42 Average and +0.33 Last over RAIL, instead of +1.20 / +2.11 / +1.67.
   - Under RAIL's full protocol, with the feature set also chosen on the first domain, the selection falls
     back to RAIL's final embedding with a wide kernel. The class token is then not among the features,
     and the gain is gone: −0.82 / +0.06 / −0.43 with a full-kernel reverse gate.
2. **Development history.**
   - The rescue gate's rule was revised once after a failed 5-shot result, and one test seed (seed 1) was
     used during its selection.
   - The reverse gate first used the full kernel, which passed on fresh seeds 11–15. A later study of
     signals found the layer-10 class token.
   - For that signal, the threshold criterion was changed after seeing development seeds (see "Reverse
     gate").
   - Every reported number comes from fresh seeds 16–20, run once with all settings fixed. Seeds 6–10 and
     11–15 are earlier held-out runs and agree to within 0.2 for every shared method.
3. **The rescue gate alone at 5 shots** costs more Transfer than its pre-set tolerance on two of four seed
   sets. The combined method is above ML-RAIL in Transfer on every seed.

## Setting and protocol

- **X-TAIL** (Xu et al., 2024) learns 10 domains in sequence. Order I is Aircraft, Caltech101, DTD,
  EuroSAT, Flowers, Food101, MNIST, OxfordPets, StanfordCars and SUN397. After each step, every domain's
  test set is classified over the union of all ~1,100 class names, without domain hints. The metrics
  follow RAIL's code exactly:
  - Transfer: mean accuracy on domains not yet learned.
  - Average: mean over all steps and domains.
  - Last: mean after the final step.
- **Replication.** RAIL's official code on our data gives 61.5 / 71.2 / 82.3. The paper reports 62.4 /
  71.9 / 82.4 (arXiv:2406.18868v3); LADA quotes 71.3 Average. Our harness's Dual-RAIL gives 61.55 / 71.71
  / 82.94 on seeds 16–20. The changes needed to run the official code are in
  [third_party/RAIL_CHANGES.md](../third_party/RAIL_CHANGES.md).
- **Splits and seeds.**
  - Hyperparameters were chosen on CoOp validation splits (sampling seed 100).
  - Development test runs used seeds 0–5.
  - Seeds 6–10 and 11–15 were earlier held-out runs, before each version of the reverse gate.
  - Seeds 16–20 are the final run of every method, made once after every setting was fixed.
- **Rules.** Before each experiment, its settings and a pass/fail rule were fixed in a short note; the
  notes are not included. Baselines get every change that is not part of the method.
- **Precision.** Float32 solves throughout. Float64 on seed 6 (without the reverse gate) matches within
  0.05 points.

## Method

**ML-RAIL (adapter).** RAIL's dual form: kernel ridge regression over all stored training features.

- Five blocks come from the same forward pass: the final embedding, the layer-10 class token, the mean
  patch token, and GeM-pooled layers 8 and 10. Each block is unit-normalized, and the concatenation is
  rescaled to unit length.
- RBF kernel exp(−γ‖x − x′‖²) with γ = 0.2 and λ = 1e−3.
- Fusion follows RAIL: p ← 0.2 p_zs + 0.8 f(x) for admitted images.
- There is one joint solve over all learned domains. Per-domain models collapse (Last 84.9 → 61.4),
  because they never see other domains as negatives.
- The multi-layer idea is LayUP's (Ahrens et al., arXiv:2312.08888). ML-RAIL is a stronger baseline, not a
  new mechanism.

**Per-domain thresholds** (`DomainThresholdGate`). When domain d arrives, compute for each of its training
images the highest similarity to d's other training images, excluding exact duplicates. Two percentiles of
these values are d's thresholds:
- τ_d, the 25th percentile of kernel similarities;
- ρ_d, the 0.8th percentile of cosine similarities between layer-10 class tokens.

For a test image, s_d(x) and c_d(x) are its highest kernel and class-token similarities to d's stored
images.

**Rescue gate.**

- An image is admitted if RAIL's rule holds (its zero-shot top-1 class is learned), or if both of these
  hold:
  - zero-shot puts at least m = 0.1 probability on learned classes;
  - s_d(x) ≥ τ_d for some learned domain d.
- The two settings were chosen on order-I validation. The thresholds use only data the method already
  stores.
- There is no coverage guarantee. The realized acceptance rate is close to nominal: 76.0% of each domain's
  own test images clear τ_d (nominal 75%, per domain 71–79%; seeds 6–10). At q = 0.5 it is 51.0%.

**Reverse gate** (`reverse_q=0.008`, `reverse_slice` = the layer-10 class token).

- An admitted image of a domain not yet learned is always wrong, because the adapter scores only learned
  classes. This happens to 4–5% of such images.
- An admitted image is taken back if c_d(x) < ρ_d for every learned domain d. Its prediction is zero-shot
  CLIP's best class among those not yet learned.
- It is inactive at the last step, when every class is learned.
- The percentile was chosen on validation as the one with the largest ΔTransfer + ΔAverage over the
  rescue gate alone. This criterion was adopted after seeing development results; see "Where Transfer is
  lost".

**Last is unchanged by construction.** After the final step every class is learned, so RAIL's rule admits
every image and nothing can be taken back.

**Why per-domain thresholds.**

- With an oracle gate, Average would rise by +1.5 on order I, all from learned-domain images that
  zero-shot CLIP confuses with a not-yet-learned class.
- A single global similarity threshold captures that headroom when tuned on the same ten domains. It does
  not carry over: tuned on half the domains and tested on the other half, it gives −1.4 to +0.65 across
  three splits, because domains differ in how tightly their images cluster.
- Per-domain thresholds gave +0.55 / +0.81 / +0.86 in the same splits.
- This held-out-domain check stays within X-TAIL's ten domains.

## Results (held-out seeds 16–20, mean ± std)

| Setting | Method | Transfer | Average | Last |
|---|---|---|---|---|
| Order I, 16-shot | Zero-shot CLIP | 61.55 | 57.81 | 57.81 |
| | Dual-RAIL (our harness) | 61.55 | 71.71 ± 0.09 | 82.94 ± 0.10 |
| | ML-RAIL | 61.55 | 72.69 ± 0.11 | 84.53 ± 0.17 |
| | + rescue gate | 61.42 | 73.58 ± 0.11 | 84.53 ± 0.17 |
| | + reverse gate, full kernel (earlier version) | 62.05 ± 0.16 | 73.67 ± 0.11 | 84.53 ± 0.17 |
| | **+ reverse gate, class token** | **62.75 ± 0.01** | **73.76 ± 0.08** | **84.53 ± 0.17** |
| Order II, 16-shot | Zero-shot CLIP | 57.06 | 57.90 | 57.90 |
| | Dual-RAIL (our harness) | 57.06 | 67.87 ± 0.04 | 82.88 ± 0.13 |
| | ML-RAIL | 57.06 | 68.55 ± 0.03 | 84.58 ± 0.07 |
| | + rescue gate | 57.02 ± 0.01 | 69.35 ± 0.03 | 84.58 ± 0.07 |
| | + reverse gate, full kernel (earlier version) | 57.54 ± 0.03 | 69.49 ± 0.05 | 84.58 ± 0.07 |
| | **+ reverse gate, class token** | **57.83 ± 0.02** | **69.73 ± 0.03** | **84.58 ± 0.07** |
| 11 domains, 5-shot | Zero-shot CLIP | 57.84 | 54.78 | 54.78 |
| | Dual-RAIL (our harness) | 57.84 | 65.86 ± 0.19 | 76.09 ± 0.38 |
| | ML-RAIL | 57.84 | 66.86 ± 0.17 | 78.21 ± 0.24 |
| | + rescue gate | 57.69 ± 0.01 | 68.15 ± 0.16 | 78.21 ± 0.24 |
| | + reverse gate, full kernel (earlier version) | 58.36 ± 0.04 | 68.12 ± 0.17 | 78.21 ± 0.24 |
| | **+ reverse gate, class token** | **59.29 ± 0.05** | **68.39 ± 0.17** | **78.21 ± 0.24** |

Against RAIL in the same harness, the full method gains:
- Order I: +1.20 Transfer, +2.05 Average, +1.59 Last.
- Order II: +0.77 / +1.86 / +1.70.
- 5-shot: +1.45 / +2.53 / +2.12.

Published rows, for reference only:

- Order I: RAIL Dual 62.4 / 71.9 / 82.4; MoRAM 63.3 / 72.7 / 80.9; LADA – / 72.7 / 83.1.
- Order II: RAIL Dual 57.7 / 68.1 / 82.5 (RAIL paper); LADA also reports order II.
- 5-shot, 11 domains: RAIL 60.0 / 65.7 / 73.2 and a rank-1 LoRA expert pool 60.6 / 66.9 / 74.3, both as
  reported by arXiv:2601.22828. Our harness's zero-shot Transfer there is 2.2 points below that paper's for
  every method; the cause was not found. Absolute numbers in this setting are therefore not comparable;
  only the same-harness differences are.

Change over ML-RAIL, per seed (16–20):

| Setting | Rescue gate ΔAverage | Rescue gate ΔTransfer | Both gates ΔAverage | Both gates ΔTransfer |
|---|---|---|---|---|
| Order I | +0.87 to +0.91 | −0.120 to −0.131 | +0.98 to +1.11 | +1.19 to +1.21 |
| Order II | +0.72 to +0.85 | −0.028 to −0.046 | +1.13 to +1.24 | +0.75 to +0.79 |
| 5-shot | +1.26 to +1.32 | −0.140 to −0.176 | +1.46 to +1.59 | +1.39 to +1.49 |

**The rescue gate's Transfer cost at 5 shots.** Its mean was −0.214 on development seeds, then −0.164,
−0.234 and −0.154 on seeds 6–10, 11–15 and 16–20. Against the 0.2-point tolerance set before its runs, it
fails on two of four seed sets. The cause may be that each τ_d is estimated from fewer images; this was not
tested.

**Class-token vs. full-kernel reverse gate** (seeds 16–20; rule fixed before the run: mean ΔTransfer ≥
+0.2, mean ΔAverage ≥ 0, ΔTransfer > 0 on every seed, Last unchanged):

| Setting | ΔTransfer | ΔAverage |
|---|---|---|
| Order I | +0.70 (+0.58 to +0.97) | +0.09 (one seed −0.006) |
| Order II | +0.30 | +0.24 |
| 5-shot | +0.93 | +0.26 |

It passed in all three settings, as it had on development seeds 1–5 (+0.66 / +0.10, +0.28 / +0.22, +1.02 /
+0.21).

**Gate settings.** For the rescue gate on order I, over q ∈ {0.25, 0.5} and m ∈ {0.05, 0.1}, ΔAverage ranges
from +0.69 to +0.92 and ΔTransfer from −0.09 to −0.21. The chosen point (0.25, 0.1) is not a sharp optimum.

## Where Transfer is lost (development, order I, seeds 1–5 unless noted)

**The ceiling.** An oracle takes back exactly the admitted images of domains not yet learned: +1.94
Transfer and +0.65 Average over ML-RAIL + rescue gate. It is limited because:
- only 4–5% of unseen-domain images are admitted at all;
- for those, zero-shot CLIP's best not-yet-learned class is right 44% of the time (68% on validation).

The remaining Transfer error is zero-shot CLIP's own, among the classes not yet learned.

**Signals** (`run_xtail_ceiling_signals.py`). All are training-free and computed from stored features. Each
is turned into a per-domain percentile, as the gates do.

| Signal | AUROC, unseen vs. learned among admitted |
|---|---|
| Layer-10 class token, cosine, nearest neighbour | **0.956** |
| Kernel, mean of 5 nearest neighbours | 0.948 |
| Kernel, nearest neighbour (first reverse gate) | 0.942 |
| Kernel, nearest training image of the adapter's top class | 0.938 |
| Pooled layer-10 patch tokens | 0.895 |
| Mahalanobis to the nearest class mean (in-sample reference) | 0.875 |
| Pooled layer-8 patch tokens | 0.851 |
| Mean patch token | 0.840 |
| Final embedding | 0.818 |

**Trade-off.** Change over ML-RAIL + rescue gate, with the threshold chosen on validation by two criteria:

| Signal | "Most Transfer, Average ≥ 0" (ΔT / ΔA) | "Most Transfer + Average" (ΔT / ΔA) |
|---|---|---|
| Kernel, nearest neighbour | 2.2nd percentile: +0.95 / −0.03 | 1.1st: +0.69 / +0.14 |
| Layer-10 class token | 1.6th: +1.36 / +0.10 | **0.8th: +1.32 / +0.24** |

The first criterion is the one that chose the full-kernel gate's percentile. It picked the 1st there, which
reaches +0.67 / +0.14 on these seeds; the grid above is finer. For the class token it picks a threshold that
loses Average against the full-kernel gate. That is why the second criterion was adopted for the class
token, after seeing these seeds. For the full kernel, the second criterion picks practically the same
percentile as before (1.1st against 1st), so it does not favour the new signal.

**Combinations did not help.** Requiring several signals to agree, or adding zero-shot mass, a probability
ratio or a logistic score over zero-shot and adapter confidence (`run_xtail_takeback_signals.py`), never
beat the best single signal at equal Average cost.

**Other settings** (seeds 1–5, class token at the 0.8th percentile):
- order II: +0.80 / +0.38 against an oracle of +1.73 / +1.19 (46% of the Transfer ceiling);
- 5-shot: +1.59 / +0.23 against +2.51 / +0.84 (64%).

The full-kernel gate reaches 30% and 23% there.

**What is left.** Pooled over seeds, these unseen domains hold most of the oracle's per-image gain:

| Unseen domain | Share of images | Mostly resembles |
|---|---|---|
| MNIST | 34% | DTD |
| SUN397 | 34% | Caltech101 |
| Food101 | 24% | Caltech101 |
| Flowers | 4% | Caltech101 |

These shares count images, while the metrics average per domain, so they show where the gain sits rather
than its exact size. The class-token gate catches nearly all of MNIST's and Caltech101's gain, but only 12%
of Food101's, 27% of SUN397's and none of Flowers'. Those images resemble Caltech101's own training images
(Caltech101 has pizza, sunflower and lotus classes), so similarity alone cannot tell them apart.

**Why the class token works, a hypothesis not tested:** intermediate-layer class tokens keep more domain
and style information, while the final embedding is shaped toward semantics, where Caltech101's broad
object classes overlap other domains.

## Tuning protocol (order I, seeds 6–10)

| Hyperparameters chosen on | Features, γ, λ | Transfer | Average | Last |
|---|---|---|---|---|
| RAIL's published setting | final embedding, 0.1, 1e−3 | 61.55 | 71.62 | 82.81 |
| All domains' validation | five blocks, 0.2, 1e−3 | 61.55 | 72.64 | 84.48 |
| + rescue gate | | 61.42 | 73.55 | 84.48 |
| + rescue and reverse gates | | 62.75 | 73.72 | 84.48 |
| First domain's validation | final embedding, 0.8, 0.1 | 61.55 | 71.43 | 82.38 |
| + rescue gate | | 60.57 | 71.80 | 82.38 |
| + rescue and reverse gates (full kernel)* | | 60.73 | 71.68 | 82.38 |
| First domain, five blocks kept | five blocks, 0.8, 0.1 | 61.55 | 71.97 | 83.13 |
| + rescue gate | | 61.43 | 72.86 | 83.13 |
| + rescue and reverse gates | | 62.75 | 73.04 | 83.13 |

\*The final embedding alone has no layer-10 class token, so the class-token gate is not available there.

- Selection is by validation regression error on Aircraft, as in RAIL. All three feature sets come within
  4% of each other there, so choosing the feature set from one domain is nearly arbitrary.
- The gates' settings stay at the all-domain values in every row. Measuring their effect on Transfer needs
  domains not yet learned, so they cannot be chosen from the first domain alone.
- With the wide kernel on the final embedding, the rescue gate costs a full Transfer point. A reverse gate
  on the same features wins back only 0.16 of it.
- With the five blocks kept, the class-token reverse gate adds +1.32 Transfer, as much as under
  all-domain tuning.
- An implementation note: the first run of this table applied the class-token gate to the
  final-embedding-only features as well. The class-token slice was empty there, so the gate silently did
  nothing. The gate now raises an error in that case, and those runs were discarded.

## What did not work

- **Spatial readouts** for fine-grained domains (attentive patch pooling, high-resolution tiles): at most
  +0.2 overall.
- **Better combinations of the same information:** calibrated fusion, flip averaging, sums of per-block
  kernels, per-domain kernel settings and fusion weights by leave-one-out, and residual fitting on
  zero-shot. All lose to ML-RAIL.
- **Replacing or softening RAIL's gate:** a gate-free Bayesian residual, an adapter-confidence gate,
  zero-shot mass and top-k gates, global-threshold gates, routing by zero-shot's domain (Last −8.5), and a
  smaller fusion weight for rescued images.
- **Per-domain experts behind cost-free routing:** re-tuned kernel experts, and rank-2 LoRA experts on the
  image encoder. The LoRA experts were weaker than ML-RAIL in every domain at a feasible training budget.
- **Fixed memory:** random Fourier features at width 16,000 come within 0.27 Average and 0.52 Last of the
  dual model, which missed its tolerance.
- **Take-back signals other than the class token:** Mahalanobis distance, pooled patch tokens, the final
  embedding, top-k and class-conditional kernels, combinations of signals, and zero-shot or adapter
  confidence. See "Where Transfer is lost".

## Position relative to prior work

- **RAIL** (NeurIPS 2024) uses the same adapter family. ML-RAIL changes only the representation, and both
  gates edit RAIL's zero-shot gate.
- **LayUP** concatenates multi-layer features of a frozen ViT for rehearsal-free class-incremental learning
  with a RanPAC-style classifier. ML-RAIL applies the same idea to RAIL's dual adapter.
- **DIKI** (Tang et al., ECCV 2024) is the closest mechanism to the rescue gate. It fits a Gaussian per
  task on frozen features and maps the max likelihood through a sigmoid to weight task knowledge against
  frozen CLIP. It is trained, evaluated on MTIL, and reports no X-TAIL numbers. The rescue gate is
  training-free and nonparametric, uses per-domain nearest-neighbour quantiles, and only rescues images
  that a zero-shot gate rejects.
- **LADA** (ICML 2025) already reports that learned knowledge can raise Transfer by separating learned from
  unseen classes; its own Transfer measure exceeds zero-shot CLIP's. It uses trained label-specific memory
  units and logit weighting. The reverse gate is a training-free, explicit take-back rule inside RAIL's
  fusion, with an oracle ceiling. **MoRAM** also exceeds zero-shot Transfer (63.3 vs. 62.4), by continual
  adaptation of rank-1 experts.
- **MoE-Adapters** (CVPR 2024, the DDAS router) and prototype routers also choose per task between experts
  and frozen CLIP, with one threshold or a learned router.
- **OOD detection** in class-incremental learning (Kim et al., NeurIPS 2022; kNN OOD, Sun et al., ICML
  2022; OpenCIL) is the background for using nearest-neighbour similarity as a membership test. There,
  rejection means abstaining; here it means predicting among the classes not yet learned. Per-group
  quantile thresholds resemble Mondrian conformal prediction, but no guarantee is claimed.
- **Novelty is incremental.** The contribution is the combination, measured carefully: oracle headroom for
  both gates, global versus per-domain thresholds on held-out domains, a comparison of take-back signals,
  and a training-free result that is competitive with trained methods.

## Limitations

- One backbone (CLIP ViT-B/16) and one benchmark family (X-TAIL and its variants).
- The gain over RAIL depends on tuning with all domains' validation data. Under RAIL's protocol with the
  five blocks kept it is smaller, and under RAIL's full protocol it is gone (table above).
- The reverse gate's selection criterion was changed after seeing development results.
- Transfer stays below MoRAM's 63.3, although the comparison crosses harnesses.
- The dual adapter stores all training features, as Dual-RAIL does. The fixed-memory variant missed its
  tolerance.
- The rescue gate recovers about 60% of its oracle's +1.5 Average headroom, and the reverse gate about 70%
  of its oracle's Transfer headroom on order I (46% on order II, 64% at 5 shots).

## Future work

- **The rest of the Transfer ceiling:** images that resemble another domain's classes. Telling them apart
  probably needs class-level semantics, not domain similarity. Beyond the ceiling, Transfer is CLIP's own
  error on unseen classes. A learned prior from past domains tied but did not beat CLIP's text
  embeddings.
- **The rescue gate on the class token,** the mirror image of the reverse gate.
- **Thresholds with few shots:** shrink per-domain thresholds toward a pooled estimate.
- **First-domain tuning:** a selection criterion that is less noisy than regression error on one domain.
- **Other backbones** (CLIP ViT-L/14, SigLIP) and **longer domain streams** with the fixed-memory adapter.

## Reproducing

- Everything: `python scripts/reproduce.py` (data, features, evaluation, checks, figures); it resumes if
  interrupted.
- Protocol: `evaluation/xtail_protocol.py` (`DualRAIL`, `DomainThresholdGate`, `run_stream`) and
  `evaluation/ml_rail.py`.
- Reported numbers: `scripts/evaluate.py`. Tuning protocol, gate checks, Transfer headroom and take-back
  signals: `scripts/experiments/`.
- Results: [results/xtail_results.json](../results/xtail_results.json).
- Hardware: a single GTX 1070. Set `XTAIL_WORKERS=1`, `XTAIL_GPU_FRACTION=0.5` and `XTAIL_SOLVE=cpu` to
  keep the machine usable during runs.
