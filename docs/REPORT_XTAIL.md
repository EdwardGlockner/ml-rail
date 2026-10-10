# ML-RAIL with per-domain gating: training-free continual learning for CLIP on X-TAIL

*Technical report, 2026-10-09. The path to this result, including the negative results, is in
[RESEARCH_LOG.md](RESEARCH_LOG.md). Per-seed numbers are in
[results/xtail_results.json](../results/xtail_results.json).*

## Summary

X-TAIL is cross-domain task-agnostic incremental learning, here with CLIP ViT-B/16, 10 domains and 16 shots
per class. On this benchmark, a training-free extension of RAIL uses the same inputs as RAIL: one frozen
backbone, the shots, the class names, and RAIL's single prompt "a photo of a {}.". The hyperparameters,
however, were tuned on all domains' validation data (see qualification 1). On held-out seeds 11–15 it
reaches **Transfer 62.16 / Average 73.61 / Last 84.52**. In the same harness, RAIL's Dual-RAIL gives
61.55 / 71.63 / 82.84, and zero-shot CLIP's Transfer is 61.55. The best published numbers we found are
63.3 (MoRAM) / 72.7 (MoRAM, LADA) / 83.1 (LADA). Those come from other code, so the comparison is
indicative only.

The method has three parts:

1. **ML-RAIL** runs RAIL's dual adapter on multi-layer frozen features: +1.02 Average, +1.69 Last.
2. **A rescue gate** admits images that RAIL's zero-shot gate rejects but that lie close to a learned
   domain: +0.86 Average, −0.13 Transfer.
3. **A reverse gate** takes back admitted images that are far from every learned domain, and predicts
   them among the classes not yet learned: +0.73 Transfer and +0.09 Average on top of the rescue gate.

Last is unchanged by both gates.

Two qualifications:

1. **Tuning protocol.** The hyperparameters were chosen on validation data from all ten domains; RAIL tunes
   on the first domain only (check on seeds 6–10).
   - Under RAIL's protocol with the five feature blocks kept, the full method gives 62.24 / 72.96 / 83.13.
     That is +0.69 Transfer, +1.34 Average and +0.32 Last over RAIL, instead of +0.60 / +2.03 / +1.67.
   - Under RAIL's full protocol, with the feature set also chosen on the first domain, the selection falls
     back to RAIL's final embedding with a wide kernel, and the gain is gone: −0.82 / +0.06 / −0.43.
2. **Development history.**
   - The rescue gate's rule was revised once after a failed 5-shot result, and one test seed (seed 1) was
     used during its selection. It was therefore rerun once on fresh seeds 6–10.
   - The reverse gate was developed afterwards on validation and seeds 0–5.
   - All methods were then run once on fresh seeds 11–15. Those are the reported numbers, and seeds 6–10
     serve as a replication.
   - On its own, the rescue gate's 5-shot Transfer cost exceeds its 0.2 tolerance on two of the three seed
     sets.

## Setting and protocol

- **X-TAIL** (Xu et al., 2024) learns 10 domains in sequence. Order I is Aircraft, Caltech101, DTD,
  EuroSAT, Flowers, Food101, MNIST, OxfordPets, StanfordCars and SUN397. After each step, every domain's
  test set is classified over the union of all ~1,100 class names, without domain hints. The metrics
  follow RAIL's code exactly:
  - Transfer: mean accuracy on domains not yet learned.
  - Average: mean over all steps and domains.
  - Last: mean after the final step.
- **Replication.** RAIL's official code on our data gives 61.5 / 71.2 / 82.3. The paper reports 62.4 /
  71.9 / 82.4 (arXiv:2406.18868v3); LADA quotes 71.3 Average. Our harness's Dual-RAIL gives 61.55 / 71.63
  / 82.84 on seeds 11–15. The changes needed to run the official code are in
  [third_party/RAIL_CHANGES.md](../third_party/RAIL_CHANGES.md).
- **Splits and seeds.**
  - Hyperparameters were chosen on CoOp validation splits (sampling seed 100).
  - Development test runs used seeds 0–5.
  - Seeds 6–10 were a held-out run before the reverse gate.
  - Seeds 11–15 are the final run of every method, made once after every setting was fixed.
- **Rules.** Before each experiment, its settings and a pass/fail rule were fixed in a short note; the
  notes are not included. Baselines get every change that is not part of the method.
- **Precision.** Float32 solves throughout. Float64 on seed 6 (without the reverse gate) matches within 0.05 points.

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
images the max kernel similarity to d's other training images, excluding exact duplicates. Let s_d(x) be an
image's max similarity to d's stored images. τ_d is the 25th percentile of the nearest-neighbour values,
and ρ_d the 1st.

**Rescue gate.**

- An image is admitted if RAIL's rule holds (its zero-shot top-1 class is learned), or if both of these
  hold:
  - zero-shot puts at least m = 0.1 probability on learned classes;
  - s_d(x) ≥ τ_d for some learned domain d.
- The two settings were chosen on order-I validation. The thresholds use only data the method already
  stores.
- There is no coverage guarantee. The realized acceptance rate is close to nominal: 76.0% of each domain's
  own test images clear τ_d (nominal 75%, per domain 71–79%; seeds 6–10). At q = 0.5 it is 51.0%.

**Reverse gate** (`reverse_q=0.01`).

- An admitted image of a domain not yet learned is always wrong, because the adapter scores only learned
  classes. This happens to 4–5% of such images.
- An admitted image is taken back if s_d(x) < ρ_d for every learned domain d. Its prediction is zero-shot
  CLIP's best class among those not yet learned.
- It is inactive at the last step, when every class is learned.

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

## Results (held-out seeds 11–15, mean ± std)

| Setting | Method | Transfer | Average | Last |
|---|---|---|---|---|
| Order I, 16-shot | Zero-shot CLIP | 61.55 | 57.81 | 57.81 |
| | Dual-RAIL (our harness) | 61.55 | 71.63 ± 0.10 | 82.84 ± 0.13 |
| | ML-RAIL | 61.55 | 72.66 ± 0.08 | 84.52 ± 0.10 |
| | + rescue gate | 61.42 ± 0.01 | 73.52 ± 0.09 | 84.52 ± 0.10 |
| | **+ rescue and reverse gates** | **62.16 ± 0.04** | **73.61 ± 0.07** | **84.52 ± 0.10** |
| Order II, 16-shot | Zero-shot CLIP | 57.06 | 57.90 | 57.90 |
| | Dual-RAIL (our harness) | 57.06 | 67.91 ± 0.10 | 82.86 ± 0.07 |
| | ML-RAIL | 57.06 | 68.52 ± 0.10 | 84.47 ± 0.14 |
| | + rescue gate | 57.02 ± 0.01 | 69.32 ± 0.13 | 84.47 ± 0.14 |
| | **+ rescue and reverse gates** | **57.54 ± 0.03** | **69.44 ± 0.14** | **84.47 ± 0.14** |
| 11 domains, 5-shot | Zero-shot CLIP | 57.84 | 54.78 | 54.78 |
| | Dual-RAIL (our harness) | 57.84 | 65.79 ± 0.26 | 75.88 ± 0.54 |
| | ML-RAIL | 57.84 | 66.81 ± 0.19 | 78.16 ± 0.31 |
| | + rescue gate | 57.61 ± 0.10 | 68.09 ± 0.22 | 78.16 ± 0.31 |
| | **+ rescue and reverse gates** | **58.25 ± 0.15** | **68.09 ± 0.23** | **78.16 ± 0.31** |

Seeds 6–10 give the same numbers for the first four methods of each setting to within 0.2 (largest gap: Dual-RAIL's order-II Last, 0.15).

Published rows, for reference only:

- Order I: RAIL Dual 62.4 / 71.9 / 82.4; MoRAM 63.3 / 72.7 / 80.9; LADA – / 72.7 / 83.1.
- Order II: RAIL Dual 57.7 / 68.1 / 82.5 (RAIL paper); LADA also reports order II.
- 5-shot, 11 domains: RAIL 60.0 / 65.7 / 73.2 and a rank-1 LoRA expert pool 60.6 / 66.9 / 74.3, both as
  reported by arXiv:2601.22828. Our harness's zero-shot Transfer there is 2.2 points below that paper's for
  every method; the cause was not found. Absolute numbers in this setting are therefore not comparable;
  only the same-harness differences are.

Change over ML-RAIL, per seed (11–15):

| Setting | Rescue gate ΔAverage | Rescue gate ΔTransfer | Both gates ΔAverage | Both gates ΔTransfer |
|---|---|---|---|---|
| Order I | +0.83 to +0.89 | −0.118 to −0.137 | +0.91 to +0.99 | +0.574 to +0.672 |
| Order II | +0.72 to +0.87 | −0.026 to −0.041 | +0.80 to +1.00 | +0.431 to +0.524 |
| 5-shot | +1.22 to +1.34 | −0.152 to −0.350 | +1.22 to +1.38 | +0.275 to +0.575 |

**The rescue gate's Transfer cost at 5 shots.** Its mean was −0.214 on development seeds, −0.164 on seeds
6–10 and −0.234 on seeds 11–15. Against the 0.2-point tolerance set before its runs, it is a fail on two of
three seed sets. The cause may be that each τ_d is estimated from fewer images; this was not tested. With
the reverse gate added, 5-shot Transfer is above ML-RAIL's on every seed.

**Reverse gate vs. rescue gate alone** (the rule fixed before the run: mean ΔTransfer ≥ +0.3, mean
ΔAverage ≥ 0, ΔTransfer > 0 on every seed, Last unchanged): order I +0.73 / +0.09, order II +0.52 /
+0.11, 5-shot +0.64 / +0.004 (ΔTransfer / ΔAverage). It passed in all three settings, as it had on
development seeds 1–5 (+0.68 / +0.13, +0.53 / +0.15, +0.63 / +0.00). At 5 shots its Average effect is nil
(two seeds at −0.02).

**Gate settings.** For the rescue gate on order I, over q ∈ {0.25, 0.5} and m ∈ {0.05, 0.1}, ΔAverage ranges
from +0.69 to +0.92 and ΔTransfer from −0.09 to −0.21. The chosen point (0.25, 0.1) is not a sharp optimum.

## Where Transfer is lost (development, order I)

| Change vs. ML-RAIL + rescue gate | Validation ΔT / ΔA | Seeds 1–5 ΔT / ΔA (mean) |
|---|---|---|
| Oracle: take back every admitted unseen-domain image | +1.47 / +0.70 | +1.94 / +0.65 |
| Reverse gate at the 0.1st percentile | +0.09 / +0.01 | +0.14 / +0.02 |
| **Reverse gate at the 1st percentile (chosen)** | **+0.80 / +0.20** | **+0.68 / +0.13** |
| Reverse gate at the 2.5th percentile | +0.91 / −0.00 | +0.99 / −0.10 |
| Reverse gate at the 5th percentile | +0.97 / −0.81 | +1.18 / −0.62 |
| Reverse gate at the 10th percentile | +1.04 / −1.86 | +1.34 / −1.92 |

- About 4% of unseen-domain images are admitted on order I (5% on order II), and all of them are wrong.
  For them, zero-shot CLIP's best not-yet-learned class is right 44% of the time on test seeds (68% on
  validation).
- The percentile was chosen on validation as the loosest in the grid that did not lower Average.
- The reverse gate recovers about 40% of the oracle on order I, 30% on order II and 26% at 5 shots
  (oracles +1.73 and +2.50).

## Tuning protocol (order I, seeds 6–10)

| Hyperparameters chosen on | Features, γ, λ | Transfer | Average | Last |
|---|---|---|---|---|
| RAIL's published setting | final embedding, 0.1, 1e−3 | 61.55 | 71.62 | 82.81 |
| All domains' validation | five blocks, 0.2, 1e−3 | 61.55 | 72.64 | 84.48 |
| + rescue gate | | 61.42 | 73.55 | 84.48 |
| + rescue and reverse gates | | 62.15 | 73.65 | 84.48 |
| First domain's validation | final embedding, 0.8, 0.1 | 61.55 | 71.43 | 82.38 |
| + rescue gate | | 60.57 | 71.80 | 82.38 |
| + rescue and reverse gates | | 60.73 | 71.68 | 82.38 |
| First domain, five blocks kept | five blocks, 0.8, 0.1 | 61.55 | 71.97 | 83.13 |
| + rescue gate | | 61.43 | 72.86 | 83.13 |
| + rescue and reverse gates | | 62.24 | 72.96 | 83.13 |

- Selection is by validation regression error on Aircraft, as in RAIL. All three feature sets come within
  4% of each other there, so choosing the feature set from one domain is nearly arbitrary.
- The gates' settings stay at the all-domain values in every row. Measuring their effect on Transfer
  needs domains not yet learned, so they cannot be chosen from the first domain alone.
- With the wide kernel on the final embedding, the rescue gate costs a full Transfer point and the
  reverse gate wins back only 0.16 of it. Similarities under a wide kernel separate domains poorly.
- With the five blocks kept, the reverse gate adds +0.81 Transfer, as much as under all-domain tuning.
- The all-domain row with both gates (62.15 / 73.65 / 84.48 on seeds 6–10) replicates the final run on
  seeds 11–15 (62.16 / 73.61 / 84.52).

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
- **Looser reverse thresholds:** above the 1st percentile, the reverse gate starts taking back
  learned-domain images, and Average falls (see the table above).
- **Other take-back signals** (`run_xtail_takeback_signals.py`, development seeds 1–5, order I). Rules
  were tried that add zero-shot CLIP's mass on not-yet-learned classes, the ratio of its best
  not-yet-learned to best learned probability, or a logistic score over these and the adapter's top
  score. Each was selected on validation as the reverse gate was. None met the bar set beforehand (+0.2
  Transfer over the reverse gate with no loss in Average): each gained +0.06 to +0.28 Transfer but lost
  0.07 to 0.19 Average. Even selected on the development seeds themselves, the best rule with no loss in
  Average reaches +0.87 Transfer with the plain similarity threshold and at most +0.86 with the other
  signals. The zero-shot and adapter signals do not separate the images worth taking back.

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
  both gates, global versus per-domain thresholds on held-out domains, and a training-free result that is
  competitive with trained methods.

## Limitations

- One backbone (CLIP ViT-B/16) and one benchmark family (X-TAIL and its variants).
- The gain over RAIL depends on tuning with all domains' validation data. Under RAIL's protocol with the
  five blocks kept it is smaller, and under RAIL's full protocol it is gone (table above).
- Transfer stays below MoRAM's 63.3, although the comparison crosses harnesses.
- The dual adapter stores all training features, as Dual-RAIL does. The fixed-memory variant missed its
  tolerance.
- The rescue gate recovers about 60% of its oracle's +1.5 Average headroom. The reverse gate recovers a
  quarter to two fifths of its oracle's Transfer headroom.

## Future work

- **More Transfer:** the zero-shot distribution and the adapter's confidence did not improve the take-back
  rule, so reaching more of the oracle needs a different signal. Beyond the oracle, Transfer is CLIP's own
  error on unseen classes. A learned prior from past domains tied but did not beat CLIP's text
  embeddings.
- **Thresholds with few shots:** shrink per-domain thresholds toward a pooled estimate.
- **First-domain tuning:** a selection criterion that is less noisy than regression error on one domain.
- **Other backbones** (CLIP ViT-L/14, SigLIP) and **longer domain streams** with the fixed-memory adapter.

## Reproducing

- Everything: `python scripts/reproduce.py` (data, features, evaluation, checks, figures); it resumes if
  interrupted.
- Protocol: `evaluation/xtail_protocol.py` (`DualRAIL`, `DomainThresholdGate`, `run_stream`) and
  `evaluation/ml_rail.py`.
- Reported numbers: `scripts/evaluate.py`. Tuning protocol, gate checks and the Transfer headroom:
  `scripts/experiments/`.
- Results: [results/xtail_results.json](../results/xtail_results.json).
- Hardware: a single GTX 1070. Set `XTAIL_WORKERS=1`, `XTAIL_GPU_FRACTION=0.5` and `XTAIL_SOLVE=cpu` to
  keep the machine usable during runs.
