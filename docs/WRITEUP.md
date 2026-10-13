# Per-Domain Gating for Training-Free Continual Learning with CLIP

*Edward Glockner, October 2026. Code, per-seed results and the full research log:
[github.com/EdwardGlockner/ml-rail](https://github.com/EdwardGlockner/ml-rail).*

## Abstract

RAIL (NeurIPS 2024) learns new image domains on top of a frozen CLIP model with a closed-form kernel ridge
adapter, and uses zero-shot CLIP to decide which test images the adapter should handle. We keep RAIL's
inputs and its training-free design, and change three things.

- **ML-RAIL:** the adapter sees five feature blocks from different depths of the same forward pass, not
  only the final embedding.
- **Rescue gate:** images that zero-shot CLIP routes away from the adapter, but that resemble a learned
  domain, are admitted anyway.
- **Reverse gate:** images routed to the adapter that do not resemble any learned domain are taken back
  and classified among the classes not yet learned.

Both gates compare an image's similarity to each learned domain with a percentile of that domain's own
nearest-neighbour similarities. On the X-TAIL benchmark (CLIP ViT-B/16, ten domains, 16 shots per class),
the result is 62.75 Transfer, 73.76 Average and 84.53 Last accuracy on held-out seeds, against 61.55 /
71.71 / 82.94 for RAIL in the same code. The gain depends on tuning with validation data from all domains;
under RAIL's first-domain tuning protocol it shrinks or disappears. That and the other limitations are
stated alongside the results.

## 1. Problem

In X-TAIL [1], a model learns ten image domains in sequence (aircraft, Caltech101, DTD, EuroSAT, Flowers,
Food101, MNIST, Pets, Cars, SUN397) from a few labelled images per class. After every step it classifies
test images from *all* domains, among the ~1,100 class names of all of them, without being told the
domain. Three numbers summarize the 10 × 10 table of accuracies:

- **Transfer:** accuracy on domains not yet learned.
- **Average:** accuracy over all steps and domains.
- **Last:** accuracy after the final step.

RAIL [1] solves this without gradient training. A frozen CLIP encoder feeds a kernel ridge regression
whose solution is exact and independent of the order of domains, so nothing is forgotten. At test time,
an image goes to the adapter only if zero-shot CLIP's best class is one already learned; otherwise
zero-shot CLIP's prediction stands. That keeps zero-shot accuracy on unseen domains, but the router makes
mistakes in both directions.

## 2. Method

**Multi-layer features (ML-RAIL).** Five blocks come from one forward pass:
- the final embedding;
- the layer-10 class token;
- the mean patch token;
- GeM-pooled patch tokens of layers 8 and 10.

Each block is normalized and they are concatenated. The adapter is RAIL's kernel ridge regression with an
RBF kernel over all stored training features. The multi-layer idea is LayUP's [4], applied here to RAIL's
dual adapter.

**Per-domain thresholds.** When domain d is learned, compute, for each of its training images, the highest
similarity to d's other training images. A low percentile of these values is d's threshold. It adapts to
how tightly each domain's images cluster: MNIST digits are far more alike than SUN397 scenes, so no single
global threshold fits all domains.

**Rescue gate.** If zero-shot CLIP puts at least 10% of its probability on learned classes, and the image's
kernel similarity to some learned domain clears that domain's 25th-percentile threshold, the image goes to
the adapter even though zero-shot CLIP's top class is unlearned.

**Reverse gate.** An image of an unseen domain that the router sends to the adapter is always wrong,
because the adapter only knows learned classes. If an admitted image's similarity to *every* learned
domain falls below that domain's 0.8th-percentile threshold, it is taken back and predicted as zero-shot
CLIP's best class among those not yet learned. This similarity is measured on the layer-10 class token,
which separates unseen from learned domains better than any other signal tested.

Both gates are inactive at the last step, when every class is learned, so Last is exactly ML-RAIL's.

## 3. Protocol

- **Rules before results:** each experiment's settings and pass/fail rule were fixed in a note before it
  ran.
- **Selection only on validation:** all hyperparameters were chosen on CoOp validation splits. Development
  runs used test seeds 0–5.
- **Fresh seeds for every final run:** the reported numbers come from seeds 16–20, run once after all
  settings were fixed. Two earlier held-out runs (seeds 6–10 and 11–15) agree to within 0.2.
- **Replicated baseline:** RAIL was reproduced with its official code (61.5 / 71.2 / 82.3) and with an
  independent implementation before any comparison.
- **Disclosed deviations:**
  - The rescue gate's rule was revised once after a failed 5-shot result.
  - One test seed was used during its selection.
  - The reverse gate's threshold criterion was changed after seeing development results.

## 4. Results

Order I, 16 shots, CLIP ViT-B/16, seeds 16–20 (mean; standard deviations ≤ 0.17):

| Method | Transfer | Average | Last |
|---|---|---|---|
| Zero-shot CLIP | 61.55 | 57.81 | 57.81 |
| RAIL (Dual), same code | 61.55 | 71.71 | 82.94 |
| ML-RAIL | 61.55 | 72.69 | 84.53 |
| + rescue gate | 61.42 | 73.58 | 84.53 |
| **+ reverse gate (class token)** | **62.75** | **73.76** | **84.53** |
| *Published: MoRAM [2] / LADA [3]* | *63.3 / –* | *72.7 / 72.7* | *80.9 / 83.1* |

The published rows come from other code. Our run of RAIL is 0.2–0.9 points from its published numbers, so
that comparison is indicative only. The same ordering holds on the second task order and on a 5-shot
variant with CIFAR-100 added:

| Setting | RAIL | Full method | Gain (T / A / L) |
|---|---|---|---|
| Order II, 16-shot | 57.06 / 67.87 / 82.88 | 57.83 / 69.73 / 84.58 | +0.77 / +1.86 / +1.70 |
| 11 domains, 5-shot | 57.84 / 65.86 / 76.09 | 59.29 / 68.39 / 78.21 | +1.45 / +2.53 / +2.12 |

**Tuning protocol.** These hyperparameters were chosen on validation data from all ten domains. RAIL
chooses its own on the first domain only (seeds 6–10, order I):

- With the five blocks kept and only the kernel width and ridge chosen on the first domain, the gains
  shrink to +1.20 / +1.42 / +0.33.
- If the feature set is also chosen on the first domain, the selection falls back to RAIL's final
  embedding with a wide kernel. The gain is then gone: −0.82 / +0.06 / −0.43.

The validation errors of the candidate feature sets on that one domain differ by less than 4%, so that
selection is close to noise.

**Second backbone.** On CLIP ViT-L/14, every setting is kept from ViT-B/16, and the feature blocks are
taken at the same relative depths. On seeds 16–20:

| Setting | RAIL | Full method | Gain (T / A / L) |
|---|---|---|---|
| Order I, 16-shot | 70.57 / 78.66 / 86.81 | 71.51 / 79.98 / 88.08 | +0.94 / +1.32 / +1.28 |
| Order II, 16-shot | 65.48 / 75.11 / 86.78 | 66.11 / 76.26 / 88.14 | +0.62 / +1.14 / +1.36 |
| 11 domains, 5-shot | 66.66 / 73.76 / 81.73 | 67.84 / 75.56 / 83.39 | +1.18 / +1.80 / +1.66 |

The gains are smaller than on ViT-B/16, as expected with a stronger frozen backbone, but every part of
the pass/fail rule set before the run held in every setting:
- ML-RAIL above RAIL;
- both gates above ML-RAIL;
- Transfer above zero-shot CLIP's.

On ViT-L/14, validation picks the same kernel width for ML-RAIL as on ViT-B/16. RAIL was also run at the
width its own validation prefers, which turned out slightly worse on test.

## 5. Analysis

**Why per-domain thresholds.** A single global similarity threshold captured the rescue gate's headroom
only when tuned on the same domains it was tested on. Tuned on half the domains and tested on the other
half, it changed Average by −1.4 to +0.65. Per-domain thresholds gave +0.55 to +0.86 in the same split.

**How far can taking images back go?** An oracle that takes back exactly the admitted unseen-domain
images adds +1.9 Transfer. It isn't larger because only 4–5% of unseen-domain images are admitted at all,
and zero-shot CLIP is right about 44% of the time on those. Among nine training-free signals, the layer-10
class token separates these images best:

- its AUROC is 0.956, against 0.942 for the full kernel and 0.818 for the final embedding;
- it reaches about 70% of the oracle, against 35% for a reverse gate on the full kernel.

What remains is mostly Food101, Flowers and SUN397 images that resemble Caltech101's own training images.
Caltech101 has pizza, sunflower and lotus classes. A plausible reason the intermediate class token works,
not tested, is that it keeps more domain and style information than the final embedding.

**What did not work.** Many alternatives were tried and lost, each with a rule set beforehand:
- spatial readouts for fine-grained domains;
- per-domain kernel settings;
- calibrated or softened fusion;
- a gate-free Bayesian residual;
- global-threshold gates;
- per-domain LoRA experts;
- zero-shot or adapter confidence as take-back signals;
- the class token for the rescue gate.

The research log records each one.

## 6. Related work

- **RAIL** [1] introduced X-TAIL and the zero-shot-gated closed-form adapter.
- **LayUP** [4] showed the value of multi-layer features for rehearsal-free continual learning.
- **DIKI** [5] is the closest prior work to the rescue gate. It weights task knowledge against frozen CLIP
  by a per-task Gaussian likelihood, but it is trained and evaluated on a different benchmark.
- **LADA** [3] observed that learned knowledge can raise Transfer by separating learned from unseen
  classes, using trained modules. The reverse gate is a training-free, explicit version of that idea.
- **Nearest-neighbour OOD detection** [6] uses the same kind of similarity score. There, rejecting an
  image means abstaining; here it means predicting among the remaining classes.

The contribution is therefore incremental: a careful combination of known pieces, measured against
oracles, with a training-free result competitive with gradient-trained methods.

## 7. Limitations

- **One benchmark family and one model family** (CLIP ViT-B/16 and ViT-L/14). The tuning-protocol check
  and the signal study used ViT-B/16 only.
- **The tuning protocol:** the gain depends on using validation data from all domains.
- **Selection history:**
  - the rescue gate on its own misses its Transfer tolerance at 5 shots on two of four seed sets, and
    on ViT-L/14 also on order I;
  - the reverse gate's selection criterion was chosen after seeing development results.
- **Memory:** the adapter stores all training features, as RAIL's does.

## References

1. Xu et al. *Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models* (RAIL). NeurIPS 2024.
2. *Little by Little: Continual Learning via Incremental Mixture of Rank-1 Associative Memory Experts* (MoRAM). arXiv:2506.21035.
3. *LADA: Scalable Label-Specific CLIP Adapter for Continual Learning*. ICML 2025.
4. Ahrens et al. *Read Between the Layers* (LayUP). arXiv:2312.08888.
5. Tang et al. *Mind the Interference* (DIKI). ECCV 2024.
6. Sun et al. *Out-of-Distribution Detection with Deep Nearest Neighbors*. ICML 2022.
