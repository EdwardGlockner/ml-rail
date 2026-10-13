# Research log

How this project went from a self-organizing visual dictionary to a training-free continual learner
for CLIP. Before each experiment, a hypothesis, a settings grid and a pass/fail rule were fixed in a
short note (the notes are not included). Negative results are kept, because most of the project's direction
came from them. The final method is described in [REPORT_XTAIL.md](REPORT_XTAIL.md).

## Phase 1 -- A dictionary that grows its own vocabulary

**Question.** Can a visual representation keep improving in an open world by creating, merging,
pruning and splitting its own primitives, instead of relying on a vocabulary fixed at pretraining?

**Mechanism.** A frozen DINOv2 backbone turns each image into patch features. A dictionary
reconstructs each patch from its best-matching primitive. Where reconstruction keeps failing on the
same recurring pattern, a new primitive is **born**; near-duplicates are **merged**; unused ones are
**pruned**; a primitive that turns out to mix two concepts can be **split**. No gradient descent.

**Findings.**
- On a synthetic stream the dictionary births a primitive exactly when a genuinely new motif
  appears, reuses it 96% of the time afterwards, and does not grow on recombinations of known parts.
- A scaled-up benchmark first *reversed* the result: an oversized fixed dictionary won. The cause
  was a confound in the benchmark -- its categories were partly defined by *where* a motif appeared,
  which a fixed dictionary could exploit -- not the method. With categories defined only by which
  motifs are present, the growing dictionary won clearly again (harmonic ~0.81 vs. ~0.61). Fixing
  the benchmark, rather than explaining the reversal away, was the key correction of this phase.

## Phase 2 -- Real images and a complete lifecycle

**Findings.**
- On CIFAR-100 the first real-image run failed outright; adaptive, data-calibrated birth criteria
  fixed it. Over five seeds, growth beat a fixed dictionary, and the result held on the full
  few-shot class-incremental (FSCIL) protocol, on Caltech-101, and on a second backbone (CLIP).
- **Wake/sleep.** Splitting learning into a cheap buffering pass and a periodic consolidation pass
  matched the original accuracy at ~7x the speed. A faster clustering algorithm then closed most of
  the remaining full-scale gap while raising the speedup to ~16x.
- **Learned controllers.** A small meta-trained, then frozen controller replaced hand-tuned birth
  thresholds and recovered automatically from a failure on CLIP that otherwise needed manual tuning.
  Extending it to merge and prune worked, at a small, measured accuracy cost at full scale.
- **Quiet-cycle pruning.** A CLIP-specific regression survived three wrong hypotheses before the real
  cause was found: pruning on a single quiet session. Requiring several consecutive quiet cycles
  fixed it.
- **Kill-criterion check.** A fixed dictionary of the *same* size, fit with exact k-means, lost
  clearly (harmonic accuracy 0.466 vs. 0.538): it kept old classes but failed on new ones, exactly
  the failure the thesis predicts.
- Also built: an interactive dictionary explorer, cross-architecture tests (ResNet, a text encoder),
  spatial-structure measurements of the learned primitives, and a label-aware split.

## Phase 3 -- The reality check

The comparisons so far were against other dictionaries. Against the baselines that matter, the
picture changed.

- On CIFAR-100 FSCIL, **raw backbone features with no dictionary at all** beat every dictionary
  readout (harmonic 0.645 vs. 0.596 for the best fusion of the two), and so did a small replay-protected
  fine-tuned head (0.647).
- On a task-incremental CLIP setup the dictionary did beat the frozen backbone on every seed
  (0.604 vs. 0.558). But the published method it was compared with turned out to have been
  **undertrained**. Trained properly, the published method beat the dictionary by 12-20 points at
  equal or lower memory (0.751 vs. 0.536 on a six-task stream).
- Two last tests of what is specific to the dictionary -- discovering unlabeled classes and flagging
  unseen ones -- failed on every seed. **The dictionary line was retired by its own pre-set kill
  criteria.**

Lesson kept for the rest of the project: verify that every baseline is converged (accuracy *and*
output confidence) before trusting a comparison against it.

## Phase 4 -- After the dictionary

- Stacking known frozen add-ons (intermediate layers, test-time views, an attentive probe) cleared a
  pass/fail rule set in advance against the best published re-ranker on every seed (0.767 vs. 0.757). A
  published random-projection method (RanPAC) on multi-layer features did slightly better (0.771).
- **Growing** the random-projection layer during the stream lost to a plain fixed layer of the same
  size, even with exact statistics of old data: choosing units to fit a few labeled images overfits.
- **Forward transfer.** A "schema prior" learned from past domains improved as domains accumulated,
  but only tied CLIP's own class-name embeddings, so the direction was closed.

## Phase 5 -- X-TAIL

The pipeline had converged to the published frontier, so the work moved to the field's own
benchmark, X-TAIL: CLIP ViT-B/16, ten domains in sequence, 16 shots per class, every test image
classified among ~1,100 class names with no domain label.

- **Replication first.** RAIL (NeurIPS 2024) was reproduced with its official code (61.5 / 71.2 /
  82.3 Transfer / Average / Last vs. 62.4 / 71.9 / 82.4 published) and with an independent harness.
- **ML-RAIL.** RAIL's closed-form kernel ridge adapter on five feature blocks from the same CLIP
  forward pass raised Last by about 1.7 over Dual-RAIL in the same harness. Better class text (prompt
  ensembles) also helped, but was kept out of the final method because RAIL does not use it.
- **Where the remaining error lives.** A perfect gate between "use the adapter" and "trust zero-shot"
  would add +1.5 Average, all from learned-domain images that zero-shot confuses with a not-yet-learned
  class.
- **What did not recover it:**
  - a gate-free Bayesian model -- no distance in CLIP space separates learned from unseen domains as
    sharply as zero-shot's top guess;
  - looser zero-shot gates -- every gain was paid back one-for-one on unseen domains;
  - a single global similarity threshold -- it works only when tuned on the domains it is tested on
    (-1.4 to +0.65 on held-out domains);
  - per-domain kernel settings, residual fitting and calibrated fusion;
  - per-domain experts behind cost-free routing, including LoRA adapters trained on each domain,
    which were weaker than ML-RAIL in every domain at a feasible training budget.
- **What did: a per-domain threshold rescue gate.** Each domain sets its own threshold from how
  similar its own training images are to each other. In a split of X-TAIL's domains it carried over
  to held-out domains (+0.55 to +0.86).
- **Two rounds, and one fail.** The first version (mass floor 0.05) passed both 16-shot task orders
  but cost 0.27 Transfer in the 5-shot variant, a fail. A second round raised the floor to 0.1 and
  still missed the 5-shot tolerance by 0.014 on the development seeds. Test seed 1 had also been part
  of the gate selection. So the final numbers were re-run once on fresh seeds 6-10, with every setting
  fixed. There the gate passes on both 16-shot orders and, borderline, at 5 shots (Transfer -0.03 to
  -0.16 on the mean, against -0.21 at 5 shots on the development seeds; Average +0.8 to +1.3 on every
  seed; Last unchanged by construction).
- **Bounded memory.** A fixed-size random-feature version of ML-RAIL came within 0.27 Average /
  0.52 Last of the full model, just outside its tolerance.
- **An external review** pointed out that the hyperparameters had been chosen on validation data
  from all ten domains, while RAIL tunes on the first domain only. Under RAIL's protocol, with the
  five blocks kept, ML-RAIL + gate gives 72.86 / 83.13 instead of 73.55 / 84.48; choosing the feature
  set on the first domain too falls back to RAIL's final embedding, and the gain over RAIL nearly
  disappears (+0.18 Average, -0.43 Last). Both are reported.

Result at this point (order I, held-out seeds 6-10): 61.42 / 73.55 / 84.48, against 61.55 / 71.62 /
82.81 for Dual-RAIL in the same harness.

## Phase 6 -- Transfer, and the reverse gate

Transfer had been written off as "zero-shot CLIP by construction" for every RAIL-style method. It isn't
quite: zero-shot CLIP also loses accuracy on unseen domains by confusing their images with classes that
are already learned, and RAIL's gate then sends those images to an adapter that can only name learned
classes.

- **Measure first.** About 4-5% of unseen-domain images were admitted, and all of them were wrong. An
  oracle that takes exactly those images back, and predicts zero-shot's best not-yet-learned class,
  would add +1.9 Transfer on order I (+1.7 order II, +2.5 at 5 shots). The bar set before the
  measurement was +0.5, so the idea went ahead.
- **The reverse gate** reuses the rescue gate's per-domain statistics. An admitted image is taken back
  if its similarity to every learned domain is below a low percentile of that domain's own
  nearest-neighbour similarities. The percentile was chosen on validation as the loosest that did not
  lower Average (the 1st). Looser thresholds take more Transfer back, but start taking back
  learned-domain images: Average falls by 0.6 at the 5th percentile and 1.9 at the 10th.
- **Development seeds 1-5** passed the rule in all three settings (mean ΔTransfer +0.68 / +0.53 /
  +0.63, ΔAverage +0.13 / +0.15 / +0.00).
- **Literature check.** LADA (ICML 2025) already reports that learned knowledge can lift Transfer by
  separating learned from unseen classes, using trained modules. The reverse gate is a training-free,
  explicit version inside RAIL's fusion, so it is reported as an extension of the gate.
- **Final run on fresh seeds 11-15,** for every method at once. The reverse gate passed again:
  ΔTransfer +0.73 / +0.52 / +0.64 over the rescue gate alone, ΔAverage +0.09 / +0.11 / +0.00, Last
  unchanged. Transfer is now above zero-shot CLIP's in all three settings.
- **And one more fail.** On the same seeds, the rescue gate alone cost -0.234 Transfer at 5 shots, so
  it fails its tolerance on two of its three seed sets (-0.214, -0.164, -0.234). The combined method
  does not have the problem, but the rescue gate on its own is not reliable at 5 shots.

- **Under RAIL's tuning protocol** (seeds 6-10): with the five blocks kept and only γ and λ chosen on
  the first domain, the reverse gate adds +0.81 Transfer, as before. With the feature set also chosen
  there, the wide kernel it brings breaks both gates (the reverse gate wins back 0.16 of the rescue
  gate's 1-point Transfer cost).
- **A better take-back signal?** Rules adding zero-shot CLIP's mass or probability ratio on the
  not-yet-learned classes, or a logistic score over these and the adapter's confidence, were selected on
  validation and checked on seeds 1-5. None beat the plain similarity threshold. Each bought a little
  more Transfer with Average, and at equal Average cost the plain threshold was as good or better.

Result at this point (order I, held-out seeds 11-15): 62.16 / 73.61 / 84.52.

## Phase 7 -- How close to the ceiling?

Why is the ceiling +1.9 and not much more? Only 4-5% of unseen-domain images are admitted, and for those
zero-shot CLIP's best not-yet-learned class is right less than half the time; the rest of the Transfer
error is CLIP's own. So the question became how much of the +1.9 a similarity signal can reach.

- **Many signals, one table.** For every admitted image: kernel similarity (nearest, top-5, to the
  top class), cosine in each of the five feature blocks separately, and Mahalanobis distance, each as a
  per-domain percentile. AUROC for "this image is from an unseen domain": the layer-10 class token
  alone 0.956, the full kernel 0.942, the final embedding only 0.818. The intermediate class token
  separates domains better than any combination including it.
- **A wrong turn in selection.** The rule that had chosen the first reverse gate's percentile ("most
  Transfer with no Average loss on validation") pushed the class token's threshold too loose. On
  seeds 1-5 it gained Transfer but lost 0.04 Average against the full-kernel gate, a fail by the rule
  set for adopting a new signal. A balanced criterion (most Transfer + Average on validation) picks the
  0.8th percentile. Applied to the old signal it picks the same percentile as before, so it is not
  tailored to the new one. But it was adopted after seeing the development seeds, and is reported as
  such.
- **Development checks, then a final run.** The class-token gate beat the full-kernel gate on seeds
  1-5 of all three settings (+0.66 / +0.28 / +1.02 Transfer, Average up too). On fresh seeds 16-20 it
  did again (+0.70 / +0.30 / +0.93), positive on every seed. It reaches about 70% of the ceiling on
  order I, against 35% before.
- **What is left** is mostly Food101, Flowers and SUN397 images that look like Caltech101's own
  training images. Caltech101 has pizza, sunflower and lotus classes. A domain-similarity signal
  cannot separate those.
- **A bug caught.** Rerunning the tuning-protocol table with the new gate, the row for RAIL's full
  protocol came out identical, seed for seed, to the rescue gate alone. Those features are only the
  final embedding, so the class-token slice was empty and the gate silently never fired. An early
  look at one seed had been read as "the class token repairs the wide-kernel case", which was wrong.
  The gate now raises an error for a slice outside the features, and those runs were discarded.

Final result (order I, held-out seeds 16-20): **62.75 / 73.76 / 84.53**, against 61.55 / 71.71 / 82.94 for
Dual-RAIL in the same harness and best published 63.3 / 72.7 / 83.1 (other code).

## Phase 8 -- A second backbone

Everything so far used CLIP ViT-B/16. CLIP ViT-L/14 tests whether the method is tied to that one model.

- **Same images, same relative depths.** The class token is taken at layer 20 of 24, and pooled patch
  tokens at 16 and 20. Encoding took about two hours on the GTX 1070. A sanity check: ViT-L/14 zero-shot
  accuracy on Aircraft is 29.9%, against 24.2% for B/16, in line with the CLIP paper.
- **Nothing re-tuned.** Validation on ViT-L/14 also picks ML-RAIL's γ 0.2. For RAIL it picks 0.4, so
  RAIL was run with both widths; on test the frozen 0.1 is slightly better.
- **The rule passed in all three settings**: ML-RAIL above RAIL, both gates above ML-RAIL, and Transfer
  above zero-shot. The gains are smaller than on B/16 (order I: +0.94 / +1.32 / +1.28). The rescue gate
  on its own again overshoots its Transfer tolerance, and the reverse gate again makes up for it.
- **A bug avoided.** Both backbones used seeds 16-20, and the figure script filtered results by seed
  only. It was fixed to filter by backbone too before the B/16 tables were regenerated.

Final result on ViT-L/14 (order I, seeds 16-20): **71.51 / 79.98 / 88.08**, against 70.57 / 78.66 / 86.81
for Dual-RAIL.

## Lessons

- Change the benchmark when it is confounded; don't explain the result away.
- A comparison is only as good as its weakest baseline. Check convergence, confidence, and that the
  tuning protocol matches the baseline's.
- Measure the ceiling (an oracle) before building the mechanism meant to reach it.
- Test on held-out domains, not just held-out images, and keep the final test seeds out of every
  selection loop.
- Record narrow misses as misses.
- Ask where a metric is lost before accepting that it is fixed "by construction".
- A result identical to the baseline, seed for seed, is a bug until shown otherwise.
