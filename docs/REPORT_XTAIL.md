# ML-RAIL with a per-domain rescue gate: training-free continual learning for CLIP on X-TAIL

*Technical report, 2026-10-09. The path to this result, including the negative results, is in
[RESEARCH_LOG.md](RESEARCH_LOG.md). Per-seed numbers are in
[results/xtail_results.json](../results/xtail_results.json).*

## Summary

X-TAIL is cross-domain task-agnostic incremental learning, here with CLIP ViT-B/16, 10 domains and 16 shots
per class. On this benchmark, a training-free extension of RAIL uses the same inputs as RAIL: one frozen
backbone, the shots, the class names, and RAIL's single prompt "a photo of a {}.". The hyperparameters,
however, were tuned on all domains' validation data (see qualification 1). On held-out seeds
6–10 it reaches **Transfer 61.42 / Average 73.55 / Last 84.48**. In the same harness RAIL's Dual-RAIL gives
61.55 / 71.62 / 82.81. The best published numbers we found are 63.3 (MoRAM) / 72.7 (MoRAM, LADA) / 83.1
(LADA). Those come from other code, so the comparison is indicative only.

The method has two parts. ML-RAIL runs RAIL's dual adapter on multi-layer frozen features (+1.03 Average,
+1.67 Last). A rescue gate admits images that RAIL's zero-shot gate rejects but that lie close to a learned
domain (+0.91 Average, −0.13 Transfer, Last unchanged).

Two qualifications:

1. **Tuning protocol.** The hyperparameters were chosen on validation data from all ten domains; RAIL tunes
   on the first domain only. Under RAIL's protocol, with the five feature blocks kept, the result is
   61.43 / 72.86 / 83.13. That is +1.25 Average and +0.33 Last over RAIL instead of +1.93 / +1.67. Under
   RAIL's full protocol, with the feature set also chosen on the first domain, the selection falls back to
   RAIL's final embedding and the gain nearly disappears: +0.18 Average and −0.43 Last with the gate.
2. **Development history.** The gate rule was revised once after a failed 5-shot result, and one test seed
   (seed 1) was used during gate selection. All reported numbers therefore come from fresh seeds 6–10, run
   once after every setting was fixed.

## Setting and protocol

- **X-TAIL** (Xu et al., 2024) learns 10 domains in sequence. Order I is Aircraft, Caltech101, DTD,
  EuroSAT, Flowers, Food101, MNIST, OxfordPets, StanfordCars and SUN397. After each step, every domain's
  test set is classified over the union of all ~1,100 class names, without domain hints. The metrics
  follow RAIL's code exactly:
  - Transfer: mean accuracy on domains not yet learned.
  - Average: mean over all steps and domains.
  - Last: mean after the final step.
- **Replication.** RAIL's official code on our data gives 61.5 / 71.2 / 82.3. The paper reports 62.4 /
  71.9 / 82.4 (arXiv:2406.18868v3); LADA quotes 71.3 Average. Our harness's Dual-RAIL gives 61.55 / 71.62
  / 82.81 on seeds 6–10. The changes needed to run the official code are in
  [third_party/RAIL_CHANGES.md](../third_party/RAIL_CHANGES.md).
- **Splits and seeds.**
  - Hyperparameters were chosen on CoOp validation splits (sampling seed 100).
  - Development test runs used seeds 0–5.
  - The reported numbers use seeds 6–10, which were not looked at before the final run.
- **Rules.** Before each experiment, its settings and a pass/fail rule were fixed in a short note; the
  notes are not included. Baselines get every change that is not part of the method.
- **Precision.** Float32 solves throughout. Float64 on seed 6 matches within 0.05 points.

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

**Rescue gate** (`DomainThresholdGate`).

- When domain d arrives, compute for each of its training images the max kernel similarity to d's other
  training images, excluding exact duplicates. τ_d is the q = 0.25 quantile of these values.
- At test time, an image is admitted if RAIL's rule holds (its zero-shot top-1 class is learned), or if
  both of these hold:
  - zero-shot puts at least m = 0.1 probability on learned classes;
  - its max similarity to some learned domain d's stored images is at least τ_d.
- The two global settings were chosen on order-I validation. τ_d uses only data the method already stores.
- This is an empirical per-domain threshold. There is no coverage guarantee, although the realized
  acceptance rate is close to nominal: on held-out seeds, 76.0% of each domain's own test images clear τ_d
  (nominal 75%, per domain 71–79%). At q = 0.5 the rate is 51.0% (nominal 50%).
- Last is unchanged by construction. After the final step every class is learned, so RAIL's rule already
  admits every image.

**Why per-domain thresholds.**

- With an oracle gate, Average would rise by +1.5 on order I, all from learned-domain images that
  zero-shot CLIP confuses with a not-yet-learned class.
- A single global similarity threshold captures that headroom when tuned on the same ten domains. It does
  not carry over: tuned on half the domains and tested on the other half, it gives −1.4 to +0.65 across
  three splits, because domains differ in how tightly their images cluster.
- Per-domain thresholds gave +0.55 / +0.81 / +0.86 in the same splits.
- This held-out-domain check stays within X-TAIL's ten domains.

## Results (held-out seeds 6–10, mean ± std)

| Setting | Method | Transfer | Average | Last |
|---|---|---|---|---|
| Order I, 16-shot | Zero-shot CLIP | 61.55 | 57.81 | 57.81 |
| | Dual-RAIL (our harness) | 61.55 | 71.62 ± 0.12 | 82.81 ± 0.14 |
| | ML-RAIL | 61.55 | 72.64 ± 0.11 | 84.48 ± 0.17 |
| | **ML-RAIL + rescue gate** | 61.42 | **73.55 ± 0.10** | **84.48 ± 0.17** |
| Order II, 16-shot | Dual-RAIL (our harness) | 57.06 | 67.96 ± 0.12 | 83.01 ± 0.18 |
| | ML-RAIL | 57.06 | 68.60 ± 0.10 | 84.60 ± 0.14 |
| | **ML-RAIL + rescue gate** | 57.03 | **69.41 ± 0.09** | **84.60 ± 0.14** |
| 11 domains, 5-shot | Dual-RAIL (our harness) | 57.84 | 65.82 ± 0.17 | 76.03 ± 0.41 |
| | ML-RAIL | 57.84 | 66.79 ± 0.12 | 78.14 ± 0.30 |
| | **ML-RAIL + rescue gate** | 57.67 | **68.09 ± 0.13** | **78.14 ± 0.30** |

Published rows, for reference only:

- Order I: RAIL Dual 62.4 / 71.9 / 82.4; MoRAM 63.3 / 72.7 / 80.9; LADA – / 72.7 / 83.1.
- Order II: RAIL Dual 57.7 / 68.1 / 82.5 (RAIL paper); LADA also reports order II.
- 5-shot, 11 domains: RAIL 60.0 / 65.7 / 73.2 and a rank-1 LoRA expert pool 60.6 / 66.9 / 74.3, both as
  reported by arXiv:2601.22828. Our harness's zero-shot Transfer there is 2.2 points below that paper's for
  every method; the cause was not found. Absolute numbers in this
  setting are therefore not comparable; only the same-harness differences are.

Gate vs. ML-RAIL, per seed:

| Setting | ΔAverage | ΔTransfer |
|---|---|---|
| Order I | +0.88 to +0.95 | −0.115 to −0.142 |
| Order II | +0.80 to +0.85 | −0.025 to −0.038 |
| 5-shot | +1.27 to +1.31 | −0.138 to −0.200 |

The mean Transfer cost is within the 0.2-point tolerance set before the run in all three settings, but
at 5 shots only borderline: −0.164 here against −0.214 on the development seeds. The larger 5-shot cost may come from estimating each τ_d from fewer images; this was not tested.

**Gate settings.** On order I, over q ∈ {0.25, 0.5} and m ∈ {0.05, 0.1}, ΔAverage ranges from +0.69 to
+0.92 and ΔTransfer from −0.09 to −0.21. The chosen point (0.25, 0.1) is not a sharp optimum.

## Tuning protocol (order I, seeds 6–10)

| Hyperparameters chosen on | Features, γ, λ | Transfer | Average | Last |
|---|---|---|---|---|
| RAIL's published setting | final embedding, 0.1, 1e−3 | 61.55 | 71.62 | 82.81 |
| All domains' validation | five blocks, 0.2, 1e−3 | 61.55 | 72.64 | 84.48 |
| + rescue gate | | 61.42 | 73.55 | 84.48 |
| First domain's validation | final embedding, 0.8, 0.1 | 61.55 | 71.43 | 82.38 |
| + rescue gate | | 60.57 | 71.80 | 82.38 |
| First domain, five blocks kept | five blocks, 0.8, 0.1 | 61.55 | 71.97 | 83.13 |
| + rescue gate | | 61.43 | 72.86 | 83.13 |

- Selection is by validation regression error on Aircraft, as in RAIL. All three feature sets come within
  4% of each other there, so choosing the feature set from one domain is nearly arbitrary.
- The gate's settings stay at the all-domain values in every row. Measuring their Transfer cost needs
  domains not yet learned, so they cannot be chosen from the first domain alone.
- With the wide kernel on the final embedding, the gate costs a full Transfer point.

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

## Position relative to prior work

- **RAIL** (NeurIPS 2024) uses the same adapter family. ML-RAIL changes only the representation, and the
  gate extends RAIL's zero-shot gate.
- **LayUP** concatenates multi-layer features of a frozen ViT for rehearsal-free class-incremental learning
  with a RanPAC-style classifier. ML-RAIL applies the same idea to RAIL's dual adapter.
- **DIKI** (Tang et al., ECCV 2024) is the closest mechanism to the gate. It fits a Gaussian per task on
  frozen features and maps the max likelihood through a sigmoid to weight task knowledge against frozen
  CLIP. It is trained, evaluated on MTIL, and reports no X-TAIL numbers. The rescue gate is training-free
  and nonparametric, uses per-domain nearest-neighbour quantiles, and only rescues images that a zero-shot
  gate rejects.
- **MoE-Adapters** (CVPR 2024, the DDAS router) and prototype routers also choose per task between experts
  and frozen CLIP, with one threshold or a learned router.
- **kNN OOD detection** (Sun et al., ICML 2022) and Kim et al. (NeurIPS 2022, the link between incremental
  learning and OOD detection) are the background for using nearest-neighbour similarity as a membership
  test. Per-group quantile thresholds resemble Mondrian conformal prediction, but no guarantee is claimed.
- **Novelty is incremental.** The contribution is the combination, measured carefully: oracle headroom,
  global versus per-domain thresholds on held-out domains, and a training-free result that is competitive
  with trained methods.

## Limitations

- One backbone (CLIP ViT-B/16) and one benchmark family (X-TAIL and its variants).
- The gain over RAIL depends on tuning with all domains' validation data. Under RAIL's protocol it is
  smaller (Table above).
- Transfer is unchanged by design, apart from the gate's small cost, and stays below MoRAM's 63.3.
- The dual adapter stores all training features, as Dual-RAIL does. The fixed-memory variant missed its
  tolerance.
- The gate recovers about 60% of the oracle gate's +1.5 Average headroom.

## Future work

- **Transfer** is the main open problem. Unseen domains are classified by zero-shot CLIP, so every
  RAIL-style method inherits CLIP's Transfer. Raising it without extra information needs forward transfer
  from learned domains, for example refined class-name embeddings or calibrated zero-shot predictions. A
  learned prior from past domains tied but did not beat CLIP's text embeddings.
- **Thresholds with few shots:** shrink per-domain thresholds toward a pooled estimate.
- **First-domain tuning:** a selection criterion that is less noisy than regression error on one domain.
- **Other backbones** (CLIP ViT-L/14, SigLIP) and **longer domain streams** with the fixed-memory adapter.

## Reproducing

- Everything: `python scripts/reproduce.py` (data, features, evaluation, checks, figures); it resumes if
  interrupted.
- Protocol: `evaluation/xtail_protocol.py` (`DualRAIL`, `DomainThresholdGate`, `run_stream`) and
  `evaluation/ml_rail.py`.
- Reported numbers: `scripts/evaluate.py`; tuning protocol and gate checks: `scripts/experiments/`.
- Results: [results/xtail_results.json](../results/xtail_results.json).
- Hardware: a single GTX 1070. Set `XTAIL_WORKERS=1`, `XTAIL_GPU_FRACTION=0.5` and `XTAIL_SOLVE=cpu` to
  keep the machine usable during runs.
