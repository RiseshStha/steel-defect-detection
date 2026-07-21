# Classification Error Analysis Summary

**Source:** `results/error_analysis/classification/` (32 figures) and `classification_error_manifest.csv`  
**Setup:** Top-2 validation errors per model × 4 defect classes × false negative (FN) / false positive (FP). Threshold 0.5; checkpoints are seed-42 `best.pth` models used by the error-analysis script.

**Training prevalence (train split, positive-only counts):** Class 1: 536, **Class 2: 135 (rarest)**, Class 3: 3341, Class 4: 905.

---

## Baseline ResNet18

### False negatives (missed defects)

| Class | Support images | Visible pattern |
|---|---|---|
| **Class 1** | `baseline_class1_false_negative_1.png` (`fc0541a53.jpg`), `baseline_class1_false_negative_2.png` (`f9b98ab64.jpg`) | Complex strip-wide **vertical banding**, glare, and mottled texture; defects are diffuse rather than a single crisp cue. `f9b98ab64` has a **large black edge region** (~left third), so most of the frame is non-steel. Very low scores (p ≈ 0.06–0.07). |
| **Class 2** | `baseline_class2_false_negative_1.png` (`e97006670.jpg`), `baseline_class2_false_negative_2.png` (`0c6720401.jpg`) | **Subtle pitting / small dark spots** with low contrast against grainy vertical streaks. Both show **edge/coverage artifacts** (half-frame black or strong right-edge shadow). p ≈ 0.05–0.10 — among the hardest misses. |
| **Class 3** | `baseline_class3_false_negative_1.png` (`82e82b713.jpg`), `baseline_class3_false_negative_2.png` (`85b72c878.jpg`) | Defects are **bright vertical streaks or white patches** on dark, striated steel. Model still assigns p ≈ 0.05–0.09 despite visible marks — suggests confusion with normal rolling texture or inconsistent defect scale in the crop. |
| **Class 4** | `baseline_class4_false_negative_1.png` (`2749c24a9.jpg`), `baseline_class4_false_negative_2.png` (`7982045ad.jpg`) | **Higher FN probabilities** (p ≈ 0.20–0.22) than other classes. Defects are **small, at strip bottom or amid wash/streak noise**; heavy vertical texture competes with the true signal. |

**Class concentration:** Examples are balanced (2 per class), but **Class 2 FNs look qualitatively hardest** (faintest defects + worst imaging), consistent with Class 2 being the smallest training class (135 positives). Class 4 FNs are “near-misses” (higher p) rather than complete blindness.

### False positives (background called defect)

| Class | Support images | Visible pattern |
|---|---|---|
| **Class 1** | `baseline_class1_false_positive_1.png` (`681af7b91.jpg`), `baseline_class1_false_positive_2.png` (`b568146ec.jpg`) | **Grainy/dirty texture** and **isolated bright vertical scratch-like lines** without a true Class 1 label. |
| **Class 2** | `baseline_class2_false_positive_1.png` (`f9b98ab64.jpg`), `baseline_class2_false_positive_2.png` (`b08712015.jpg`) | Dominated by **black edge + vertical banding + speckle** — no obvious pitting. Same `f9b98ab64` also appears as Class 1 FN, i.e. ambiguous strip. |
| **Class 3** | `baseline_class3_false_positive_1.png` (`b08f17760.jpg`), `baseline_class3_false_positive_2.png` (`7e18303fd.jpg`) | **Strong vertical glare bands**, right-edge black cutoff, and **large textured/scaly regions** mistaken for Class 3 (p up to ~0.93). |
| **Class 4** | `baseline_class4_false_positive_1.png` (`0d1b42754.jpg`), `baseline_class4_false_positive_2.png` (`ded3e76e3.jpg`) | **Peeling/scaly patches**, diagonal grinding marks, and **dark smudged vertical blobs** on otherwise labeled-clean images. |

**FP theme:** False positives cluster on **process texture, lighting bands, and strip-edge black regions**, not on random clean metal.

---

## CBAM-ResNet18

### False negatives

| Class | Support images | Visible pattern |
|---|---|---|
| **Class 1** | `cbam_class1_false_negative_1.png` (`9d57d9095.jpg`), `cbam_class1_false_negative_2.png` (`fc0541a53.jpg`) | **Edge glare + black bar** (`9d57d9095`) and the **same complex banded strip** as baseline (`fc0541a53`). Thin vertical defect cues are overshadowed. p ≈ 0.09–0.10. |
| **Class 2** | `cbam_class2_false_negative_1.png` (`0c6720401.jpg`), `cbam_class2_false_negative_2.png` (`e97006670.jpg`) | **Identical images to baseline Class 2 FNs** — subtle pits and half-frame black; still p ≈ 0.11. |
| **Class 3** | `cbam_class3_false_negative_1.png` (`82e82b713.jpg`), `cbam_class3_false_negative_2.png` (`8981c24c8.jpg`) | Shared **`82e82b713`** with baseline (bright central patches). Second example is **dense vertical drip/streak** texture (`8981c24c8`) where defect and background texture merge. |
| **Class 4** | `cbam_class4_false_negative_1.png` (`7982045ad.jpg`), `cbam_class4_false_negative_2.png` (`61457b567.jpg`) | Shared **`7982045ad`** with baseline. **`61457b567`** is extreme **underexposure** (~70% black frame) — defect only visible in a narrow lit band (p ≈ 0.16). |

### False positives

| Class | Support images | Visible pattern |
|---|---|---|
| **Class 1** | `cbam_class1_false_positive_1.png` (`acd30bd6a.jpg`), `cbam_class1_false_positive_2.png` (`f3e1e7c47.jpg`) | **Steel-to-black strip edge** and vertical brightness split; moderate confidence (p ≈ 0.57–0.62), lower than baseline Class 1 FPs. |
| **Class 2** | `cbam_class2_false_positive_1.png` (`f9b98ab64.jpg`), `cbam_class2_false_positive_2.png` (`e9f4baa36.jpg`) | Again **edge artifact + banding** (`f9b98ab64` shared with baseline errors; `e9f4baa36` left black wedge). |
| **Class 3** | `cbam_class3_false_positive_1.png` (`b9acd9a0f.jpg`), `cbam_class3_false_positive_2.png` (`c1482dc0c.jpg`) | **Diagonal shadow / half-frame black** and **cross-hatched abrasion + central glare + wavy shadows** — high p (~0.81). |
| **Class 4** | `cbam_class4_false_positive_1.png` (`4340e1e42.jpg`), `cbam_class4_false_positive_2.png` (`d9d9f2b25.jpg`) | **Vertical smear bands** and **localized scale/patch clusters** on clean labels (p ≈ 0.84–0.89). |

---

## Baseline vs CBAM: Same mistakes or different?

### Overlap on the same validation images

| Image ID | Baseline role | CBAM role |
|---|---|---|
| `fc0541a53.jpg` | Class 1 FN | Class 1 FN |
| `0c6720401.jpg`, `e97006670.jpg` | Class 2 FN | Class 2 FN (all four slots) |
| `82e82b713.jpg` | Class 3 FN | Class 3 FN |
| `7982045ad.jpg` | Class 4 FN | Class 4 FN |
| `f9b98ab64.jpg` | Class 1 FN + Class 2 FP | Class 2 FP |

**5 of 8 FN example images (62.5%) recur across models**; Class 2 FN examples are **100% shared**. False-positive sets **mostly differ by class** (different top-confidence mistakes), except **`f9b98ab64`** flagged again for CBAM Class 2 FP.

### Interpretation

1. **Same failure modes:** Both models miss **low-contrast Class 2 pitting** and struggle when **large black edge/coverage regions** or **vertical glare bands** dominate the crop. This aligns with FP triggers (texture, edges, lighting), not with memorizing different semantics per architecture.

2. **Not just more/fewer of identical FP errors:** CBAM’s saved FPs use **different image IDs** for Classes 1, 3, and 4 (often still the same *types* of artifacts: edges, glare, scale-like texture). CBAM Class 1 FPs tend to be **less overconfident** than baseline on comparable edge cases.

3. **CBAM-specific extras:** Unique CBAM FN examples (`9d57d9095`, `8981c24c8`, `61457b567`) stress **glare at strip edge**, **streak-dominated Class 3**, and **catastrophic underexposure** — cases where attention may not recover usable defect signal.

4. **Class 2 and data scarcity:** Class 2 has the **fewest training positives (135)** and the **most consistent cross-model FN overlap**, supporting that rarity + subtle appearance drive misses more than a CBAM-specific blind spot.

---

## Takeaways for the report

- **FNs:** Class 2 and imaging artifacts (edge black, uneven lighting) are the clearest systematic themes; Class 4 FNs are often “almost detected” (higher p).
- **FPs:** Both models fire on **vertical banding, speckle, scale-like texture, and strip boundaries** — review whether augmentations or cropping should suppress edge-dominated patches.
- **CBAM vs baseline:** CBAM **does not** appear to fail on a wholly disjoint set; it **shares the hardest FN images** (especially Class 2) while **reshuffling which high-texture clean strips become Class 3/4 FPs**. Improving shared hard cases (Class 2, edge crops) would likely help both models.
