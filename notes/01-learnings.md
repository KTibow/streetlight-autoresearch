# Learnings: pole detection from multi-year orthoimagery

(Numbers are on 383 spatially held-out Seattle blocks with Seattle City Light labels unless stated;
"F1@2 m" = point match within 20 px ≈ 2 m at zoom-20 Web Mercator, ≈0.10 m ground pixels.)

## What worked
1. **Poles are detectable from ortho with a small net, quickly.** A ResNet-34 heatmap net reaches
   F1 0.65 after 4 epochs (~1 minute of H100) and ~0.70 after 30–60. The previous attempts failed on
   data, not on vision: with 99k authoritative points to learn from, the model has no trouble telling a
   pole shadow from a lane line or a car edge. It does not need explicit shadow labels.
2. **Multi-year stacking is the single biggest win: +0.085 F1, all of it recall (0.52 → 0.70).**
   Shared per-year encoder, permutation-invariant max+mean fusion of features, random subsets of years
   at train time (year dropout) so inference can use any number of years. Poles hidden by cars,
   deciduous canopy, or a building shadow in one year are visible in another, and sun angle differs
   per flight so at least one year gives a crisp shadow. Using all 7 years at inference beats 4
   (0.703 vs 0.687).
3. **Co-registration is good enough to ignore inside a CNN, but worth measuring.** Years are
   within a few px of each other (2013–2017 are systematically ~0.3–0.4 m off vs 2019+). I estimate
   per-block integer shifts by edge cross-correlation and translate the older years to the 2023 grid;
   the stride-4 heatmap and 0.8 m target sigma absorb the rest. No hand-set (-2, +1) m offset was needed
   for SCL labels vs the KC cache; the labels sit on the pole bases in 2019–2025 imagery.
4. **Self-refinement of labels (snap train labels onto confident predictions ≤5 m) helps a bit
   (0.701 → 0.709) and converges 3× faster.** Val labels stay untouched.

## What limits the score (and why the model is better than the number)
5. **The labels, not the imagery, set the ceiling.** Crops of the highest-confidence "false
   positives" are all real poles (wires converge on them; shadows in every year). Many "misses" have a
   real pole 2–4 m from the label point. The public SCL layer is "generalized for public use":
   positions coarsened, and some poles absent from both public SCL layers. F1 rises from 0.69 to 0.77
   as the match radius grows 2 → 6 m, and matched detections sit a median 0.7 m from the label. A
   human audit of high-confidence detections, not the raw F1, is the honest quality measure.
6. **Short poles are the hard class.** Recall by SCL pole height: <20 ft 0.29, 20–34 ft 0.42,
   35–49 ft 0.74, ≥50 ft 0.82. Tall wood poles have long shadows and a fat crossarm; a 15 ft pedestrian
   light has neither at 10 cm/px.
7. **Generalization follows the label distribution, not the geography.** Scored on Redmond (Eastside,
   PSE-owned poles, city streetlight layer): utility poles with lights recall 0.81 (same as Seattle),
   steel/concrete light standards 0.43, pedestrian lights 0.04, decorative 0.00. The detector learned
   "wood distribution pole" because that is 90%+ of SCL's layer. Fix: add city streetlight layers to
   training (v2 below).
8. **Evergreen canopy is the one occlusion multi-year cannot fix.** Missed Redmond lights sit under
   dense conifers along arterials; no season gives a look. LiDAR (or oblique imagery) is the only
   route for those.

## Implications for ortho + image AI work
- Frame these as **point heatmap regression at stride 4**, not boxes: labels are points, objects are
  1–3 px wide, and CenterNet-style focal loss handles the 1:10⁵ class imbalance without hard-negative
  mining.
- **Stack epochs, don't pick one.** Anything that casts a shadow or gets occluded benefits. Keep the
  fusion permutation-invariant so any subset of years works at inference, and drop years at train time.
- **Do not trust a pixel-radius metric against municipal GIS layers below ~3 m.** Report a radius
  sweep and audit the top-scored disagreements visually; treat the model as a label QA tool.
- **Label coverage beats label count.** 99k poles of one type taught one type. A few hundred blocks
  of the missing class matter more than another 50k of the dominant one.
- **Self-training on the model's own train-set predictions barely adds points (44 of 8k)** because a
  model memorizes its training labels; pseudo-labels for a *new* area/class from a model that never
  saw it (as done for the city sets) are the useful kind.
- Infra notes: KC tiles at z20 are the true resolution ceiling (finer exports are upsampled); ~550
  mosaics/min from one sandbox; a 5.5 GB dataset moves sandbox → S3 → node in ~7 min; H100 epochs on
  2k blocks take 12 s, so the whole study cost a few dollars of GPU. The wall clock went to data.

## v2 (SCL + city streetlight layers + pseudo-labelled utility poles)
Adding ~430 blocks from Redmond/Renton/Federal Way/Kirkland streetlight layers (with the v1 model's
confident detections filling in their unlabelled utility poles) and switching to ConvNeXt-Tiny:

| model | Seattle F1@2 m | Seattle @3 m | Redmond R@2 m | Renton R@2 m |
|---|---|---|---|---|
| v1 ResNet-34 (run 1) | 0.703 | 0.73 | 0.35 | – |
| v1 ResNet-34 refined labels | 0.716 | 0.767 | 0.42 | 0.735 |
| v2 ResNet-34 | 0.711 | – | 0.54 | 0.75 |
| **v2 ConvNeXt-Tiny (final)** | **0.727** (0.730 at thr 0.25) | **0.776** | **0.58** | 0.735 |

Years at inference matter even for the final model: Seattle F1@2 m 0.730 with 7 years, 0.705 with 4,
0.594 with only 2025. Federal Way's legacy layer scores ~0.06 for every model; its points are corner
signal/light masts with coarse positions and I treat it as not-ground-truth rather than a model failure.
Redmond's remaining misses are pedestrian/decorative lights under 20 ft and poles under evergreens.

Cost: ≈ $8 of H100 time for everything above (7 training runs, all evaluation), ~3 h wall clock for
data, ~2 h for training and analysis.
