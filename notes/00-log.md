# Research log

## 2026-09-10 — kickoff

Goal: a model that detects poles (streetlights, utility/power poles, masts) in King County
orthoimagery, runnable over an arbitrary King County area, plus learnings about ortho + image AI.

Budget: $25 on givemeanode (H100 ≈ $3.60/h list, cheaper on Fridays). Plan: all data prep in the
sandbox container (4 cores, 15 GB RAM), rent the GPU only for training.

### Setup
- Test push to GitHub: **failed with 403** on both `git push` and the GitHub API (branch creation
  "Resource not accessible by integration"). Reads work. The Claude GitHub App installation on this
  repo appears to lack write (contents) permission. Committing locally, retrying later.
- Three scouting subagents launched: KC ortho endpoints (multi-year), pole label datasets
  (Seattle City Light, KC GIS, cities, WSDOT, OSM), Mapillary map features.

### Approach (initial)
- Labels: point datasets of poles → Gaussian heatmap targets on ortho chips.
- Input: stack of several ortho years at the same footprint (co-registered after the known
  ~(-2, +1) m offset), so the net can exploit differing shadow directions; also single-year variant
  as a baseline to measure what the multi-year stack buys.
- Model: small UNet-style keypoint/heatmap net (pretrained ResNet/ConvNeXt encoder), peak-picking
  → point detections. Metric: precision/recall at a distance threshold (e.g. 3 m).

### Ortho scout results (verified)
- KC aerials are cached Web Mercator XYZ tiles: `.../BaseMaps/KingCo_Aerial_{year}/MapServer/tile/{z}/{y}/{x}`.
- z20 (≈0.149 m nominal, ≈0.10 m ground at 47.6°) for 2013, 2015, 2017, 2019, 2021, 2023, 2025; z19 for 2007/2009/2012.
  `export` at finer than z20 just upsamples the cache, so z20 tiles are the true ceiling.
- No rate limits seen; 40 concurrent tile requests fine. Tile fetcher: `src/tiles.py`.
- Phase correlation between 2021/2023/2025 blocks at one site: 0–1 px relative offset. Years are well
  co-registered relative to each other (the user's ~(-2,+1) m offset is labels→imagery, to be estimated).

### Mapillary scout results (verified)
- Graph `map_features` bbox search silently truncates (a 1 km bbox returned 1.8k while the vector tile
  holds 97k) and has no pagination; only the z14 vector tiles
  (`tiles.mapillary.com/maps/vtp/mly_map_feature_point/2/14/{x}/{y}`) are complete. Layer `point`,
  props `id, value, first_seen_at, last_seen_at`.
- Classes: `object--street-light`, `object--support--utility-pole`, `object--support--pole`.
  County estimate 0.4–1M features, but **massively duplicated** (downtown: 62k street-light features in
  14k distinct 10 m cells; one cell had 1,852) and positions are off by "a few metres" (Mapillary's
  own words), worse in downtown canyons.
- Verdict: not a primary training label source (needs dedupe, positional error ≈ the pole spacing of
  interest, incomplete off the driven network). Possible use: weak labels / a sanity check outside
  cities with authoritative pole datasets.

### Label scout results (verified)
Primary: **Seattle City Light poles** (`Seattle_City_Light_Poles_PROD/FeatureServer/1`, 99,114 points,
Seattle + Shoreline/Burien/Tukwila; fields FACILITYTYPE O/S/X, HEIGHT, HAS_STREETLIGHT 75% yes).
Overlay check on 2023 imagery (Wallingford): points sit on the pole bases where the wires converge, i.e.
no visible label→image offset at this site/year. Also pulled: Redmond (5.3k, rich attrs), Renton (5k),
Federal Way (4.5k), Kirkland (2.2k), Shoreline SCL subset (5.9k), KC traffic poles (1.9k).
OSM (Overpass, private.coffee mirror): ~52k pole-ish nodes in the KC bbox; Geofabrik WA extract only over
plain HTTP through this proxy. WSDOT: nothing public. PSE: nothing public → the Eastside and
unincorporated KC have no authoritative ground truth except OSM.

### Dataset v1 (building)
- Training area derived from SCL label density: 4×4-block (~410 m) cells with ≥12 SCL points and all
  4 neighbours labelled → 1,286 cells, 20,576 candidate blocks. Sampled 2,100 blocks with poles + 286
  empty; spatial split by cell → 2,003 train / 383 val.
- Blocks are 1024 px at z20 (~103 m ground), 7 years (2013–2025), JPEG q92.

### Inter-year registration (measured on 300 blocks, edge cross-correlation vs 2023)
| year | median (dx,dy) px | within 6 px | >10 px |
|---|---|---|---|
| 2013 | (-3, +4) | 48% | 21% |
| 2015 | (-2, +3) | 76% | 11% |
| 2017 | (-2, +3) | 71% | 11% |
| 2019 | (0, 0) | 85% | 10% |
| 2021 | (0, 0) | 84% | 8% |
| 2025 | (0, 0) | 77% | 13% |

1 px ≈ 0.10 m ground. The Pictometry-era years (2013–2017) are shifted ~0.3–0.4 m relative to the
EagleView years, with a wider per-block spread in 2013. Some ">10 px" cases are correlation failures
(trees, redevelopment) rather than true offsets. Fix: `src/register_years.py` stores per-block per-year
integer shifts (clamped to ±16 px, dropped when the correlation peak is not distinct) in the index and
the loader translates each year to the 2023 grid. Labels themselves sit on the recent imagery.

Full-dataset registration (2,386 blocks, vs 2023): 2013 median (0,+2) px with 39% of blocks >6 px;
2015/2017 median (-2,+3)/(-2,+2), 16–18% >6 px; 2019/2021 6% >6 px; 2025 11% >6 px. Shifts stored in
the index and applied at load time. Dataset shipped to the H100 node as a 5.5 GB context
(upload ~35 MB/s per stream ×4; S3 us-west-2 → europe-north1 node ~14 MB/s).

### Run 1: `multi_r34`
ResNet-34 shared encoder, up to 4 of 7 years per sample (year dropout 0.3), 512 px crops, bs 16,
AdamW 3e-4 one-cycle, 30 epochs, focal heatmap loss at stride 4 (sigma 2 = 0.8 m), match radius 20 px (2 m).

### Results, runs 1–2 (30 epochs each, val = 383 spatially held-out blocks, 1,563 SCL poles)
| run | years/sample | F1@2 m | P | R | best thresh |
|---|---|---|---|---|---|
| multi_r34 | up to 4 of 7 | **0.688** | 0.680 | 0.695 | 0.20 |
| single_r34 | 1 | 0.603 | 0.711 | 0.523 | 0.25 |

Multi-year is worth +0.085 F1, almost all of it recall (0.52 → 0.70): a pole invisible in one year
(canopy, cars, shadow of a building) is visible in another. Inference with all 7 years instead of 4:
F1@2 m 0.703. Match radius sensitivity (run 1, 7 years, edge 24 px ignored):
1 m 0.51 · 1.5 m 0.63 · 2 m 0.69 · 3 m 0.73 · 4 m 0.75 · 6 m 0.77. Median matched distance 0.7 m.
Qualitative: several "errors" are a miss + false positive pair 3–4 m apart (label offset / pole
replaced nearby); the rest are subtle poles under canopy and in wide medians, plus block-edge FPs.

### Error analysis, run 1 (val, thresh 0.2, 2 m radius; 1,093 TP / 470 FN / 484 FP)
Recall by SCL pole HEIGHT: <20 ft **0.29** (n=147) · 20–34 ft 0.42 (n=134) · 35–49 ft 0.74 (n=702) ·
≥50 ft **0.82** (n=580). HAS_STREETLIGHT yes 0.73 vs no 0.60. FACILITYTYPE S (n=87) 0.17.
→ Misses are dominated by short poles (small/no shadow, small top). 44 of 484 FPs are within 24 px of
the block edge (an artifact the inference script already crops away); only 20 FPs are within 4 m of an
OSM pole-ish node, so "unlabelled but mapped" poles do not explain the FPs.
