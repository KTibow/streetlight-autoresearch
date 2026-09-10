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
