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
