# streetlight-autoresearch

Detecting poles (streetlights, utility/power poles) in King County multi-year orthoimagery.
Research log: `notes/00-log.md`. Learnings: `notes/01-learnings.md`. Weights: `weights/`.

## Pipeline
| step | script |
|---|---|
| tile access (KC aerial XYZ cache, z20 ≈ 0.10 m) | `src/tiles.py` |
| label download (SCL, Redmond, Renton, Federal Way, Kirkland, KC, OSM) | `src/fetch_labels.py`, `src/osm_extract.py` |
| block dataset (1024 px blocks × 7 years, label-density AOI, spatial split) | `src/build_dataset.py`, `src/merge_index.py` |
| per-block per-year registration | `src/register_years.py` |
| label refinement / set combination with pseudo-labels | `src/refine_labels.py`, `src/combine_sets.py` |
| model (shared encoder per year, max+mean fusion, FPN, heatmap head) | `src/model.py` |
| training / evaluation | `src/train.py`, `src/eval_radii.py`, `src/viz_pred.py` |
| inference over any bbox → GeoJSON | `src/infer.py` |

## Run the detector on an area
```
pip install torch timm pillow numpy scipy requests
cd src
python infer.py --ckpt ../weights/<weights>.pt --bbox -122.35,47.65,-122.33,47.66 \
    --years 2025,2023,2021,2019,2017,2015,2013 --out poles.geojson
```
Tiles are fetched from King County on the fly (cached under `~/tile_cache`). Output points carry a
`score`; ~0.2–0.3 is the F1-optimal threshold, ≥0.5 is high precision.

## Data credits
Imagery: King County GIS (EagleView/Pictometry). Labels: Seattle City Light poles (City of Seattle),
City of Redmond, City of Renton, City of Federal Way, City of Kirkland, King County GIS, OpenStreetMap
contributors (ODbL). See the disclaimers on the OSM Contributors page for each agency.
