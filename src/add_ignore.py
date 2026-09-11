"""Rasterize OSM don't-care polygons into <id>_ignore.png for an existing block set (no re-download).
usage: python add_ignore.py data/scl_v1 labels/osm_ignore_areas.wkt.jsonl.gz"""
import json, os, sys
from build_dataset import rasterize_ignore
d = sys.argv[1]; idx = json.load(open(os.path.join(d, "index.json")))
rasterize_ignore(sys.argv[2], idx, d, idx[0]["size"], idx[0].get("z", 20))
