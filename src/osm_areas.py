"""Extract don't-care polygons (parking lots, sports pitches/fields, stadiums) from a Geofabrik .osm.pbf
into labels/osm_ignore_areas.geojson.gz (KC bbox). Ways and multipolygon relations, WGS84."""
import osmium, json, gzip, sys, os
BBOX = (-122.54, 47.08, -121.07, 47.78)
KEEP = [("amenity", "parking"), ("leisure", "pitch"), ("leisure", "stadium"), ("leisure", "track"), ("leisure", "sports_centre"), ("leisure", "golf_course")]
wkt = osmium.geom.WKTFactory()
feats = []
class H(osmium.SimpleHandler):
    def area(self, a):
        t = a.tags
        cls = next((f"{k}={v}" for k, v in KEEP if t.get(k) == v), None)
        if cls is None or (t.get("parking") in ("underground", "multi-storey", "rooftop")):
            return
        try:
            w = wkt.create_multipolygon(a)
        except Exception:
            return
        # crude bbox prefilter on first coordinate
        first = w[w.index("(") :].strip("() ").split(",")[0].split()
        lon, lat = float(first[0]), float(first[1])
        if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
            return
        feats.append(dict(type="Feature", properties=dict(cls=cls, osm_id=a.orig_id()), geometry=dict(type="wkt", coordinates=w)))
h = H(); h.apply_file(sys.argv[1], locations=True, idx="flex_mem")
out = os.path.join(os.path.dirname(__file__), "..", "labels", "osm_ignore_areas.wkt.jsonl.gz")
with gzip.open(out, "wt") as f:
    for ft in feats:
        f.write(json.dumps(ft) + "\n")
import collections; print(len(feats), collections.Counter(ft["properties"]["cls"] for ft in feats))
