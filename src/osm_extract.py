"""Extract pole-ish nodes from a Geofabrik .osm.pbf into labels/osm.geojson.gz (KC bbox only)."""
import osmium, json, gzip, sys, os
BBOX = (-122.54, 47.08, -121.07, 47.78)
KEEP = {("highway", "street_lamp"), ("power", "pole"), ("power", "tower"), ("man_made", "utility_pole"), ("man_made", "mast"), ("man_made", "flagpole")}
class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.feats = []
    def node(self, n):
        if not n.location.valid(): return
        lon, lat = n.location.lon, n.location.lat
        if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]): return
        t = dict(n.tags)
        for k, v in KEEP:
            if t.get(k) == v:
                self.feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[lon, lat]),
                                       properties=dict(osm_id=n.id, cls=v, tags={kk: vv for kk, vv in t.items() if kk in ("lamp_mount", "lamp_type", "height", "material", "support", "operator", "highway", "power", "man_made")})))
                break
h = H(); h.apply_file(sys.argv[1], locations=False)
out = os.path.join(os.path.dirname(__file__), "..", "labels", "osm.geojson.gz")
with gzip.open(out, "wt") as f: json.dump(dict(type="FeatureCollection", features=h.feats), f)
print(len(h.feats), "features")
