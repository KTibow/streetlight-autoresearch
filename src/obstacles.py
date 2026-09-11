"""FAA Digital Obstacle File + FCC Antenna Structure Registration -> labels/obstacles.geojson.gz (KC bbox).
Keeps pole-like structure types only. Props: src (dof|asr), cls (mast|tower|pole|tl_tower|antenna), agl_m, accuracy.
usage: python obstacles.py data/obst/dof/DOF.CSV data/obst/asr
"""
import csv, gzip, json, os, sys
BBOX = (-122.54, 47.08, -121.07, 47.78)
DOF_TYPES = {"POLE": "pole", "TOWER": "mast", "T-L TWR": "tl_tower", "UTILITY POLE": "pole", "ANTENNA": "antenna", "GEN UTIL": "pole", "VERTICAL STRUCTURE": "mast"}
ASR_TYPES = {"TOWER": "mast", "MTOWER": "mast", "GTOWER": "mast", "POLE": "pole", "MAST": "mast", "LTOWER": "mast", "PIPE": "pole", "UPOLE": "pole", "NNTANN": "antenna", "NNGTANN": "antenna", "NNMTANN": "antenna", "NNLTANN": "antenna", "NNPOLE": "pole"}


def dms(deg, mn, sec, d):
    v = float(deg) + float(mn) / 60 + float(sec) / 3600
    return -v if d in ("W", "S") else v


def main(dof_csv, asr_dir):
    feats = []
    with open(dof_csv, newline="", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            if r["STATE"].strip() != "WA":
                continue
            lat, lon = float(r["LATDEC"]), float(r["LONDEC"])
            if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
                continue
            t = r["TYPE"].strip()
            if t not in DOF_TYPES:
                continue
            feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[lon, lat]),
                              properties=dict(src="dof", cls=DOF_TYPES[t], dof_type=t, agl_m=round(float(r["AGL"]) * 0.3048, 1), accuracy=r["ACCURACY"].strip(), quantity=int(r["QUANTITY"] or 1))))
    n_dof = len(feats)
    # ASR: RA (type, height) joined with CO (coords) on registration number (field 3)
    ra = {}
    with open(os.path.join(asr_dir, "RA.dat"), errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) > 33 and p[0] == "RA":
                ra[p[2]] = dict(stype=p[32].strip(), agl_m=p[30])
    with open(os.path.join(asr_dir, "CO.dat"), errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) < 16 or p[0] != "CO":
                continue
            try:
                lat = dms(p[6], p[7], p[8], p[9]); lon = dms(p[11], p[12], p[13], p[14])
            except (ValueError, IndexError):
                continue
            if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
                continue
            info = ra.get(p[2], {})
            cls = ASR_TYPES.get(info.get("stype", ""), None)
            if cls is None:
                continue
            try:
                agl = round(float(info.get("agl_m") or 0), 1)
            except ValueError:
                agl = None
            feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[lon, lat]),
                              properties=dict(src="asr", cls=cls, asr_type=info["stype"], agl_m=agl, reg=p[2])))
    out = os.path.join(os.path.dirname(__file__), "..", "labels", "obstacles.geojson.gz")
    with gzip.open(out, "wt") as f:
        json.dump(dict(type="FeatureCollection", features=feats), f)
    import collections
    print(f"dof {n_dof}, asr {len(feats) - n_dof}; cls", collections.Counter(x["properties"]["cls"] for x in feats))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
