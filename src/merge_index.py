"""Merge builder indexes into index.json: years rescanned from disk, existing shifts preserved.
usage: python merge_index.py data/scl_v1 [--years 2013,...] [--min_years 3]"""
import json, os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--years", default="2013,2015,2017,2019,2021,2023,2025"); ap.add_argument("--min_years", type=int, default=3)
a = ap.parse_args()
years = [int(y) for y in a.years.split(",")]
recs = {}
for name in ["index.json", "index_rev.json", "index_merged.json"]:
    p = os.path.join(a.data, name)
    if os.path.exists(p):
        for it in json.load(open(p)):
            old = recs.get(it["id"], {})
            if "shifts" in old and "shifts" not in it:
                it["shifts"] = old["shifts"]
            recs[it["id"]] = it
files = set(os.listdir(os.path.join(a.data, "blocks")))
out = []
for it in recs.values():
    it["years"] = [y for y in years if f"{it['id']}_{y}.jpg" in files]
    if "shifts" in it and set(map(str, it["years"])) != set(it["shifts"]):
        del it["shifts"]
    if len(it["years"]) >= a.min_years:
        out.append(it)
out.sort(key=lambda r: r["id"])
json.dump(out, open(os.path.join(a.data, "index_merged.json"), "w"))
full = sum(len(r["years"]) == len(years) for r in out)
print(f"{len(out)} blocks ({full} with all {len(years)} years); train {sum(r['split']=='train' for r in out)} val {sum(r['split']=='val' for r in out)}; with shifts {sum('shifts' in r for r in out)}")
