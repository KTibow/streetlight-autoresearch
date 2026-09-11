"""Combine several block sets into one dataset dir (blocks symlinked, one index.json).

python combine_sets.py --out data/v2 --set scl:data/scl_v1:index_refined.json --set redmond:data/redmond_eval:index.json:preds.json ...
Each --set is name:dir[:train_index[:preds_json]]. Val items always come from the set's index.json.
If preds_json is given, confident predictions (>= add_thresh, >= add_px from any label, away from the
edge) are ADDED to that set's TRAIN labels (pseudo-labels for poles the source layer does not cover).
"""
import argparse, json, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--set", action="append", required=True)
ap.add_argument("--add_thresh", type=float, default=0.5); ap.add_argument("--add_px", type=float, default=50); ap.add_argument("--edge", type=int, default=24)
ap.add_argument("--val_frac_override", type=float, default=None, help="re-split a set by cell hash with this val fraction (for sets built all-val)")
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "blocks"), exist_ok=True)
out = []
for spec in a.set:
    parts = spec.split(":"); name, d = parts[0], parts[1]
    tr_index = parts[2] if len(parts) > 2 and parts[2] else "index.json"
    preds = json.load(open(parts[3])) if len(parts) > 3 and parts[3] else None
    val_items = {it["id"]: it for it in json.load(open(os.path.join(d, "index.json")))}
    tr_items = {it["id"]: it for it in json.load(open(os.path.join(d, tr_index)))}
    n_add = 0
    for bid, vit in val_items.items():
        split = vit["split"]
        if a.val_frac_override is not None and name != "scl":
            S = vit["size"]; cell = (vit["px0"] // (S * 4), vit["py0"] // (S * 4))
            h = (cell[0] * 73856093 ^ cell[1] * 19349663) % 1000
            split = "val" if h < a.val_frac_override * 1000 else "train"
        it = dict(tr_items.get(bid, vit)) if split == "train" else dict(vit)
        it["split"] = split; it["set"] = name
        if split == "train" and preds is not None and bid in preds:
            S = it["size"]
            L = np.array(it["points"], np.float32).reshape(-1, 2)
            P = np.array(preds[bid], np.float32).reshape(-1, 3)
            P = P[(P[:, 2] >= a.add_thresh) & (P[:, 0] >= a.edge) & (P[:, 0] < S - a.edge) & (P[:, 1] >= a.edge) & (P[:, 1] < S - a.edge)]
            for p in P[np.argsort(-P[:, 2])]:
                if len(L) and np.hypot(L[:, 0] - p[0], L[:, 1] - p[1]).min() <= a.add_px:
                    continue
                L = np.vstack([L, p[:2]]); n_add += 1
            it["points"] = L.round(1).tolist()
            pwl = list(it.get("point_weight", [])); it["point_weight"] = (pwl + [1.0] * len(L))[:len(L)]
        for y in it["years"]:
            dst = os.path.join(a.out, "blocks", f"{bid}_{y}.jpg")
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(os.path.join(d, "blocks", f"{bid}_{y}.jpg")), dst)
        ig = os.path.join(d, "blocks", f"{bid}_ignore.png")
        if os.path.exists(ig) and not os.path.lexists(os.path.join(a.out, "blocks", f"{bid}_ignore.png")):
            os.symlink(os.path.abspath(ig), os.path.join(a.out, "blocks", f"{bid}_ignore.png"))
        out.append(it)
    ntr = sum(i["split"] == "train" and i["set"] == name for i in out); nva = sum(i["split"] == "val" and i["set"] == name for i in out)
    print(f"{name}: train {ntr} val {nva} blocks, pseudo-labels added {n_add}")
json.dump(out, open(os.path.join(a.out, "index.json"), "w"))
print("total", len(out), "blocks;", sum(len(i["points"]) for i in out if i["split"] == "train"), "train points")
