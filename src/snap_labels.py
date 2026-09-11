"""Snap imprecise labels (FAA/FCC obstacle points, OSM masts) onto the current model's nearest response,
and protect the unsnapped ones with an ignore disc so the true structure is never a negative.

python snap_labels.py --data data/tall_set --preds preds.json --radius_px 150 --min_score 0.08 --split train
Rewrites index.json points in place (train split only) and unions ignore discs into <id>_ignore.png.
"""
import argparse, json, os, numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--preds", required=True)
ap.add_argument("--radius_px", type=float, default=150); ap.add_argument("--min_score", type=float, default=0.08)
ap.add_argument("--split", default="train"); ap.add_argument("--ignore_disc_px", type=float, default=150)
a = ap.parse_args()
idx_path = os.path.join(a.data, "index.json"); idx = json.load(open(idx_path)); preds = json.load(open(a.preds))
n_snap = n_keep = 0
for it in idx:
    if it["split"] != a.split or it["id"] not in preds:
        continue
    S = it["size"]
    P = np.array(preds[it["id"]], np.float32).reshape(-1, 3); P = P[P[:, 2] >= a.min_score]
    pts = np.array(it["points"], np.float32).reshape(-1, 2); newpts = pts.copy(); unsnapped = []
    for j, (x, y) in enumerate(pts):
        if len(P):
            d = np.hypot(P[:, 0] - x, P[:, 1] - y)
            k = int(np.argmax(np.where(d <= a.radius_px, P[:, 2], -1)))
            if d[k] <= a.radius_px and P[k, 2] >= a.min_score:
                newpts[j] = P[k, :2]; n_snap += 1; continue
        unsnapped.append((x, y)); n_keep += 1
    it["points"] = newpts.round(1).tolist()
    if unsnapped:
        p = os.path.join(a.data, "blocks", f"{it['id']}_ignore.png")
        im = Image.open(p).convert("L") if os.path.exists(p) else Image.new("L", (S, S), 0)
        d = ImageDraw.Draw(im); r = a.ignore_disc_px
        for (x, y) in unsnapped:
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        im.save(p)
json.dump(idx, open(idx_path, "w"))
print(f"snapped {n_snap}, kept+ignore-disc {n_keep}")
