"""Refine TRAIN labels with a model's own confident predictions (val labels are never touched).

For each train block:
  - snap: a label with a prediction (score >= snap_thresh) within snap_px is moved onto the prediction
  - add:  a prediction with score >= add_thresh and no label within add_px becomes a new label
  - labels with no supporting prediction are kept (they may be hidden by canopy)
python refine_labels.py --data data/scl_v1 --preds train_preds.json --out index_refined.json
"""
import argparse, json, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--out", default="index_refined.json")
ap.add_argument("--index", default="index.json")
ap.add_argument("--snap_thresh", type=float, default=0.3); ap.add_argument("--snap_px", type=float, default=50)
ap.add_argument("--add_thresh", type=float, default=0.5); ap.add_argument("--add_px", type=float, default=50)
ap.add_argument("--edge", type=int, default=24)
a = ap.parse_args()
idx = json.load(open(os.path.join(a.data, a.index)))
preds = json.load(open(a.preds))
n_snap = n_add = n_lab = 0
for it in idx:
    if it["split"] != "train" or it["id"] not in preds:
        continue
    S = it["size"]
    L = np.array(it["points"], np.float32).reshape(-1, 2)
    P = np.array(preds[it["id"]], np.float32).reshape(-1, 3)
    P = P[(P[:, 0] >= a.edge) & (P[:, 0] < S - a.edge) & (P[:, 1] >= a.edge) & (P[:, 1] < S - a.edge)]
    n_lab += len(L)
    newL = L.copy(); used = np.zeros(len(P), bool)
    # snap (greedy by score)
    order = np.argsort(-P[:, 2])
    for k in order:
        if P[k, 2] < a.snap_thresh or not len(L):
            continue
        d = np.hypot(L[:, 0] - P[k, 0], L[:, 1] - P[k, 1])
        j = int(d.argmin())
        if d[j] <= a.snap_px and not used[k] and np.all(newL[j] == L[j]):
            newL[j] = P[k, :2]; used[k] = True
            if d[j] > 2: n_snap += 1
    # add
    for k in order:
        if used[k] or P[k, 2] < a.add_thresh:
            continue
        if len(newL):
            d = np.hypot(newL[:, 0] - P[k, 0], newL[:, 1] - P[k, 1])
            if d.min() <= a.add_px:
                continue
        newL = np.vstack([newL, P[k, :2]]); n_add += 1
    it["points"] = newL.round(1).tolist()
    it["orig_points"] = L.round(1).tolist()
json.dump(idx, open(os.path.join(a.data, a.out), "w"))
print(f"train labels {n_lab}: snapped(>2px) {n_snap}, added {n_add} -> {n_lab + n_add}")
