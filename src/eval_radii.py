"""Evaluate a checkpoint at several match radii and report localisation error of matched detections.
python eval_radii.py --ckpt runs/x/best.pt --data data/scl_v1 --years 2025,2023,2021,2019 [--thresh 0.2]
"""
import argparse, json, numpy as np, torch
from torch.utils.data import DataLoader
from model import PoleNet, decode_peaks
from train import BlockDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--years", required=True)
    ap.add_argument("--thresh", type=float, default=None); ap.add_argument("--split", default="val")
    ap.add_argument("--radii", default="10,15,20,30,40,60"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--edge", type=int, default=0, help="ignore preds and labels within this many px of the block edge")
    ap.add_argument("--out", default=None); ap.add_argument("--only_set", default=None, help="restrict to items whose 'set' field matches")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    model = PoleNet(ck["args"]["backbone"], pretrained=False); model.load_state_dict(ck["model"]); model.to(a.device).eval()
    years = [int(y) for y in a.years.split(",")]
    ds = BlockDataset(a.data, a.split, years, False, max_years=len(years))
    if a.only_set:
        ds.items = [it for it in ds.items if it.get("set") == a.only_set]
    print(len(ds.items), "blocks")
    dl = DataLoader(ds, 8, num_workers=8)
    threshes = [a.thresh] if a.thresh else [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    radii = [float(r) for r in a.radii.split(",")]
    preds_all, gts_all = [], []
    with torch.no_grad():
        for x, mask, hm, idx, *_ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.to(a.device), mask.to(a.device)).float()
            for b in range(x.shape[0]):
                it = ds.items[int(idx[b])]
                S = it["size"]
                gt = np.array(it["points"], np.float32).reshape(-1, 2)
                pk = decode_peaks(logits[b:b + 1], thresh=0.05)[0]
                pk = np.array([(px * 4 + 2, py * 4 + 2, s) for px, py, s in pk], np.float32).reshape(-1, 3)
                if a.edge:
                    e = a.edge
                    gt = gt[(gt[:, 0] >= e) & (gt[:, 0] < S - e) & (gt[:, 1] >= e) & (gt[:, 1] < S - e)]
                    pk = pk[(pk[:, 0] >= e) & (pk[:, 0] < S - e) & (pk[:, 1] >= e) & (pk[:, 1] < S - e)]
                preds_all.append(pk); gts_all.append(gt)
    res = {}
    for t in threshes:
        for r in radii:
            tp = fp = fn = 0; dists = []
            for pk, gt in zip(preds_all, gts_all):
                pk = pk[pk[:, 2] >= t]; pk = pk[np.argsort(-pk[:, 2])]
                used = np.zeros(len(gt), bool)
                for (px, py, s) in pk:
                    if len(gt):
                        d = np.hypot(gt[:, 0] - px, gt[:, 1] - py); d[used] = 1e9; j = int(d.argmin())
                        if d[j] <= r:
                            used[j] = True; tp += 1; dists.append(d[j]); continue
                    fp += 1
                fn += int((~used).sum())
            p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f1 = 2 * p * rc / max(p + rc, 1e-9)
            res[f"t{t}_r{int(r)}"] = dict(p=round(p, 4), r=round(rc, 4), f1=round(f1, 4), tp=tp, fp=fp, fn=fn, med_dist_px=float(np.median(dists)) if dists else None)
            print(f"thresh {t:.2f} radius {int(r):3d}px ({r*0.1:.1f} m): P {p:.3f} R {rc:.3f} F1 {f1:.3f}  tp {tp} fp {fp} fn {fn}  median match dist {np.median(dists) if dists else 0:.1f}px")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
