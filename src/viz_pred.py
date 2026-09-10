"""Render predictions vs ground truth for val blocks.
python viz_pred.py --ckpt runs/x/best.pt --data data/scl_v1 --years 2025,2023,2021,2019 --n 6 --out viz/
Green = TP, red = FN (missed label), magenta = FP. Right panel: heatmap on the first year.
"""
import argparse, json, os, numpy as np, torch
from PIL import Image, ImageDraw
from model import PoleNet, decode_peaks
from train import BlockDataset, match_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--years", required=True)
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--out", required=True); ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--split", default="val"); ap.add_argument("--dist_px", type=float, default=20); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    model = PoleNet(ck["args"]["backbone"], pretrained=False); model.load_state_dict(ck["model"]); model.to(a.device).eval()
    thresh = a.thresh if a.thresh is not None else float(ck.get("thresh", 0.4))
    years = [int(y) for y in a.years.split(",")]
    ds = BlockDataset(a.data, a.split, years, False, max_years=len(years))
    os.makedirs(a.out, exist_ok=True)
    idxs = [i for i, it in enumerate(ds.items) if a.ids is None or it["id"] in a.ids.split(",")][: a.n]
    for i in idxs:
        x, mask, hm, _ = ds[i]
        it = ds.items[i]
        with torch.no_grad():
            logits = model(x[None].to(a.device), mask[None].to(a.device)).float().cpu()
        p = torch.sigmoid(logits)[0, 0].numpy()
        peaks = [(px * 4 + 2, py * 4 + 2, s) for px, py, s in decode_peaks(logits, thresh)[0]]
        gt = np.array(it["points"], np.float32).reshape(-1, 2)
        # matching for colours
        used = np.zeros(len(gt), bool); tps = []
        for (px, py, s) in sorted(peaks, key=lambda t: -t[2]):
            if len(gt):
                d = np.hypot(gt[:, 0] - px, gt[:, 1] - py); d[used] = 1e9; j = int(d.argmin())
                if d[j] <= a.dist_px:
                    used[j] = True; tps.append((px, py)); continue
            tps.append(None)
        base = Image.fromarray(x[0].permute(1, 2, 0).numpy().astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(base)
        for k, (px, py, s) in enumerate(sorted(peaks, key=lambda t: -t[2])):
            col = "lime" if tps[k] is not None else "magenta"
            d.ellipse([px - 9, py - 9, px + 9, py + 9], outline=col, width=3)
        for j, (gx, gy) in enumerate(gt):
            if not used[j]:
                d.ellipse([gx - 9, gy - 9, gx + 9, gy + 9], outline="red", width=3)
        hmimg = Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR).convert("RGB")
        hmimg = Image.blend(base, hmimg, 0.6)
        W = Image.new("RGB", (base.width * 2, base.height)); W.paste(base, (0, 0)); W.paste(hmimg, (base.width, 0))
        tp = sum(t is not None for t in tps); fp = len(tps) - tp; fn = int((~used).sum())
        W.save(os.path.join(a.out, f"{it['id']}_tp{tp}_fp{fp}_fn{fn}.jpg"), quality=85)
        print(it["id"], "tp", tp, "fp", fp, "fn", fn)


if __name__ == "__main__":
    main()
