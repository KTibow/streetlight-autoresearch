"""Train the pole heatmap detector.

Dataset layout (built by src/build_dataset.py):
  data/<split>/index.json   list of samples: {id, years:[..], points:[[px,py],..], meta...}
  data/<split>/chips/<id>_<year>.png   RGB chip, all years share the footprint/pixel grid

Usage: python train.py --data data --out runs/exp1 --years 2021,2019,2017 --epochs 20
"""
import argparse, json, os, random, time, math
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import PoleNet, focal_heatmap_loss, decode_peaks


def gaussian_heatmap(points, h, w, sigma):
    hm = np.zeros((h, w), np.float32)
    if len(points) == 0:
        return hm
    ys, xs = np.mgrid[0:h, 0:w]
    for (px, py) in points:
        if not (0 <= px < w and 0 <= py < h):
            continue
        g = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))
        np.maximum(hm, g, out=hm)
    # force exact 1 at integer peak
    for (px, py) in points:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < w and 0 <= iy < h:
            hm[iy, ix] = 1.0
    return hm


class ChipDataset(Dataset):
    def __init__(self, root, split, years, train, size=None, stride=4, sigma_px=2.0, max_years=None, year_dropout=0.0):
        self.root = os.path.join(root, split)
        self.items = json.load(open(os.path.join(self.root, "index.json")))
        self.years = years
        self.train = train
        self.stride = stride
        self.sigma = sigma_px
        self.max_years = max_years or len(years)
        self.year_dropout = year_dropout
        self.size = size

    def __len__(self):
        return len(self.items)

    def load(self, it, year):
        p = os.path.join(self.root, "chips", f"{it['id']}_{year}.png")
        return np.asarray(Image.open(p).convert("RGB"))

    def __getitem__(self, i):
        it = self.items[i]
        avail = [y for y in self.years if y in it["years"]]
        if self.train and self.year_dropout > 0 and len(avail) > 1:
            avail = [y for y in avail if random.random() > self.year_dropout] or [random.choice(avail)]
        avail = avail[: self.max_years]
        imgs = [self.load(it, y) for y in avail]
        H, W = imgs[0].shape[:2]
        pts = np.array(it["points"], np.float32).reshape(-1, 2)
        x = np.stack(imgs, 0)  # Y,H,W,3
        if self.train:
            # random flips / 90-degree rotations (shadows rotate consistently across years)
            k = random.randint(0, 3)
            x = np.rot90(x, k, axes=(1, 2))
            for _ in range(k):  # rotate points by 90deg CCW like np.rot90 on axes (H,W)
                pts = np.stack([pts[:, 1], W - 1 - pts[:, 0]], 1) if len(pts) else pts
                H, W = W, H
            if random.random() < 0.5:
                x = x[:, :, ::-1]
                if len(pts):
                    pts[:, 0] = W - 1 - pts[:, 0]
            x = np.ascontiguousarray(x)
        hm = gaussian_heatmap([(p[0] / self.stride, p[1] / self.stride) for p in pts], H // self.stride, W // self.stride, self.sigma)
        Y = len(avail)
        mask = np.zeros(self.max_years, bool); mask[:Y] = True
        if Y < self.max_years:
            x = np.concatenate([x, np.zeros((self.max_years - Y,) + x.shape[1:], x.dtype)], 0)
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # Y,3,H,W uint8
        if self.train:
            # photometric jitter per year
            x = x.float()
            for yi in range(Y):
                a = 1 + (random.random() - 0.5) * 0.4
                b = (random.random() - 0.5) * 40
                x[yi] = (x[yi] * a + b).clamp(0, 255)
        return x, torch.from_numpy(mask), torch.from_numpy(hm)[None], i


def match_points(pred, gt, thresh):
    """Greedy matching by distance; returns tp, fp, fn."""
    pred = sorted(pred, key=lambda t: -t[2])
    used = np.zeros(len(gt), bool)
    tp = 0
    for (x, y, s) in pred:
        if len(gt) == 0:
            break
        d = np.hypot(gt[:, 0] - x, gt[:, 1] - y)
        d[used] = 1e9
        j = int(d.argmin())
        if d[j] <= thresh:
            used[j] = True
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


@torch.no_grad()
def evaluate(model, loader, ds, device, stride, dist_px, threshes=(0.2, 0.3, 0.4, 0.5, 0.6)):
    model.eval()
    stats = {t: [0, 0, 0] for t in threshes}
    for x, mask, hm, idx in loader:
        x, mask = x.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, mask)
        logits = logits.float()
        for t in threshes:
            peaks = decode_peaks(logits, thresh=t)
            for b, pk in enumerate(peaks):
                gt = np.array(ds.items[int(idx[b])]["points"], np.float32).reshape(-1, 2)
                # eval loader has train=False so no augmentation; gt in chip px
                pk = [(px * stride, py * stride, s) for px, py, s in pk]
                tp, fp, fn = match_points(pk, gt, dist_px)
                s = stats[t]; s[0] += tp; s[1] += fp; s[2] += fn
    out = {}
    for t, (tp, fp, fn) in stats.items():
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        out[t] = dict(p=p, r=r, f1=2 * p * r / max(p + r, 1e-9), tp=tp, fp=fp, fn=fn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", required=True, help="comma list, e.g. 2021,2019,2017")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=2.0, help="heatmap sigma in heatmap px (stride 4)")
    ap.add_argument("--dist_px", type=float, default=20, help="match radius in chip px")
    ap.add_argument("--year_dropout", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    years = [int(y) for y in args.years.split(",")]
    device = "cuda"
    tr = ChipDataset(args.data, "train", years, True, sigma_px=args.sigma, year_dropout=args.year_dropout)
    va = ChipDataset(args.data, "val", years, False, sigma_px=args.sigma)
    tl = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    vl = DataLoader(va, args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = PoleNet(args.backbone).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best = -1
    print(f"train {len(tr)} val {len(va)} steps {steps}", flush=True)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0; n = 0
        for x, mask, hm, _ in tl:
            x, mask, hm = x.to(device, non_blocking=True), mask.to(device), hm.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, mask)
            loss = focal_heatmap_loss(logits.float(), hm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        rec = dict(epoch=ep, loss=tot / n, time=time.time() - t0)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            ev = evaluate(model, vl, va, device, 4, args.dist_px)
            rec["val"] = ev
            bt = max(ev, key=lambda t: ev[t]["f1"])
            rec["best_thresh"] = bt; rec["f1"] = ev[bt]["f1"]
            if ev[bt]["f1"] > best:
                best = ev[bt]["f1"]
                torch.save(dict(model=model.state_dict(), args=vars(args), thresh=bt, f1=best), os.path.join(args.out, "best.pt"))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
    torch.save(dict(model=model.state_dict(), args=vars(args)), os.path.join(args.out, "last.pt"))
    json.dump(dict(best_f1=best), open(os.path.join(args.out, "result.json"), "w"))


if __name__ == "__main__":
    main()
