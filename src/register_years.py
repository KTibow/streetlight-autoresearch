"""Per-block, per-year integer registration against a reference year via edge cross-correlation.
Writes `shifts` {year: [dx, dy]} into the index; loaders translate each year's image by (dx, dy) so it
aligns with the reference year (and hence with the labels, which sit on the recent imagery).
usage: python register_years.py --data data/scl_v1 [--ref 2023] [--max 16] [--procs 4]
"""
import argparse, json, os, numpy as np
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve


def prep(p, lo=128, hi=896):
    a = np.asarray(Image.open(p).convert("L")).astype(np.float32)[lo:hi, lo:hi]
    g = np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1)); g = np.minimum(g, np.percentile(g, 99))
    return g - g.mean()


def xshift(a, b):
    c = fftconvolve(a, b[::-1, ::-1], mode="same"); iy, ix = np.unravel_index(c.argmax(), c.shape)
    # sharpness: peak vs. the best value outside a 5px neighbourhood
    m = c.copy(); m[max(0, iy - 5):iy + 6, max(0, ix - 5):ix + 6] = -np.inf
    return int(ix - a.shape[1] // 2), int(iy - a.shape[0] // 2), float(c.max() / max(m.max(), 1e-6))


def work(args):
    root, it, ref, mx = args
    years = it["years"]
    if not years:
        return it["id"], {}
    r = ref if ref in years else max(years)
    a = prep(os.path.join(root, "blocks", f"{it['id']}_{r}.jpg"))
    out = {str(r): [0, 0, 1.0]}
    for y in years:
        if y == r:
            continue
        dx, dy, q = xshift(a, prep(os.path.join(root, "blocks", f"{it['id']}_{y}.jpg")))
        if max(abs(dx), abs(dy)) > mx or q < 1.05:
            dx, dy = 0, 0
        out[str(y)] = [dx, dy, round(q, 3)]
    return it["id"], out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ref", type=int, default=2023)
    ap.add_argument("--max", type=int, default=16); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--index", default="index.json")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.data, a.index)))
    todo = [(a.data, it, a.ref, a.max) for it in idx if "shifts" not in it]
    print(len(todo), "blocks to register", flush=True)
    res = {}
    with Pool(a.procs) as p:
        for i, (bid, sh) in enumerate(p.imap_unordered(work, todo, chunksize=4)):
            res[bid] = sh
            if (i + 1) % 100 == 0:
                print(i + 1, flush=True)
    for it in idx:
        if it["id"] in res:
            it["shifts"] = res[it["id"]]
    json.dump(idx, open(os.path.join(a.data, a.index), "w"))
    # summary
    import collections
    st = collections.defaultdict(list)
    for it in idx:
        for y, (dx, dy, q) in it.get("shifts", {}).items():
            st[y].append((dx, dy))
    for y in sorted(st):
        d = np.array(st[y]); print(y, "n", len(d), "median", np.median(d, 0), "nonzero frac", np.mean(np.abs(d).max(1) > 0).round(2), ">6px", np.mean(np.abs(d).max(1) > 6).round(2))
