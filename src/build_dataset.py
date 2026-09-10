"""Build a block dataset: for each ~1024px (z20) block containing labelled poles, fetch every year's
mosaic and record pole pixel positions.

Inputs:
  --points  GeoJSON FeatureCollection of Point features (EPSG:4326). Optional properties: cls, src.
  --aoi     GeoJSON polygon(s) inside which labels are considered COMPLETE (blocks are only sampled here).
Outputs (under --out):
  blocks/<id>_<year>.jpg       1024x1024 mosaics in the z20 pixel grid
  index.json                   [{id, px0, py0, size, years:[..], points:[[x,y],..], split}]
"""
import argparse, json, os, random, sys, math
import numpy as np
from PIL import Image
from shapely.geometry import shape, Point, box
from shapely.strtree import STRtree
import tiles


def load_points(path):
    import gzip
    gj = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    pts = []
    for f in gj["features"]:
        g = f["geometry"]
        if g["type"] != "Point":
            continue
        lon, lat = g["coordinates"][:2]
        pts.append((lon, lat, f.get("properties", {})))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--aoi", default=None, help="GeoJSON polygons where labels are complete; if omitted, derived from label density")
    ap.add_argument("--cover_cells", type=int, default=4, help="coverage cell = this many blocks per side")
    ap.add_argument("--cover_min", type=int, default=12, help="min points per coverage cell to count as labelled area")
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", default="2013,2015,2017,2019,2021,2023,2025")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--z", type=int, default=20)
    ap.add_argument("--max_blocks", type=int, default=100000)
    ap.add_argument("--empty_frac", type=float, default=0.15, help="fraction of kept blocks that have zero poles")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--offset_m", default="0,0", help="label shift dx,dy metres (east,north) applied before projecting")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()
    random.seed(args.seed)
    years = [int(y) for y in args.years.split(",")]
    S = args.size
    dx_m, dy_m = [float(v) for v in args.offset_m.split(",")]

    pts = []
    for pth in args.points.split(","):
        pts += load_points(pth)
    print(f"{len(pts)} points", flush=True)

    # project points to z20 global pixels (with metre offset applied in mercator, scaled by 1/cos(lat))
    ppx = []
    for lon, lat, props in pts:
        x, y = tiles.lonlat_to_merc(lon, lat)
        k = 1 / math.cos(math.radians(lat))
        x += dx_m * k; y += dy_m * k
        px, py = tiles.merc_to_pixel(x, y, args.z)
        ppx.append((px, py, lon, lat, props))
    P = np.array([(p[0], p[1]) for p in ppx])

    # candidate blocks
    blocks = []
    if args.aoi is None:
        C = S * args.cover_cells
        cells = {}
        for px, py in P:
            c = (int(px) // C, int(py) // C)
            cells[c] = cells.get(c, 0) + 1
        good = {c for c, n in cells.items() if n >= args.cover_min}
        # require the 4-neighbourhood to be labelled too (avoid edge-of-coverage blocks)
        good = {c for c in good if all(((c[0] + dx, c[1] + dy) in good) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))}
        for (cx, cy) in good:
            for j in range(args.cover_cells):
                for i in range(args.cover_cells):
                    blocks.append((cx * C + i * S, cy * C + j * S))
        polys = []
        print(f"{len(cells)} cells with points, {len(good)} labelled cells", flush=True)
    else:
        aoi = json.load(open(args.aoi))
        polys = [shape(f["geometry"]) for f in aoi["features"]] if "features" in aoi else [shape(aoi)]
    for poly in polys:
        minx, miny, maxx, maxy = poly.bounds
        x0, y0 = tiles.merc_to_pixel(*tiles.lonlat_to_merc(minx, maxy), args.z)
        x1, y1 = tiles.merc_to_pixel(*tiles.lonlat_to_merc(maxx, miny), args.z)
        gx0, gy0 = int(x0) // S * S, int(y0) // S * S
        for by in range(gy0, int(y1) + 1, S):
            for bx in range(gx0, int(x1) + 1, S):
                # block polygon in lonlat
                lon0, lat0 = tiles.merc_to_lonlat(*tiles.pixel_to_merc(bx, by, args.z))
                lon1, lat1 = tiles.merc_to_lonlat(*tiles.pixel_to_merc(bx + S, by + S, args.z))
                b = box(lon0, lat1, lon1, lat0)
                if poly.contains(b):  # only fully-inside blocks: labels complete
                    blocks.append((bx, by))
    blocks = sorted(set(blocks))
    print(f"{len(blocks)} candidate blocks", flush=True)

    # assign points to blocks
    recs = []
    for bx, by in blocks:
        m = (P[:, 0] >= bx) & (P[:, 0] < bx + S) & (P[:, 1] >= by) & (P[:, 1] < by + S)
        idx = np.nonzero(m)[0]
        points = [[float(P[i, 0] - bx), float(P[i, 1] - by)] for i in idx]
        props = [ppx[i][4] for i in idx]
        recs.append(dict(id=f"b{bx}_{by}", px0=bx, py0=by, size=S, z=args.z, points=points,
                         cls=[p.get("cls", "") for p in props], src=[p.get("src", "") for p in props]))
    withp = [r for r in recs if r["points"]]
    empty = [r for r in recs if not r["points"]]
    random.shuffle(withp); random.shuffle(empty)
    n_with = min(len(withp), args.max_blocks)
    n_empty = min(len(empty), int(n_with * args.empty_frac / max(1 - args.empty_frac, 1e-9)))
    keep = withp[:n_with] + empty[:n_empty]
    # spatial split: hash of coarse cell (4x4 blocks) so val is spatially separate
    for r in keep:
        cell = (r["px0"] // (S * 4), r["py0"] // (S * 4))
        h = (cell[0] * 73856093 ^ cell[1] * 19349663) % 1000
        r["split"] = "val" if h < args.val_frac * 1000 else "train"
    print(f"keeping {n_with} blocks with poles + {n_empty} empty; train {sum(r['split']=='train' for r in keep)} val {sum(r['split']=='val' for r in keep)}", flush=True)

    os.makedirs(os.path.join(args.out, "blocks"), exist_ok=True)
    pool = tiles.make_pool(args.workers)
    done = 0
    for r in keep:
        r["years"] = []
        r["cov"] = {}
        for yr in years:
            p = os.path.join(args.out, "blocks", f"{r['id']}_{yr}.jpg")
            if os.path.exists(p):
                r["years"].append(yr); continue
            img, cov = tiles.fetch_block(yr, args.z, r["px0"], r["py0"], S, S, pool)
            if cov.mean() < 0.99:
                r["cov"][str(yr)] = float(cov.mean())
                continue
            Image.fromarray(img).save(p, quality=args.quality)
            r["years"].append(yr)
        done += 1
        if done % 20 == 0:
            print(f"{done}/{len(keep)} blocks", flush=True)
            json.dump(keep, open(os.path.join(args.out, "index.json"), "w"))
    json.dump(keep, open(os.path.join(args.out, "index.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
