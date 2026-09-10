"""Run the pole detector over an arbitrary King County area and write GeoJSON.

python infer.py --ckpt runs/exp/best.pt --bbox -122.35,47.65,-122.33,47.66 --years 2025,2023,2021 --out poles.geojson
bbox = lon_min,lat_min,lon_max,lat_max (EPSG:4326). Tiles are fetched (and cached) from King County.
"""
import argparse, json, math, sys
import numpy as np, torch, torch.nn.functional as F
import tiles
from model import PoleNet, decode_peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--years", default="2025,2023,2021,2019")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    model = PoleNet(ck["args"]["backbone"], pretrained=False)
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    thresh = args.thresh if args.thresh is not None else float(ck.get("thresh", 0.4))
    years = [int(y) for y in args.years.split(",")]
    z = 20
    lon0, lat0, lon1, lat1 = [float(v) for v in args.bbox.split(",")]
    px0, py0 = tiles.merc_to_pixel(*tiles.lonlat_to_merc(lon0, lat1), z)
    px1, py1 = tiles.merc_to_pixel(*tiles.lonlat_to_merc(lon1, lat0), z)
    px0, py0, px1, py1 = int(px0), int(py0), int(math.ceil(px1)), int(math.ceil(py1))
    pool = tiles.make_pool(args.workers)
    step = args.block - args.overlap
    dets = []
    stride = 4
    n = 0
    for by in range(py0, py1, step):
        for bx in range(px0, px1, step):
            imgs, ok = [], []
            for yr in years:
                img, cov = tiles.fetch_block(yr, z, bx, by, args.block, args.block, pool)
                if cov.mean() > 0.5:
                    imgs.append(img)
            if not imgs:
                continue
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].to(args.device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                logits = model(x)
            peaks = decode_peaks(logits.float(), thresh=thresh)[0]
            for (hx, hy, s) in peaks:
                cx, cy = hx * stride + stride / 2, hy * stride + stride / 2
                # drop peaks in the overlap margin (except at the outer edge) to avoid duplicates
                if (cx < args.overlap / 2 and bx > px0) or (cy < args.overlap / 2 and by > py0):
                    continue
                if (cx >= args.block - args.overlap / 2 and bx + step < px1) or (cy >= args.block - args.overlap / 2 and by + step < py1):
                    continue
                mx, my = tiles.pixel_to_merc(bx + cx, by + cy, z)
                lon, lat = tiles.merc_to_lonlat(mx, my)
                dets.append((lon, lat, s))
            n += 1
            print(f"block {n}: {len(peaks)} peaks", file=sys.stderr, flush=True)
    # inside bbox only
    dets = [d for d in dets if lon0 <= d[0] <= lon1 and lat0 <= d[1] <= lat1]
    gj = dict(type="FeatureCollection", features=[dict(type="Feature", properties=dict(score=round(s, 3)),
              geometry=dict(type="Point", coordinates=[lon, lat])) for lon, lat, s in dets])
    json.dump(gj, open(args.out, "w"))
    print(f"{len(dets)} detections -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
