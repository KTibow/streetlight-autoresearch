"""Overlay label points on a KC ortho block. python viz.py lon lat year out.png [--labels name ...]"""
import sys, gzip, json, argparse, numpy as np
from PIL import Image, ImageDraw
import tiles

def draw_points(img, px0, py0, pts, color, r=6, z=20):
    d = ImageDraw.Draw(img)
    n = 0
    for lon, lat in pts:
        x, y = tiles.merc_to_pixel(*tiles.lonlat_to_merc(lon, lat), z)
        x -= px0; y -= py0
        if 0 <= x < img.width and 0 <= y < img.height:
            d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2); n += 1
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lon", type=float); ap.add_argument("lat", type=float); ap.add_argument("year", type=int); ap.add_argument("out")
    ap.add_argument("--labels", nargs="*", default=["scl_poles"]); ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--offset_m", default="0,0")
    a = ap.parse_args()
    import math
    dx, dy = [float(v) for v in a.offset_m.split(",")]
    x, y = tiles.lonlat_to_merc(a.lon, a.lat); px, py = tiles.merc_to_pixel(x, y, 20)
    px0, py0 = int(px) - a.size // 2, int(py) - a.size // 2
    img, cov = tiles.fetch_block(a.year, 20, px0, py0, a.size, a.size, tiles.make_pool(16))
    im = Image.fromarray(img)
    colors = ["red", "yellow", "cyan", "lime"]
    for i, name in enumerate(a.labels):
        fc = json.load(gzip.open(f"../labels/{name}.geojson.gz", "rt"))
        pts = []
        for f in fc["features"]:
            lon, lat = f["geometry"]["coordinates"][:2]
            if abs(lon - a.lon) < 0.01 and abs(lat - a.lat) < 0.01:
                k = 1 / math.cos(math.radians(lat))
                mx, my = tiles.lonlat_to_merc(lon, lat)
                pts.append(tiles.merc_to_lonlat(mx + dx * k, my + dy * k))
        n = draw_points(im, px0, py0, pts, colors[i % 4])
        print(name, n, "points drawn")
    im.save(a.out)
