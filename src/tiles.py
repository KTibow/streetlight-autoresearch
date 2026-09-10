"""King County ortho tile access (Web Mercator XYZ cache, zoom 20 ~0.10 m ground GSD).

URL: https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_{year}/MapServer/tile/{z}/{y}/{x}
Years with z20: 2013 2015 2017 2019 2021 2023 2025; z19 only: 2007 2009 2012.
"""
import math, os, io, time, threading
import numpy as np, requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

BASE = "https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_{year}/MapServer/tile/{z}/{y}/{x}"
Z20_YEARS = [2013, 2015, 2017, 2019, 2021, 2023, 2025]
R = 6378137.0
CACHE = os.environ.get("TILE_CACHE", os.path.expanduser("~/tile_cache"))
_local = threading.local()


def _sess():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "streetlight-autoresearch/0.1"
        _local.s = s
    return _local.s


def lonlat_to_merc(lon, lat):
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def merc_to_lonlat(x, y):
    lon = math.degrees(x / R)
    lat = math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)
    return lon, lat


def merc_to_pixel(x, y, z):
    """Global pixel coords at zoom z (origin top-left of world)."""
    n = 256 * 2 ** z
    px = (x + math.pi * R) / (2 * math.pi * R) * n
    py = (math.pi * R - y) / (2 * math.pi * R) * n
    return px, py


def pixel_to_merc(px, py, z):
    n = 256 * 2 ** z
    x = px / n * 2 * math.pi * R - math.pi * R
    y = math.pi * R - py / n * 2 * math.pi * R
    return x, y


def merc_gsd(lat, z=20):
    """metres per pixel in Mercator units and on the ground at lat."""
    nominal = 2 * math.pi * R / (256 * 2 ** z)
    return nominal, nominal * math.cos(math.radians(lat))


def fetch_tile(year, z, x, y, retries=4):
    p = os.path.join(CACHE, str(year), str(z), str(x), f"{y}.jpg")
    if os.path.exists(p):
        try:
            return np.asarray(Image.open(p).convert("RGB"))
        except Exception:
            os.remove(p)
    url = BASE.format(year=year, z=z, x=x, y=y)
    for i in range(retries):
        try:
            r = _sess().get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p + ".tmp", "wb") as f:
                    f.write(r.content)
                os.replace(p + ".tmp", p)
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception as e:
            err = e
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"tile fetch failed {url}")


def fetch_block(year, z, px0, py0, w, h, pool=None):
    """Mosaic of global-pixel rect [px0, px0+w) x [py0, py0+h) at zoom z. Returns (h,w,3) uint8 and mask of coverage."""
    tx0, ty0 = px0 // 256, py0 // 256
    tx1, ty1 = (px0 + w - 1) // 256, (py0 + h - 1) // 256
    out = np.zeros((h, w, 3), np.uint8)
    cov = np.zeros((h, w), bool)
    jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    def do(t):
        return t, fetch_tile(year, z, t[0], t[1])
    it = pool.map(do, jobs) if pool else map(do, jobs)
    for (tx, ty), img in it:
        if img is None:
            continue
        x0, y0 = tx * 256 - px0, ty * 256 - py0
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        ww, hh = min(256 - sx0, w - dx0), min(256 - sy0, h - dy0)
        if ww <= 0 or hh <= 0:
            continue
        out[dy0:dy0 + hh, dx0:dx0 + ww] = img[sy0:sy0 + hh, sx0:sx0 + ww]
        cov[dy0:dy0 + hh, dx0:dx0 + ww] = True
    return out, cov


def make_pool(n=16):
    return ThreadPoolExecutor(n)
