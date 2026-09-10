# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.14.*",
#   "torchvision==0.29.*",
#   "timm>=1.0",
#   "pillow>=10",
#   "numpy>=1.26",
#   "requests>=2.31",
# ]
# [tool.uv.sources]
# # torch and torchvision must be a matched pair from the SAME index (timm imports torchvision, and a
# # mismatch fails at import with "operator torchvision::nms does not exist"). Exact versions are
# # pinned so every platform resolves the same pair: CPU wheels from the PyTorch index on
# # Linux/Windows (x86_64 and aarch64), PyPI on macOS. For CUDA change the url to e.g. .../whl/cu126.
# torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]
# torchvision = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Local streetlight/pole heat-map tile server for iD (or JOSM / QGIS / anything that speaks XYZ).

    uv run serve_heatmap.py                      # uses weights/polenet_cnxt_multiyear_v2.pt
    uv run serve_heatmap.py --years 2025,2023,2021,2019,2017,2015,2013 --device cuda

If startup fails with "operator torchvision::nms does not exist", the environment holds a torchvision
that does not match its torch (typically a stale uv script environment from an earlier header).
Rebuild it once:  rm -rf ~/.cache/uv/environments-v2/serve-heatmap-*   then run again.

Then in iD: Background settings -> "Custom" -> paste

    http://localhost:8765/{zoom}/{x}/{y}.png

(as a custom background it replaces the imagery; to keep the imagery underneath, add it as an overlay
via the "Map Data" panel -> custom data isn't a tile layer, so instead use a second browser tab, or
in JOSM add it as an imagery layer and set the opacity.)

How it works: each tile request is mapped to 512-px blocks on the King County zoom-20 pixel grid;
for each block the server fetches the chosen years (with a 64 px margin), registers them to the
newest year, runs the detector once, and caches the heat-map + peaks (in memory and under
~/.cache/streetlight-heatmap). Tiles are then cut from the cached blocks and resampled, so panning
around the same area is cheap and only new ground costs a model call (a few seconds per block on a
laptop CPU, ~0.1 s on a GPU). Tiles below --min-zoom are served empty to avoid runaway work.

Colour scale: the net rarely outputs >0.6, so the heat is saturated at --saturate (default 0.5):
p=0.05 faint yellow ... p>=0.5 solid red. Rings mark peak-picked detections above --thresh.
"""
import argparse, hashlib, io, json, math, os, sys, threading, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import tiles  # noqa: E402
from model import PoleNet, decode_peaks  # noqa: E402

Z = 20            # native grid of the KC cache
BLOCK = 512       # cached inference unit, z20 px (~51 m)
MARGIN = 64       # context around the block fed to the net
STRIDE = 4        # heat-map stride


# ----------------------------------------------------------------------------- registration
def _edges(gray):
    gy, gx = np.gradient(gray.astype(np.float32))
    g = np.hypot(gx, gy)
    g = np.minimum(g, np.percentile(g, 99))
    return g - g.mean()


def estimate_shift(ref, img, max_px=16):
    """Integer (dx, dy) that moves `img` onto `ref` (both HxW uint8 RGB); (0,0) if unsure."""
    a = _edges(ref.mean(2)[64:-64, 64:-64]); b = _edges(img.mean(2)[64:-64, 64:-64])
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    c = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    c = np.fft.fftshift(c)
    cy, cx = np.unravel_index(int(c.argmax()), c.shape)
    dy, dx = cy - a.shape[0] // 2, cx - a.shape[1] // 2
    m = c.copy(); m[max(0, cy - 5):cy + 6, max(0, cx - 5):cx + 6] = -np.inf
    if max(abs(dx), abs(dy)) > max_px or c.max() / max(m.max(), 1e-6) < 1.05:
        return 0, 0
    return int(dx), int(dy)


def translate(img, dx, dy):
    if dx == 0 and dy == 0:
        return img
    H, W = img.shape[:2]; pad = max(abs(dx), abs(dy))
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return np.ascontiguousarray(p[pad - dy:pad - dy + H, pad - dx:pad - dx + W])


# ----------------------------------------------------------------------------- inference cache
class Detector:
    def __init__(self, ckpt, years, device, register=True, cache_dir=None, mem_blocks=600, fetch_workers=16):
        ck = torch.load(ckpt, map_location="cpu")
        self.model = PoleNet(ck["args"]["backbone"], pretrained=False)
        self.model.load_state_dict(ck["model"]); self.model.to(device).eval()
        self.device, self.years, self.register = device, years, register
        self.thresh_default = float(ck.get("thresh") or 0.25)
        self.lock = threading.Lock()          # one forward pass at a time
        self.inflight = {}                    # block key -> Event, dedupes concurrent requests
        self.mem = OrderedDict(); self.mem_blocks = mem_blocks
        self.pool = ThreadPoolExecutor(fetch_workers)
        tag = hashlib.sha1((os.path.basename(ckpt) + ",".join(map(str, years)) + str(register)).encode()).hexdigest()[:10]
        self.cache_dir = os.path.join(cache_dir or os.path.expanduser("~/.cache/streetlight-heatmap"), tag)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.stats = dict(blocks_computed=0, blocks_cached=0, seconds_model=0.0)

    # --- one block: returns (heat[128x128] float16 for the core, peaks [(X,Y,score)] absolute z20 px)
    def block(self, bx, by):
        key = (bx, by)
        with self.lock:
            hit = self.mem.get(key)
            if hit is not None:
                self.mem.move_to_end(key); return hit
            ev = self.inflight.get(key)
            if ev is None:
                ev = self.inflight[key] = threading.Event(); owner = True
            else:
                owner = False
        if not owner:
            ev.wait(); return self.block(bx, by)
        try:
            res = self._load_disk(key)
            if res is None:
                res = self._compute(bx, by); self._save_disk(key, res); self.stats["blocks_computed"] += 1
            else:
                self.stats["blocks_cached"] += 1
            with self.lock:
                self.mem[key] = res
                while len(self.mem) > self.mem_blocks:
                    self.mem.popitem(last=False)
            return res
        finally:
            with self.lock:
                self.inflight.pop(key, None)
            ev.set()

    def _disk_path(self, key):
        return os.path.join(self.cache_dir, f"{key[0]}_{key[1]}.npz")

    def _load_disk(self, key):
        p = self._disk_path(key)
        if not os.path.exists(p):
            return None
        try:
            d = np.load(p)
            return d["heat"], [tuple(map(float, r)) for r in d["peaks"]]
        except Exception:
            return None

    def _save_disk(self, key, res):
        heat, peaks = res
        np.savez_compressed(self._disk_path(key), heat=heat, peaks=np.array(peaks, np.float32).reshape(-1, 3))

    def _compute(self, bx, by):
        x0, y0, S = bx - MARGIN, by - MARGIN, BLOCK + 2 * MARGIN
        imgs = []
        for yr in self.years:
            img, cov = tiles.fetch_block(yr, Z, x0, y0, S, S, self.pool)
            if cov.mean() > 0.5:
                imgs.append(img)
        if not imgs:
            return np.zeros((BLOCK // STRIDE, BLOCK // STRIDE), np.float16), []
        if self.register and len(imgs) > 1:
            ref = imgs[0]
            imgs = [ref] + [translate(im, *estimate_shift(ref, im)) for im in imgs[1:]]
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].to(self.device)
        t = time.time()
        with self.lock, torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            logits = self.model(x).float()
        self.stats["seconds_model"] += time.time() - t
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()                    # (S/4, S/4)
        m = MARGIN // STRIDE
        core = prob[m:m + BLOCK // STRIDE, m:m + BLOCK // STRIDE].astype(np.float16)
        peaks = []
        for hx, hy, s in decode_peaks(logits, thresh=0.05)[0]:
            X, Y = x0 + hx * STRIDE + STRIDE / 2, y0 + hy * STRIDE + STRIDE / 2
            if bx <= X < bx + BLOCK and by <= Y < by + BLOCK:
                peaks.append((X, Y, s))
        return core, peaks

    # --- assemble a tile
    def tile(self, z, x, y, saturate, thresh, draw_peaks=True):
        f = 2 ** (Z - z)                       # z20 px per requested-tile px
        S = 256 * f                            # tile size in z20 px
        px0, py0 = x * S, y * S
        H = S // STRIDE
        heat = np.zeros((H, H), np.float32); peaks = []
        for by in range((py0 // BLOCK) * BLOCK, py0 + S, BLOCK):
            for bx in range((px0 // BLOCK) * BLOCK, px0 + S, BLOCK):
                core, pk = self.block(bx, by)
                hx0, hy0 = (bx - px0) // STRIDE, (by - py0) // STRIDE
                sx0, sy0 = max(0, -hx0), max(0, -hy0)
                dx0, dy0 = max(0, hx0), max(0, hy0)
                w = min(core.shape[1] - sx0, H - dx0); h = min(core.shape[0] - sy0, H - dy0)
                if w > 0 and h > 0:
                    heat[dy0:dy0 + h, dx0:dx0 + w] = core[sy0:sy0 + h, sx0:sx0 + w]
                peaks += [(X, Y, s) for X, Y, s in pk if px0 <= X < px0 + S and py0 <= Y < py0 + S]
        return render(heat, peaks, px0, py0, f, saturate, thresh, draw_peaks)


def render(heat, peaks, px0, py0, f, saturate, thresh, draw_peaks):
    t = np.clip(heat / saturate, 0, 1)
    if t.shape[0] != 256:
        t = np.asarray(Image.fromarray((t * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)) / 255.0
    r = np.clip(3 * t, 0, 1); g = np.clip(3 * t - 1, 0, 1); b = np.clip(3 * t - 2, 0, 1)
    a = np.where(t < 0.04, 0, np.clip(0.25 + t, 0, 0.9))
    rgba = (np.stack([r, g, b, a], -1) * 255).astype(np.uint8)
    im = Image.fromarray(rgba, "RGBA")
    if draw_peaks and f <= 8:  # rings only at z17+
        d = ImageDraw.Draw(im); rad = max(3, int(10 / f * 2))
        for X, Y, s in peaks:
            if s < thresh:
                continue
            cx, cy = (X - px0) / f, (Y - py0) / f
            d.ellipse([cx - rad - 1, cy - rad - 1, cx + rad + 1, cy + rad + 1], outline=(0, 0, 0, 220), width=3)
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(255, 255, 255, 255), width=2)
    return im


EMPTY = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
INDEX = """<!doctype html><title>streetlight heatmap</title><body style="font:14px system-ui;max-width:48em;margin:2em auto">
<h2>streetlight / pole heat-map tiles</h2>
<p>Tile URL for iD (Background settings &rarr; Custom) or JOSM (Imagery &rarr; Imagery preferences &rarr; TMS):</p>
<pre>http://{host}/{{zoom}}/{{x}}/{{y}}.png</pre>
<p>Years: {years} &middot; device: {device} &middot; min zoom {minz} &middot; saturate {sat} &middot; peak rings &ge; {thr}</p>
<p>Query overrides per tile: <code>?saturate=0.4&amp;thresh=0.3&amp;peaks=0</code>.
GeoJSON of cached detections: <code>/peaks.geojson?bbox=lon0,lat0,lon1,lat1</code> (only blocks already computed).</p>
<p><a href="/status">/status</a></p></body>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "polenet_cnxt_multiyear_v2.pt"))
    ap.add_argument("--years", default="2025,2023,2021,2019", help="KC aerial years to stack (z20 exists for 2013..2025 odd years); more = better recall, slower")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--min-zoom", type=int, default=18, help="serve empty tiles below this zoom (z18 tile = 4 blocks, z17 = 16, z16 = 64)")
    ap.add_argument("--saturate", type=float, default=0.5, help="probability shown as full red")
    ap.add_argument("--thresh", type=float, default=None, help="ring detections at/above this score (default: checkpoint's best threshold)")
    ap.add_argument("--no-register", action="store_true", help="skip per-block year registration")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    if args.threads:
        torch.set_num_threads(args.threads)
    years = [int(y) for y in args.years.split(",")]
    det = Detector(args.ckpt, years, device, register=not args.no_register, cache_dir=args.cache_dir)
    thresh = args.thresh if args.thresh is not None else det.thresh_default
    print(f"model {os.path.basename(args.ckpt)} on {device}; years {years}; peak thresh {thresh}; tile cache {tiles.CACHE}; block cache {det.cache_dir}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Cache-Control", "max-age=300")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            parts = [p for p in u.path.split("/") if p]
            try:
                if not parts:
                    return self._send(200, INDEX.format(host=self.headers.get("Host", f"localhost:{args.port}"), years=",".join(map(str, years)), device=device, minz=args.min_zoom, sat=args.saturate, thr=thresh).encode(), "text/html")
                if parts[0] == "status":
                    return self._send(200, json.dumps(dict(det.stats, mem_blocks=len(det.mem), years=years, device=device)).encode(), "application/json")
                if parts[0] == "peaks.geojson":
                    lon0, lat0, lon1, lat1 = [float(v) for v in q["bbox"][0].split(",")]
                    feats = []
                    with det.lock:
                        blocks = list(det.mem.values())
                    for _, pk in blocks:
                        for X, Y, s in pk:
                            lon, lat = tiles.merc_to_lonlat(*tiles.pixel_to_merc(X, Y, Z))
                            if lon0 <= lon <= lon1 and lat0 <= lat <= lat1 and s >= thresh:
                                feats.append(dict(type="Feature", properties=dict(score=round(s, 3)), geometry=dict(type="Point", coordinates=[lon, lat])))
                    return self._send(200, json.dumps(dict(type="FeatureCollection", features=feats)).encode(), "application/geo+json")
                if len(parts) == 3:
                    z, x, y = int(parts[0]), int(parts[1]), int(parts[2].split(".")[0])
                    if z < args.min_zoom or z > 22:
                        im = EMPTY
                    else:
                        im = det.tile(z, x, y, float(q.get("saturate", [args.saturate])[0]), float(q.get("thresh", [thresh])[0]), q.get("peaks", ["1"])[0] != "0")
                    buf = io.BytesIO(); im.save(buf, "PNG")
                    return self._send(200, buf.getvalue(), "image/png")
                self._send(404, b"not found", "text/plain")
            except Exception as e:  # keep the server alive; iD just gets a blank tile
                print("error", self.path, repr(e), flush=True)
                self._send(500, repr(e).encode(), "text/plain")

    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"serving on http://{args.host}:{args.port}/  ->  iD custom background: http://localhost:{args.port}/{{zoom}}/{{x}}/{{y}}.png", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
