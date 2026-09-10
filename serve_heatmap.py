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

torch/torchvision must be a matched pair; uv keeps syncing one script environment in place and treats
a PyPI torchvision 0.29.0 as equal to the CPU-index 0.29.0+cpu, so an older environment can hold a
mismatched pair ("operator torchvision::nms does not exist"). The script detects that at startup and
re-runs itself once with `uv run --reinstall-package torch --reinstall-package torchvision`.

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
laptop CPU, ~0.1 s on a GPU). Tiles below --min-zoom show only blocks already computed (never-computed
areas are shaded gray) so a zoomed-out view cannot trigger runaway work.

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


def _check_torchvision():
    """timm imports torchvision; a torchvision whose compiled ops don't match this torch fails with
    'operator torchvision::nms does not exist'. That happens when uv keeps syncing an older script
    environment in place. Detect it and rebuild the two packages once, automatically."""
    try:
        import torchvision  # noqa: F401
        return
    except Exception as e:  # RuntimeError from torchvision, or ImportError
        err = e
    in_uv_env = "environments-v2" in sys.prefix or os.environ.get("UV")
    if in_uv_env and not os.environ.get("STREETLIGHT_REINSTALLED"):
        print(f"torchvision does not match torch in {sys.prefix} ({err}); reinstalling torch+torchvision once via uv...", flush=True)
        os.environ["STREETLIGHT_REINSTALLED"] = "1"
        os.execvp("uv", ["uv", "run", "--reinstall-package", "torch", "--reinstall-package", "torchvision", os.path.abspath(__file__), *sys.argv[1:]])
    sys.exit(f"torch/torchvision mismatch: {err}\nFix: uv run --reinstall-package torch --reinstall-package torchvision serve_heatmap.py\n(or delete {sys.prefix} and run again)")


_check_torchvision()
sys.path.insert(0, os.path.join(ROOT, "src"))
import tiles  # noqa: E402
from model import PoleNet, decode_peaks  # noqa: E402

_log_lock = threading.Lock()
_t0 = time.time()


def log(msg):
    with _log_lock:
        print(f"[{time.time() - _t0:8.1f}s] {msg}", flush=True)


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
        log(f"loaded {os.path.basename(ckpt)} ({ck['args']['backbone']}) on {device}")
        self.device, self.years, self.register = device, years, register
        self.thresh_default = float(ck.get("thresh") or 0.25)
        self.lock = threading.Lock()          # one forward pass at a time
        self.inflight = {}                    # block key -> Event, dedupes concurrent requests
        self.mem = OrderedDict(); self.mem_blocks = mem_blocks
        self.pool = ThreadPoolExecutor(fetch_workers)
        tag = hashlib.sha1((os.path.basename(ckpt) + ",".join(map(str, years)) + str(register)).encode()).hexdigest()[:10]
        self.cache_dir = os.path.join(cache_dir or os.path.expanduser("~/.cache/streetlight-heatmap"), tag)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.stats = dict(blocks_computed=0, blocks_cached=0, seconds_model=0.0, seconds_fetch=0.0, tiles_served=0)
        self.tiles_inflight = 0; self.blocks_inflight = 0; self.model_queue = 0
        self.recent = []                      # (t, seconds_model) of recent blocks, for the ETA
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def _heartbeat(self):
        while True:
            time.sleep(5)
            if self.tiles_inflight or self.blocks_inflight:
                per = self.block_seconds()
                log(f"working: {self.tiles_inflight} tile requests in flight, {self.blocks_inflight} blocks computing "
                    f"({self.model_queue} waiting for the model), ~{per:.1f} s/block -> ~{self.model_queue * per:.0f} s of model work queued")

    def block_seconds(self):
        r = [d for t, d in self.recent[-20:]]
        return sum(r) / len(r) if r else 3.0

    # --- one block: returns (heat[128x128] float16 for the core, peaks [(X,Y,score)] absolute z20 px)
    def block(self, bx, by):
        """-> ((heat, peaks), source) with source in {"mem", "disk", "new"}"""
        key = (bx, by)
        with self.lock:
            hit = self.mem.get(key)
            if hit is not None:
                self.mem.move_to_end(key); return hit, "mem"
            ev = self.inflight.get(key)
            if ev is None:
                ev = self.inflight[key] = threading.Event(); owner = True
            else:
                owner = False
        if not owner:
            ev.wait(); return self.block(bx, by)[0], "shared"
        try:
            res = self._load_disk(key); source = "disk"
            if res is None:
                self.blocks_inflight += 1
                try:
                    res = self._compute(bx, by)
                finally:
                    self.blocks_inflight -= 1
                self._save_disk(key, res); self.stats["blocks_computed"] += 1; source = "new"
            else:
                self.stats["blocks_cached"] += 1
                log(f"block {bx}_{by}: from disk cache ({len(res[1])} peaks)")
            with self.lock:
                self.mem[key] = res
                while len(self.mem) > self.mem_blocks:
                    self.mem.popitem(last=False)
            return res, source
        finally:
            with self.lock:
                self.inflight.pop(key, None)
            ev.set()

    def lookup(self, bx, by):
        """Cache-only: memory, then disk. Never computes. -> (heat, peaks) or None"""
        key = (bx, by)
        with self.lock:
            hit = self.mem.get(key)
        if hit is not None:
            return hit
        res = self._load_disk(key)
        if res is not None:
            with self.lock:
                self.mem[key] = res
        return res

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
        t_fetch = time.time()
        imgs, used = [], []
        for yr in self.years:
            img, cov = tiles.fetch_block(yr, Z, x0, y0, S, S, self.pool)
            if cov.mean() > 0.5:
                imgs.append(img); used.append(yr)
        t_fetch = time.time() - t_fetch; self.stats["seconds_fetch"] += t_fetch
        if not imgs:
            log(f"block {bx}_{by}: no imagery for any year (outside King County coverage?)")
            return np.zeros((BLOCK // STRIDE, BLOCK // STRIDE), np.float16), []
        t_reg = time.time(); shifts = []
        if self.register and len(imgs) > 1:
            ref = imgs[0]; out = [ref]
            for im in imgs[1:]:
                dx, dy = estimate_shift(ref, im); shifts.append((dx, dy)); out.append(translate(im, dx, dy))
            imgs = out
        t_reg = time.time() - t_reg
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].to(self.device)
        self.model_queue += 1
        t_wait = time.time()
        with self.lock, torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            self.model_queue -= 1
            t_wait = time.time() - t_wait; t = time.time()
            logits = self.model(x).float()
            t_model = time.time() - t
        self.stats["seconds_model"] += t_model
        self.recent.append((time.time(), t_model)); self.recent = self.recent[-50:]
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()                    # (S/4, S/4)
        m = MARGIN // STRIDE
        core = prob[m:m + BLOCK // STRIDE, m:m + BLOCK // STRIDE].astype(np.float16)
        peaks = []
        for hx, hy, s in decode_peaks(logits, thresh=0.05)[0]:
            X, Y = x0 + hx * STRIDE + STRIDE / 2, y0 + hy * STRIDE + STRIDE / 2
            if bx <= X < bx + BLOCK and by <= Y < by + BLOCK:
                peaks.append((X, Y, s))
        log(f"block {bx}_{by}: years {used} fetched in {t_fetch:.1f}s, registered in {t_reg:.1f}s"
            f"{' shifts ' + str(shifts) if any(shifts) else ''}, waited {t_wait:.1f}s for the model, model {t_model:.1f}s"
            f" -> {len(peaks)} peaks (max score {max([p[2] for p in peaks], default=0):.2f})")
        return core, peaks

    # --- assemble a tile
    def tile(self, z, x, y, saturate, thresh, draw_peaks=True, compute=True):
        f = 2.0 ** (Z - z)                     # z20 px per requested-tile px (fractional above z20)
        S = int(round(256 * f))                # tile size in z20 px: 8192 @z15 ... 512 @z19, 128 @z21, 32 @z23
        px0, py0 = x * S, y * S
        H = max(1, S // STRIDE)
        t0 = time.time()
        self.tiles_inflight += 1
        try:
            return self._tile(z, x, y, S, px0, py0, H, saturate, thresh, draw_peaks, t0, compute)
        finally:
            self.tiles_inflight -= 1

    def _tile(self, z, x, y, S, px0, py0, H, saturate, thresh, draw_peaks, t0, compute):
        heat = np.zeros((H, H), np.float32); peaks = []; sources = []
        missing = np.zeros((H, H), bool)      # block areas with no cached result (cache-only mode)
        for by in range((py0 // BLOCK) * BLOCK, py0 + S, BLOCK):
            for bx in range((px0 // BLOCK) * BLOCK, px0 + S, BLOCK):
                if compute:
                    (core, pk), src = self.block(bx, by)
                else:
                    hit = self.lookup(bx, by)
                    (core, pk), src = (hit, "mem") if hit is not None else ((np.zeros((BLOCK // STRIDE, BLOCK // STRIDE), np.float16), []), "missing")
                sources.append(src)
                hx0, hy0 = (bx - px0) // STRIDE, (by - py0) // STRIDE
                sx0, sy0 = max(0, -hx0), max(0, -hy0)
                dx0, dy0 = max(0, hx0), max(0, hy0)
                w = min(core.shape[1] - sx0, H - dx0); h = min(core.shape[0] - sy0, H - dy0)
                if w > 0 and h > 0:
                    heat[dy0:dy0 + h, dx0:dx0 + w] = core[sy0:sy0 + h, sx0:sx0 + w]
                    if src == "missing":
                        missing[dy0:dy0 + h, dx0:dx0 + w] = True
                peaks += [(X, Y, s) for X, Y, s in pk if px0 <= X < px0 + S and py0 <= Y < py0 + S]
        self.stats["tiles_served"] += 1
        c = {k: sources.count(k) for k in ("new", "shared", "disk", "mem", "missing")}
        if compute:
            log(f"tile {z}/{x}/{y}: {len(sources)} blocks ({c['new']} computed, {c['shared']} computed by another request, {c['disk']} from disk, {c['mem']} in memory)"
                f" in {time.time() - t0:.1f}s -> {sum(1 for p in peaks if p[2] >= thresh)} rings | still in flight: {self.tiles_inflight - 1} tiles")
        else:
            log(f"tile {z}/{x}/{y}: below min zoom, cache only: {len(sources) - c['missing']} of {len(sources)} blocks cached (gray = not computed yet)")
        return render(heat, peaks, px0, py0, 2.0 ** (Z - z), saturate, thresh, draw_peaks, missing if not compute else None)


def render(heat, peaks, px0, py0, f, saturate, thresh, draw_peaks, missing=None):
    t = np.clip(heat / saturate, 0, 1)
    if t.shape[0] != 256:
        t = np.asarray(Image.fromarray((t * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)) / 255.0
    r = np.clip(3 * t, 0, 1); g = np.clip(3 * t - 1, 0, 1); b = np.clip(3 * t - 2, 0, 1)
    a = np.where(t < 0.04, 0, np.clip(0.25 + t, 0, 0.9))
    if missing is not None and missing.any():   # cache-only tile: shade never-computed areas gray
        m = np.asarray(Image.fromarray(missing.astype(np.uint8) * 255).resize((256, 256), Image.NEAREST)) > 0
        r[m], g[m], b[m], a[m] = 0.5, 0.5, 0.5, 0.35
    rgba = (np.stack([r, g, b, a], -1) * 255).astype(np.uint8)
    im = Image.fromarray(rgba, "RGBA")
    if draw_peaks and f <= 8:  # rings only at z17+
        d = ImageDraw.Draw(im); rad = max(3, min(28, int(20 / f)))
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
    log(f"model {os.path.basename(args.ckpt)} on {device}; years {years}; peak thresh {thresh}")
    log(f"imagery tile cache {tiles.CACHE}; block result cache {det.cache_dir}")
    log("cost guide: one 512 px block (~51 m) = fetching the years' tiles once (fast, cached on disk) + one model pass"
        " (~1-3 s on a laptop CPU, ~0.1 s on a GPU). A z18 tile is 4 blocks, z19 is 1; iD requests ~20-40 tiles per screen,"
        " so the first look at a new area can take a minute or more on CPU. Every block is cached afterwards.")
    log("watch this log: each tile and block prints timings; a 'working:' heartbeat appears every 5 s while requests are pending")

    warned = set()

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
                    if z > 23 or z < 12:
                        im = EMPTY
                    else:
                        below = z < args.min_zoom
                        if below and z not in warned:
                            warned.add(z)
                            log(f"zoom {z} is below --min-zoom {args.min_zoom}: serving CACHED blocks only (gray = not computed). "
                                f"Zoom in to z{args.min_zoom}+ to compute, or restart with --min-zoom {z} (a z{z} tile is {4 ** (Z - 1 - z)} blocks).")
                        im = det.tile(z, x, y, float(q.get("saturate", [args.saturate])[0]), float(q.get("thresh", [thresh])[0]), q.get("peaks", ["1"])[0] != "0", compute=not below)
                    buf = io.BytesIO(); im.save(buf, "PNG")
                    return self._send(200, buf.getvalue(), "image/png")
                self._send(404, b"not found", "text/plain")
            except Exception as e:  # keep the server alive; iD just gets a blank tile
                import traceback
                log(f"error {self.path}: {e!r}"); traceback.print_exc()
                self._send(500, repr(e).encode(), "text/plain")

    srv = ThreadingHTTPServer((args.host, args.port), H)
    log(f"serving on http://{args.host}:{args.port}/  ->  iD custom background: http://localhost:{args.port}/{{zoom}}/{{x}}/{{y}}.png")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
