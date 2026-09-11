"""Dump peaks (score>=0.05) for every block of a split to JSON: {block_id: [[x,y,score],...]}"""
import argparse, json, numpy as np, torch
from torch.utils.data import DataLoader
from model import PoleNet, decode_peaks
from train import BlockDataset
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--years", required=True)
ap.add_argument("--split", default="val"); ap.add_argument("--out", required=True)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu")
model = PoleNet(ck["args"]["backbone"], pretrained=False); model.load_state_dict(ck["model"]); model.cuda().eval()
years = [int(y) for y in a.years.split(",")]
ds = BlockDataset(a.data, a.split, years, False, max_years=len(years))
dl = DataLoader(ds, 8, num_workers=8)
out = {}
with torch.no_grad():
    for x, mask, hm, idx, *_ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda(), mask.cuda()).float()
        for b in range(x.shape[0]):
            it = ds.items[int(idx[b])]
            pk = decode_peaks(logits[b:b + 1], thresh=0.05)[0]
            out[it["id"]] = [[round(px * 4 + 2, 1), round(py * 4 + 2, 1), round(s, 3)] for px, py, s in pk]
json.dump(out, open(a.out, "w"))
print(len(out), "blocks")
