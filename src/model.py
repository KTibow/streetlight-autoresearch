"""Pole detector: heatmap regression over a stack of ortho years.

Input:  (B, Y, 3, H, W) uint8/float chips, Y years at identical footprint.
Output: (B, 1, H/4, W/4) logits of a pole-center heatmap.

Design: each year is encoded independently by a shared pretrained encoder
(so the net learns year-invariant "pole under these shadows" features),
per-scale features are fused across years by max+mean pooling (permutation
invariant, tolerant to missing years), then a light FPN decoder predicts a
1/4-resolution heatmap. Single-year training is the Y=1 special case.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import timm


class YearFusion(nn.Module):
    """Permutation-invariant fusion across the year axis."""
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))

    def forward(self, x, mask):  # x: (B,Y,C,h,w), mask: (B,Y) bool valid
        m = mask[:, :, None, None, None].to(x.dtype)
        xm = (x * m).sum(1) / m.sum(1).clamp(min=1)
        xx = torch.where(m.bool(), x, torch.full_like(x, -1e4)).amax(1)
        return self.proj(torch.cat([xm, xx], 1))


class PoleNet(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, fpn_ch=96):
        super().__init__()
        probe = timm.create_model(backbone, pretrained=False, features_only=True)
        idx = tuple(i for i, r in enumerate(probe.feature_info.reduction()) if r in (4, 8, 16, 32))
        del probe
        self.enc = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=idx)
        chs = self.enc.feature_info.channels()  # strides 4,8,16,32
        self.fuse = nn.ModuleList([YearFusion(c) for c in chs])
        self.lat = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in chs])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU(inplace=True)) for _ in chs])
        self.head = nn.Sequential(nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(fpn_ch, 1, 1))
        self.head[-1].bias.data.fill_(-4.0)  # focal-style prior
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, mask=None):
        B, Y, C, H, W = x.shape
        if mask is None:
            mask = torch.ones(B, Y, dtype=torch.bool, device=x.device)
        x = x.reshape(B * Y, C, H, W).float()
        if x.max() > 1.5:
            x = x / 255.0
        x = (x - self.mean) / self.std
        feats = self.enc(x)
        fused = []
        for f, fu in zip(feats, self.fuse):
            _, c, h, w = f.shape
            fused.append(fu(f.view(B, Y, c, h, w), mask))
        p = None
        for i in range(len(fused) - 1, -1, -1):
            l = self.lat[i](fused[i])
            p = l if p is None else l + F.interpolate(p, size=l.shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p)
        return self.head(p)  # stride 4


def focal_heatmap_loss(logits, target, alpha=2.0, beta=4.0, ignore=None, pos_weight=None):
    """CenterNet penalty-reduced focal loss. target in [0,1], peaks == 1.
    ignore: optional (B,1,h,w) bool - no NEGATIVE penalty there (labels known incomplete: parking
    lots, sports fields); positives inside still count. pos_weight: optional (B,1,h,w) multiplier on
    the positive term (e.g. >1 for rare classes)."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    if ignore is not None:
        neg = neg * (~ignore).float()
    pw = pos if pos_weight is None else pos * pos_weight
    pos_loss = -(torch.log(p) * (1 - p) ** alpha * pw).sum()
    neg_loss = -(torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg).sum()
    npos = pos.sum().clamp(min=1)
    return (pos_loss + neg_loss) / npos


def decode_peaks(logits, thresh=0.3, nms_k=5):
    """logits (B,1,h,w) -> list of (x, y, score) at heatmap resolution."""
    p = torch.sigmoid(logits).detach()
    mx = F.max_pool2d(p, nms_k, stride=1, padding=nms_k // 2)
    keep = (p == mx) & (p > thresh)
    out = []
    for b in range(p.shape[0]):
        ys, xs = torch.nonzero(keep[b, 0], as_tuple=True)
        out.append([(float(x), float(y), float(p[b, 0, y, x])) for y, x in zip(ys, xs)])
    return out
