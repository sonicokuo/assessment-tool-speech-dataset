#!/usr/bin/env python3
"""whitebox_control.py — D2: an END-TO-END positive control through the SAME extractor.

WHY THE EXISTING POSITIVE CONTROL IS NOT ENOUGH
It computes phi-hat ANALYTICALLY from corrupted per-frame (q, m). At zero corruption
phi-hat == phi, so recovering the ceiling is an algebraic identity announcing itself. It
validates the REFERENCE and the METRIC, and its corruption sweep genuinely shows the
metric's sensitivity ordering (support errors dominate value errors). But it never touches
the left half of the pipeline: network -> autograd -> dim reduction -> 50 Hz map.

So "the model scores 10.7% therefore it grounds weakly" is NOT licensed. The alternative
-- "the EXTRACTOR loses the signal" -- is unexcluded, and that is exactly the failure class
Adebayo et al. (NeurIPS 2018) and Sixt et al. (ICML 2020) document.

THE CONTROL
Build a WHITE-BOX model that grounds BY CONSTRUCTION and push it through the identical
extractor:

    WavLM frames -> tiny conv head -> per-frame f0 qhat and voicing mhat
                 -> f0_sd computed the way the FORMULA does:
                        mu  = sum(m*q)/sum(m)
                        var = sum(m*(q-mu)^2)/(sum(m)-1)
                        sd  = sqrt(var)

All in one torch graph, so `d(sd)/d(input frames)` is well defined and can be taken by the
SAME `capture_saliency.py` code path. The per-frame targets are already on disk
(`build_oracle_maps.py` stores `q/<stem>` and `keep/<stem>`), so the probe is supervised
directly and trains in minutes.

READING THE RESULT
  score ~ ceiling  -> the extractor is faithful, and the real model's 10.7% is a REAL and
                      weak grounding result. Our negatives become interpretable.
  score ~ 10%      -> the EXTRACTOR is the bottleneck; the model panel says nothing about
                      the model, and no attribution number in this project is readable
                      until the extractor is fixed.
Either outcome is decisive, which is why this outranks everything else open.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

WAVLM_RATE, ORACLE_RATE = 50.0, 100.0


class DefiningStatisticModel(nn.Module):
    """Per-frame f0 + voicing, aggregated by the DEFINING STATISTIC for f0_sd.

    The map is the computation: grounding is architectural, not learned-and-hoped-for.
    """

    def __init__(self, in_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.GELU(),
        )
        self.q_head = nn.Conv1d(hidden, 1, kernel_size=1)      # per-frame f0 (Hz)
        self.m_head = nn.Conv1d(hidden, 1, kernel_size=1)      # per-frame voicing logit

    def forward(self, x):                                      # x: (B, T, 1024)
        h = self.trunk(x.transpose(1, 2))
        q = self.q_head(h).squeeze(1) * 100.0 + 150.0          # rough Hz scaling
        m = torch.sigmoid(self.m_head(h).squeeze(1))
        return q, m

    def aggregate(self, q, m):
        """f0_sd from GIVEN per-frame (q, m) — the defining statistic, nothing learned."""
        w = m.sum(dim=1).clamp(min=2.0)
        mu = (m * q).sum(dim=1) / w
        var = (m * (q - mu.unsqueeze(1)) ** 2).sum(dim=1) / (w - 1.0)
        return torch.sqrt(var.clamp(min=1e-6))

    def f0_sd(self, x, return_intermediates: bool = False):
        """sd through the formula.

        Returns (q, m) on request so callers can take d(sd)/d(q) in VALUE space. The
        previous version recomputed q inside, so a caller's captured q was a DIFFERENT
        tensor and `autograd.grad(sd, q)` returned None — the value-space diagnostic
        silently produced nan, which is the one measurement that separates "the formula
        is fine but the extractor loses it" from "the probe is just weak".
        """
        q, m = self(x)
        sd = self.aggregate(q, m)
        return (sd, q, m) if return_intermediates else sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="npz with q/<stem> and keep/<stem>")
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_n", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=20)   # 6 left f0 error at 32%
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    z = np.load(a.oracle, allow_pickle=True)
    names = [str(x) for x in z["names"] if f"q/{x}" in z and f"keep/{x}" in z]
    print(f"clips with per-frame targets: {len(names)}", flush=True)
    if len(names) < 200:
        print("[fatal] too few per-frame targets — rebuild oracle maps with q/keep stored")
        return 1

    model = DefiningStatisticModel().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train, test = names[: a.train_n], names[a.train_n:]

    def load(stem):
        p = os.path.join(a.test_dir, stem + ".pt")
        if not os.path.exists(p):
            return None
        d = torch.load(p, map_location="cpu", weights_only=False)
        af = d["audio_features"].float()                                  # (T,1024) @50Hz
        q = torch.from_numpy(np.asarray(z[f"q/{stem}"], dtype=np.float32))     # @100Hz
        k = torch.from_numpy(np.asarray(z[f"keep/{stem}"], dtype=np.float32))
        # oracle grid is 100 Hz, WavLM is 50 Hz -> decimate targets by 2
        q, k = q[::2], k[::2]
        n = min(af.shape[0], q.shape[0])
        return af[:n], q[:n], k[:n]

    print("training the white-box probe (per-frame supervision, already on disk)", flush=True)
    for ep in range(a.epochs):
        tot, cnt = 0.0, 0
        for stem in train:
            r = load(stem)
            if r is None:
                continue
            af, q, k = (t.to(a.device) for t in r)
            qh, mh = model(af.unsqueeze(0))
            loss = (((qh[0] - q) ** 2) * k).sum() / k.sum().clamp(min=1) / 1000.0 \
                   + nn.functional.binary_cross_entropy(mh[0].clamp(1e-6, 1 - 1e-6), k)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); cnt += 1
        print(f"  epoch {ep+1}/{a.epochs}  loss {tot/max(cnt,1):.4f}", flush=True)

    # ---- capture saliency through the IDENTICAL path the real model uses ----
    with torch.no_grad():
        eq, em, nn_ = 0.0, 0.0, 0
        for stem in test[:200]:
            r = load(stem)
            if r is None:
                continue
            af, q, k = (t.to(a.device) for t in r)
            qh, mh = model(af.unsqueeze(0))
            kb = k > 0.5
            if kb.sum() < 5:
                continue
            eq += float((qh[0][kb] - q[kb]).abs().mean() / q[kb].abs().mean())
            em += float(((mh[0] > 0.5).float() != k).float().mean())
            nn_ += 1
        print(f"[probe accuracy] f0 rel-error {eq/max(nn_,1):.3f}  voicing error "
              f"{em/max(nn_,1):.3f}   <- a map score is unreadable without these", flush=True)

    print("capturing white-box saliency (same reduction, same grid lift)", flush=True)
    model.eval()
    out = {}
    for stem in test:
        r = load(stem)
        if r is None:
            continue
        af = r[0].unsqueeze(0).to(a.device).requires_grad_(True)
        sd, qh, mh = model.f0_sd(af, return_intermediates=True)
        # TWO maps, in two different spaces. The reference phi_t = keep(q-mu)/((N-1)sd) is
        # a sensitivity in PITCH space, so `value` is the type-matched comparison and
        # `feature` is what our extractor actually produces for the real model. If value
        # aligns and feature does not, the EXTRACTOR is the bottleneck -- which would make
        # every model-side attribution number in this project unreadable.
        gq = torch.autograd.grad(sd[0], qh, retain_graph=True)[0][0]
        gx = torch.autograd.grad(sd[0], af)[0][0]
        rep = int(round(ORACLE_RATE / WAVLM_RATE))
        out[f"f0_sd/{stem}"] = np.repeat(
            (gx * af.detach()[0]).sum(dim=-1).detach().cpu().numpy().astype(np.float32), rep)
        out[f"value/{stem}"] = np.repeat(
            gq.detach().cpu().numpy().astype(np.float32), rep)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}  clips={len(out)}  (held-out: never trained on)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
