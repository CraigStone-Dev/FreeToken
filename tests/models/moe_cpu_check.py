"""fp64 CPU cross-check of the resident routed MoE (layer0) against the GPU captures.

If the GPU routed output matches the fp64 CPU computation for each run's own
(mlp_mix, router) inputs, the MoE kernel is TP-consistent and any TP=1/TP=2
routed divergence is input-driven (bf16 router noise with small top-2 gaps).
"""
import sys

import torch
import torch.nn.functional as F

def main(a_path, b_path):
    a = torch.load(a_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")
    # both captures hold the FULL (unsharded) layer0 expert weights; they must agree
    gu = a["experts_gate_up"].double()
    dp = a["experts_down"].double()
    wdiff = max((gu - b["experts_gate_up"].double()).abs().max().item(),
                (dp - b["experts_down"].double()).abs().max().item())
    print(f"expert weights tp1-vs-tp2 max_abs={wdiff:.4g} (must be 0)")
    E, H = gu.shape[0], dp.shape[1]
    inter = gu.shape[1] // 2

    def moe(x, r):
        T = x.shape[0]
        probs = torch.softmax(r, -1)
        w, idx = probs.topk(2, -1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros(T, H)
        for t in range(T):
            for j in range(2):
                e = idx[t, j].item()
                h = x[t] @ gu[e].T
                g, u = h[:inter], h[inter:]
                out[t] += w[t, j] * ((F.silu(g) * u) @ dp[e].T)
        return out

    for tag, cap in (("tp1", a), ("tp2", b)):
        x = cap["layers"]["layer0.mlp_mix"].double()
        r = cap["layers"]["layer0.router"].double()
        gpu = cap["layers"]["layer0.routed"].double()
        cpu = moe(x, r)
        d = (cpu - gpu).abs().max().item()
        print(f"{tag}: cpu-vs-gpu max_abs={d:.4g} scale={gpu.abs().max().item():.4g} "
              f"rel={d / max(gpu.abs().max().item(), 1e-9):.4g}")
    o1 = moe(a["layers"]["layer0.mlp_mix"].double(), a["layers"]["layer0.router"].double())
    o2 = moe(b["layers"]["layer0.mlp_mix"].double(), b["layers"]["layer0.router"].double())
    d = (o1 - o2).abs().max().item()
    print(f"cpu o1-vs-o2: max_abs={d:.4g} scale={o1.abs().max().item():.4g} "
          f"rel={d / max(o1.abs().max().item(), 1e-9):.4g}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
