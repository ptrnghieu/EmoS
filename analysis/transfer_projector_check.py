"""Transfer pilot diagnostic (after the locked analysis): size of the trained projector tokens per fold vs the
text-token embeddings. Usage: python analysis/transfer_projector_check.py <dir with init.pt, f0.pt, f1.pt, f2.pt> results/hsemotion_features/features.npz
(the .pt files are last/projector.pt of transfer_hse_tok_f<k>_s0 and its projector_init.pt)."""
import sys, numpy as np, torch, pandas as pd
sys.path.insert(0, "/home/user/EmoS/analysis")
from preprocess_cremad import clip_table
from visual_probe_analysis import outer_folds
D, F = sys.argv[1], sys.argv[2]
z = np.load(F); st = np.concatenate([[0], np.cumsum(z["lengths"])[:-1]]); fr = z["feats"].astype(np.float32)
P = {u: np.concatenate([fr[s:s+n].mean(0), fr[s:s+n].std(0)]) for u, s, n in zip(z["uid"].tolist(), st, z["lengths"])}
df = clip_table(); df = df[df.uid.isin(P)].reset_index(drop=True); df["fold"] = df.actor.map(outer_folds(df))
init = torch.load(f"{D}/init.pt", weights_only=True, map_location="cpu")
def run(sd, X):
    h = torch.nn.functional.gelu(X @ sd["fc1.weight"].T + sd["fc1.bias"])
    return (h @ sd["fc2.weight"].T + sd["fc2.bias"]) * sd["scale"]
for f in range(3):
    sd = torch.load(f"{D}/f{f}.pt", weights_only=True, map_location="cpu")
    tr = np.stack([P[u] for u in df.uid[df.fold != f]]); te = np.stack([P[u] for u in df.uid[df.fold == f]])
    mu, sd_ = tr.mean(0), tr.std(0) + 1e-6
    Xtr, Xte = (torch.tensor((tr - mu) / sd_)), torch.tensor((te - mu) / sd_)
    i2 = dict(init); i2["scale"] = sd["scale"]
    rel = {k: float((sd[k] - init[k]).norm() / init[k].norm()) for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")}
    with torch.no_grad():
        o0, otr, ote = run(i2, Xte), run(sd, Xtr), run(sd, Xte)
        zero = run(sd, torch.zeros(1, Xte.shape[1]))
    rms = lambda t: float(t.pow(2).mean().sqrt())
    # share of the token variance that varies across clips (vs the constant part)
    var_between = float(ote.var(0).mean()); 
    print(f"fold {f}: rel. weight change {', '.join(f'{k} {v:.2f}' for k, v in rel.items())} | token RMS init {rms(o0):.4f}, "
          f"trained held-out {rms(ote):.4f}, trained train {rms(otr):.4f}, zero input {rms(zero):.4f} | between-clip SD {var_between ** 0.5:.4f} "
          f"| max |token| {float(ote.abs().max()):.2f} | text RMS 0.0137")
