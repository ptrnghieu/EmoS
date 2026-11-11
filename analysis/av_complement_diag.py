"""Follow-up to av_complement_head.py (agreed 11/11/2026, after its results were seen; exploratory).

Part A, optimisation vs generalisation of the heads already used (AV target, R0 and R1, linear and MLP). Each of the
9 fits per set (3 checkpoints x 3 head folds) is repeated with its own inner split, for every weight decay of the
grid, logging per epoch the training CE and JSD on the fit part and the JSD on the inner validation actors, plus the
gradient norm of the last epoch. Reference: JSD of Qwen's own AV output on the same clips. A linear head on the
standardised logits can represent Qwen's own output exactly, so a training JSD above Qwen's own on the fit part is an
optimisation shortfall; a training JSD below it with a validation JSD above it is a generalisation gap.

Part B, positive control expected (not guaranteed) on the video-only votes, same actor splits and recipe as
av_complement_head.py (heads within each checkpoint's held-out actors, 3 head folds, inner split, refit):
    Qwen V (s1_noI's own V output, no head) | head on V logits | head on HSE only | head on V logits + HSE
If HSE only is good but V logits + HSE is not, the problem lies in learning the combination rather than in the feature.

Usage (from the repository root; needs torch):
    python analysis/av_complement_diag.py --out results/av_complement_diag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from av_complement_head import KINDS, N_BOOT, N_HEAD, fit_predict, head_folds, read_logits
from combined_head import hsemotion_pooled
from cremad_human import counts, jsd
from hsemotion_head import INNER_VAL_SHARE, LR, MAX_EPOCHS, WEIGHT_DECAYS, make_head
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, softmax
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, SPLIT_SEED, norm_entropy, outer_folds, partial_spearman


def logged_train(kind, X, Y, wd, epochs, Xval, Yval):
    """As hsemotion_head.train, plus per-epoch training CE / JSD and the last gradient norm."""
    head = make_head(kind, X.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=wd)
    x, y, xv = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xval)
    log, gnorm = [], np.nan
    for _ in range(epochs):
        opt.zero_grad()
        out = head(x)
        loss = -(y * torch.log_softmax(out, -1)).sum(-1).mean()
        loss.backward()
        gnorm = float(sum(p.grad.pow(2).sum() for p in head.parameters()) ** 0.5)
        opt.step()
        with torch.no_grad():
            pt = torch.softmax(out, -1).numpy()
            pv = torch.softmax(head(xv), -1).numpy()
        log.append((float(loss), float(jsd(pt, Y).mean()), float(jsd(pv, Yval).mean())))
    return np.array(log), gnorm


def load_checkpoint(k, df, hse, stage1, target):
    run = stage1 / f"stage1_s1_noI_f{k}_s0"
    lg = {c: read_logits(run / "scores.jsonl", c) for c in ("AV", "V", "A")}
    key = {"AV": "av", "V": "v"}[target]
    uids = [u for u in lg["AV"].index if u in df.index and df.loc[u, "fold"] == k and df.loc[u, f"{key}_n"] > 0
            and u in hse and all(u in lg[c].index for c in ("V", "A"))]
    d = df.loc[uids]
    P = counts(d, key)
    P = (P / P.sum(1, keepdims=True)).astype(np.float32)
    hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
    H = np.stack([hse[u] for u in uids]).astype(np.float32)
    return uids, d, P, hf, H, lg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--features", type=Path, default=Path("results/hsemotion_features/features.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/av_complement_diag"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    hse = hsemotion_pooled(args.features)

    # ---- Part A: optimisation vs generalisation (AV) -------------------------------------------------------------
    rows = []
    for k in range(N_OUTER):
        uids, d, P, hf, H, lg = load_checkpoint(k, df, hse, args.stage1, "AV")
        actors = d.actor.to_numpy()
        z_av = lg["AV"].loc[uids].to_numpy(np.float32)
        own = softmax(z_av.astype(float))
        feats = {"R0": z_av, "R1": np.hstack([z_av, H])}
        for j in range(N_HEAD):
            tr = hf != j
            tr_act = sorted(set(actors[tr]))
            inner = set(np.random.default_rng(SPLIT_SEED + 200 + 10 * k + j).permutation(tr_act)[:max(1, round(INNER_VAL_SHARE * len(tr_act)))])
            val = tr & np.isin(actors, list(inner))
            fit = tr & ~val
            for n, X in feats.items():
                mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
                for kind in KINDS:
                    for wd in WEIGHT_DECAYS:
                        log, g = logged_train(kind, (X[fit] - mu) / sd, P[fit], wd, MAX_EPOCHS, (X[val] - mu) / sd, P[val])
                        b = int(np.argmin(log[:, 2]))
                        rows.append({"checkpoint": k, "head fold": j, "features": n, "head": kind, "weight decay": wd,
                                     "train CE first": log[0, 0], "train CE last": log[-1, 0],
                                     "train CE drop, last 50 epochs": log[-51, 0] - log[-1, 0],
                                     "grad norm last": g, "train JSD last": log[-1, 1], "val JSD last": log[-1, 2],
                                     "best val epoch": b + 1, "val JSD best": log[b, 2], "train JSD at best val": log[b, 1],
                                     "Qwen own JSD fit": float(jsd(own[fit], P[fit]).mean()),
                                     "Qwen own JSD val": float(jsd(own[val], P[val]).mean())})
    A = pd.DataFrame(rows)
    A.to_csv(args.out / "partA_fits.csv", index=False)
    summ = A.groupby(["features", "head", "weight decay"]).mean(numeric_only=True).reset_index()
    summ = summ[["features", "head", "weight decay", "train CE last", "train CE drop, last 50 epochs", "grad norm last",
                 "train JSD last", "Qwen own JSD fit", "val JSD best", "val JSD last", "Qwen own JSD val", "best val epoch"]].round(4)

    # ---- Part B: positive control on the video-only votes --------------------------------------------------------
    sets = {"V logits": lambda z, H: z, "HSE only": lambda z, H: H, "V logits + HSE": lambda z, H: np.hstack([z, H])}
    prows = []
    for k in range(N_OUTER):
        uids, d, P, hf, H, lg = load_checkpoint(k, df, hse, args.stage1, "V")
        actors = d.actor.to_numpy()
        z_v = lg["V"].loc[uids].to_numpy(np.float32)
        pred = {(n, kind): np.zeros_like(P) for n in sets for kind in KINDS}
        for j in range(N_HEAD):
            te, tr = hf == j, hf != j
            for n, f in sets.items():
                for kind in KINDS:
                    pred[(n, kind)][te], _ = fit_predict(kind, f(z_v, H), P, tr, te, actors, SPLIT_SEED + 200 + 10 * k + j)
        own = softmax(z_v.astype(float))
        for i, u in enumerate(uids):
            prows.append({"uid": u, "checkpoint": k, "actor": int(actors[i]), "emotion": d.intended.iloc[i], "p": P[i],
                          "Qwen V (no head)": own[i], **{f"{n} ({kind})": pred[(n, kind)][i] for n in sets for kind in KINDS}})
    B = pd.DataFrame(prows)
    Pv = np.stack(B.p)
    hp, emo, act, ck = norm_entropy(Pv), B.emotion.to_numpy(), B.actor.to_numpy(), B.checkpoint.to_numpy()
    names = ["Qwen V (no head)"] + [f"{n} ({kind})" for kind in KINDS for n in sets]
    Q = {n: np.stack(B[n]) for n in names}
    boot = Bootstrap(act, N_BOOT)
    met = lambda q: {"JSD V": lambda i: jsd(q[i], Pv[i]).mean(),
                     "partial V": lambda i: partial_spearman(norm_entropy(q[i]), hp[i], emo[i])}
    readings = pd.DataFrame([{"predictor": n, **{m: boot(f) for m, f in met(Q[n]).items()},
                              **{f"JSD ckpt {c}": round(float(jsd(Q[n], Pv)[ck == c].mean()), 4) for c in range(N_OUTER)}} for n in names])
    pairs = []
    for kind in KINDS:
        pairs += [(f"V logits + HSE ({kind})", f"V logits ({kind})"), (f"HSE only ({kind})", "Qwen V (no head)"),
                  (f"HSE only ({kind})", f"V logits ({kind})"), (f"V logits + HSE ({kind})", f"HSE only ({kind})"),
                  (f"V logits ({kind})", "Qwen V (no head)")]
    paired = pd.DataFrame([{"comparison": f"{a} minus {b}", **{m: boot(lambda i, m=m: met(Q[a])[m](i) - met(Q[b])[m](i)) for m in ("JSD V", "partial V")}}
                           for a, b in pairs])

    tables = {"partA_summary": summ, "partB_readings": readings, "partB_paired": paired}
    lines = ["# AV complementarity follow-up (exploratory): head diagnostics and a positive control on V\n",
             "Generated by `analysis/av_complement_diag.py`; see its docstring. Part A rows are means over the 9 fits "
             "(3 checkpoints x 3 head folds), each on its inner split. Part B uses the splits and recipe of "
             "`av_complement_head.py` with the video-only votes as target.\n"]
    for name, t in tables.items():
        t.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(t.astype(object).where(t.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
