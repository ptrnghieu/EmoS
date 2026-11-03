"""Dedicated visual baseline: frozen HSEmotion frame features + a small head trained on the video-only vote
distributions. Does a different visual pipeline, with the same supervision and splits, reproduce the video-only raters
better than Qwen v_only? (Design approved 03/11/2026; locked before the outer evaluation.)

Input: results/hsemotion_features/features.npz (kaggle/hsemotion_features.ipynb): per frame of the A2 face crops (A0
timestamps, 2 fps), the 1280-d HSEmotion EfficientNet-B0 embedding (`enet_b0_8_best_afew`, frozen) and its 8 logits.

Comparison named for what it is: HSEmotion + face crop + learned head against Qwen v_only + full frame, same timestamps.

Clip feature: [mean, standard deviation] of the frame embeddings over time (2560-d; no temporal order). A clip with no
usable frame gets the training prior (mean training vote distribution) under every head.
Heads (both reported; neither chosen by outer scores):
    linear   2560 -> 6 logits (main)
    mlp      2560 -> 128 (ReLU) -> 6 logits
Training: soft-label cross-entropy to the video-only vote distribution, averaged equally over clips; no class weights,
no label smoothing, no entropy term. Features standardised with the statistics of the clips the head is fitted on.
Full-batch AdamW, lr 1e-3, up to MAX_EPOCHS epochs, torch seed 0.
Selection: the Stage 1 outer folds and the Stage 1 inner split (15% of each outer fold's training actors, same seed);
weight decay in WEIGHT_DECAYS and the epoch count chosen by the inner video-only JSD; then refitted on all of the outer
fold's training actors for that many epochs and applied to its held-out actors.
Zero-shot reference (secondary): the checkpoint's 8-class probabilities averaged over frames, the six classes shared with
CREMA-D kept (Contempt and Surprise dropped) and renormalised; the dropped mass is reported. Renormalising changes the
entropy, so this is not evidence about human disagreement.

Output: results/hsemotion_head/<name>/scores.jsonl (uid, condition V, logit_* = log probabilities) for name in linear,
mlp, zeroshot; selection.csv (inner grid and choices); report.md. Compared with v_only by vonly_compare.py --extra.

Usage (from the repository root; needs torch):
    python analysis/hsemotion_head.py --features results/hsemotion_features/features.npz --out results/hsemotion_head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cremad_human import counts, jsd
from preprocess_cremad import LABELS, clip_table
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, SPLIT_SEED, outer_folds

WEIGHT_DECAYS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
MAX_EPOCHS, LR, INNER_VAL_SHARE = 400, 1e-3, 0.15
HSE_CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
SHARED = {"Anger": "Anger", "Disgust": "Disgust", "Fear": "Fear", "Happy": "Happiness", "Neutral": "Neutral", "Sad": "Sadness"}


def make_head(kind: str, d: int) -> torch.nn.Module:
    torch.manual_seed(0)
    if kind == "linear":
        return torch.nn.Linear(d, len(LABELS))
    return torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(), torch.nn.Linear(128, len(LABELS)))


def train(kind, X, Y, wd, epochs, Xval=None, Yval=None):
    """Full-batch AdamW on soft-label CE; returns the head and, with a validation set, the JSD after every epoch."""
    head = make_head(kind, X.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=wd)
    x, y = torch.from_numpy(X), torch.from_numpy(Y)
    curve = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = -(y * torch.log_softmax(head(x), -1)).sum(-1).mean()
        loss.backward()
        opt.step()
        if Xval is not None:
            with torch.no_grad():
                p = torch.softmax(head(torch.from_numpy(Xval)), -1).numpy()
            curve.append(float(jsd(p, Yval).mean()))
    return head, curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=Path("results/hsemotion_features/features.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/hsemotion_head"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.features, allow_pickle=False)
    lengths = z["lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    feats, flog = z["feats"].astype(np.float32), z["logits"].astype(np.float64)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    uids = [u for u in z["uid"].tolist() if u in df.index and df.loc[u, "v_n"] > 0]
    row = {u: i for i, u in enumerate(z["uid"].tolist())}
    d = df.loc[uids]
    P = counts(d, "v")
    P = (P / P.sum(1, keepdims=True)).astype(np.float32)
    has = np.array([lengths[row[u]] > 0 for u in uids])
    X = np.zeros((len(uids), 2 * feats.shape[1]), np.float32)
    zs = np.full((len(uids), len(LABELS)), np.nan)
    keep = [HSE_CLASSES.index(SHARED[l]) for l in LABELS]
    dropped = np.full(len(uids), np.nan)
    for k, u in enumerate(uids):
        i = row[u]
        if lengths[i] == 0:
            continue
        f = feats[starts[i]:starts[i] + lengths[i]]
        X[k] = np.concatenate([f.mean(0), f.std(0)])
        pr = np.exp(flog[starts[i]:starts[i] + lengths[i]] - flog[starts[i]:starts[i] + lengths[i]].max(1, keepdims=True))
        pr = (pr / pr.sum(1, keepdims=True)).mean(0)
        dropped[k] = 1 - pr[keep].sum()
        zs[k] = pr[keep] / pr[keep].sum()

    folds, actors = d.fold.to_numpy(), d.actor.to_numpy()
    preds = {kind: np.zeros_like(P) for kind in ("linear", "mlp")}
    grid = []
    for fo in range(N_OUTER):
        tr_all, te = folds != fo, folds == fo
        train_actors = sorted(set(actors[tr_all]))  # the Stage 1 inner split of this outer fold
        inner = set(np.random.default_rng(SPLIT_SEED + 1 + fo).permutation(train_actors)[:max(1, round(INNER_VAL_SHARE * len(train_actors)))])
        val = tr_all & np.isin(actors, list(inner))
        fit = tr_all & ~val
        prior = P[tr_all & has].mean(0)
        for kind in ("linear", "mlp"):
            a, b_ = fit & has, val & has
            mu, sd = X[a].mean(0), X[a].std(0) + 1e-6
            best = None
            for wd in WEIGHT_DECAYS:
                _, curve = train(kind, (X[a] - mu) / sd, P[a], wd, MAX_EPOCHS, (X[b_] - mu) / sd, P[b_])
                ep = int(np.argmin(curve)) + 1
                grid.append({"fold": fo, "head": kind, "weight decay": wd, "best epoch": ep, "inner JSD at best epoch": round(curve[ep - 1], 4),
                             "inner JSD at last epoch": round(curve[-1], 4)})
                if best is None or curve[ep - 1] < best[0]:
                    best = (curve[ep - 1], wd, ep)
            _, wd, ep = best
            grid.append({"fold": fo, "head": kind, "weight decay": wd, "best epoch": ep, "inner JSD at best epoch": round(best[0], 4),
                         "inner JSD at last epoch": "chosen"})
            full = tr_all & has
            mu, sd = X[full].mean(0), X[full].std(0) + 1e-6
            head, _ = train(kind, (X[full] - mu) / sd, P[full], wd, ep)
            with torch.no_grad():
                p = torch.softmax(head(torch.from_numpy((X[te] - mu) / sd)), -1).numpy()
            p[~has[te]] = prior
            preds[kind][te] = p
        zs[te & ~has] = np.full(len(LABELS), 1 / len(LABELS))

    for name, Q in list(preds.items()) + [("zeroshot", zs)]:
        (args.out / name).mkdir(exist_ok=True)
        with open(args.out / name / "scores.jsonl", "w", encoding="utf-8") as fh:
            for u, q in zip(uids, Q):
                fh.write(json.dumps({"model": f"hsemotion-{name}", "uid": u, "condition": "V", "task": "score",
                                     **{f"logit_{l}": float(np.log(max(v, 1e-12))) for l, v in zip(LABELS, q)}}) + "\n")
    sel = pd.DataFrame(grid)
    sel.to_csv(args.out / "selection.csv", index=False)
    summary = pd.DataFrame([{"clips": len(uids), "clips without a usable frame": int((~has).sum()),
                             "zero-shot: mean probability on Contempt + Surprise (dropped)": round(float(np.nanmean(dropped)), 3),
                             **{f"{k} JSD (all clips, for orientation)": round(float(jsd(Q, P).mean()), 3) for k, Q in list(preds.items()) + [("zeroshot", zs)]}}])
    summary.to_csv(args.out / "summary.csv", index=False)
    chosen = sel[sel["inner JSD at last epoch"] == "chosen"]
    lines = ["# HSEmotion frame features + head on the video-only votes\n",
             "Generated by `analysis/hsemotion_head.py`. Choices made on the Stage 1 inner split only; the outer comparison "
             "with v_only is in `results/vonly_compare/`.\n", "## chosen\n", markdown(chosen), "", "## summary\n", markdown(summary), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
