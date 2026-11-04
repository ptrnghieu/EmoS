"""Complementarity test: HSEmotion features and the Qwen v_only hidden state together, read out by the same head family.
Does Qwen carry information useful to the video-only votes that the HSEmotion pipeline lacks? (Design approved 04/11/2026.)

Per outer fold k, every clip's feature = [Qwen v_only `norm` state from fold k's checkpoint (3584-d, as qwen_head.py) ;
HSEmotion mean and standard deviation of the A2 face-crop frame embeddings (2560-d, as hsemotion_head.py)]. Each dimension
is standardised with the statistics of the clips the head is fitted on, which standardises each source on its own; no
re-weighting between the two blocks. Head family (linear main, MLP 128 secondary), loss, weight-decay grid, epoch choice on
the Stage 1 inner split and refit: exactly as hsemotion_head.py and qwen_head.py. The inner choice is not independent of
the Qwen checkpoint (its adapter was trained on those actors), as in qwen_head.py.

Compared with HSEmotion + head and Qwen + head by vonly_compare.py (--extra ... --pairs comb_linear:hse_linear ...).

Usage (from the repository root; needs torch):
    python analysis/combined_head.py --runs results/visual_probe --features results/hsemotion_features/features.npz --out results/combined_head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cremad_human import counts
from hsemotion_head import INNER_VAL_SHARE, MAX_EPOCHS, WEIGHT_DECAYS, train
from preprocess_cremad import LABELS, clip_table
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, SPLIT_SEED, load_run, outer_folds


def hsemotion_pooled(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    lengths = z["lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    feats = z["feats"].astype(np.float32)
    out = {}
    for u, s, n in zip(z["uid"].tolist(), starts, lengths):
        if n > 0:
            f = feats[s:s + n]
            out[u] = np.concatenate([f.mean(0), f.std(0)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/visual_probe"))
    parser.add_argument("--features", type=Path, default=Path("results/hsemotion_features/features.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/combined_head"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    hse = hsemotion_pooled(args.features)
    preds = {kind: {} for kind in ("linear", "mlp")}
    grid, sizes = [], []
    for fo in range(N_OUTER):
        uids, arrays, meta = load_run(args.runs / f"vprobe_v_only_A0_f{fo}_all")
        assert meta.get("model_kind") == "v_only" and meta.get("fold") == fo and meta.get("all_clips"), meta
        keep = [i for i, u in enumerate(uids) if u in df.index and df.loc[u, "v_n"] > 0 and u in hse]
        uids = [uids[i] for i in keep]
        X = np.hstack([arrays["norm"][keep].astype(np.float32), np.stack([hse[u] for u in uids])])
        d = df.loc[uids]
        P = counts(d, "v")
        P = (P / P.sum(1, keepdims=True)).astype(np.float32)
        folds, actors = d.fold.to_numpy(), d.actor.to_numpy()
        tr_all, te = folds != fo, folds == fo
        train_actors = sorted(set(actors[tr_all]))
        inner = set(np.random.default_rng(SPLIT_SEED + 1 + fo).permutation(train_actors)[:max(1, round(INNER_VAL_SHARE * len(train_actors)))])
        val = tr_all & np.isin(actors, list(inner))
        fit = tr_all & ~val
        sizes.append({"fold": fo, "clips": len(uids), "feature dimensions (Qwen + HSEmotion)": f"{arrays['norm'].shape[1]} + {X.shape[1] - arrays['norm'].shape[1]}",
                      "fit / inner / held-out clips": f"{int(fit.sum())} / {int(val.sum())} / {int(te.sum())}"})
        for kind in ("linear", "mlp"):
            mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
            best = None
            for wd in WEIGHT_DECAYS:
                _, curve = train(kind, (X[fit] - mu) / sd, P[fit], wd, MAX_EPOCHS, (X[val] - mu) / sd, P[val])
                ep = int(np.argmin(curve)) + 1
                grid.append({"fold": fo, "head": kind, "weight decay": wd, "best epoch": ep, "inner JSD at best epoch": round(curve[ep - 1], 4)})
                if best is None or curve[ep - 1] < best[0]:
                    best = (curve[ep - 1], wd, ep)
            _, wd, ep = best
            grid.append({"fold": fo, "head": kind, "weight decay": wd, "best epoch": ep, "inner JSD at best epoch": round(best[0], 4), "chosen": True})
            mu, sd = X[tr_all].mean(0), X[tr_all].std(0) + 1e-6
            head, _ = train(kind, (X[tr_all] - mu) / sd, P[tr_all], wd, ep)
            with torch.no_grad():
                p = torch.softmax(head(torch.from_numpy((X[te] - mu) / sd)), -1).numpy()
            preds[kind].update({u: q for u, q in zip(np.array(uids)[te], p)})

    for kind, pr in preds.items():
        (args.out / kind).mkdir(exist_ok=True)
        with open(args.out / kind / "scores.jsonl", "w", encoding="utf-8") as fh:
            for u, q in pr.items():
                fh.write(json.dumps({"model": f"hsemotion+qwen-v_only-{kind}", "uid": u, "condition": "V", "task": "score",
                                     **{f"logit_{l}": float(np.log(max(v, 1e-12))) for l, v in zip(LABELS, q)}}) + "\n")
    tables = {"sizes": pd.DataFrame(sizes), "selection": pd.DataFrame(grid)}
    lines = ["# HSEmotion + Qwen v_only (norm) features, same head family, on the video-only votes\n",
             "Generated by `analysis/combined_head.py`. Inner choices are not independent of the Qwen checkpoint. "
             "The comparison is in `results/vonly_compare/`.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
