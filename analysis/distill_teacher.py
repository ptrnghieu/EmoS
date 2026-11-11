"""V distillation pilot, step 1: frozen HSE-only teacher distributions for the video-only reading (locked 11/11/2026).

For every outer fold k, using only its training actors (folds != k):
  - 5 cross-fitting parts by actor (stratified by sex, fixed seed). For each part j the teacher is fitted on the other
    four and predicts part j, so every training clip gets a teacher distribution from a head that never saw its actor.
  - Teacher recipe as analysis/hsemotion_head.py: HSEmotion mean + SD of the A2 face-crop frame embeddings (2560-d),
    standardised with the statistics of the clips being fitted, soft-label CE on the video-only votes, full-batch
    AdamW, weight-decay grid; head family (linear or MLP 128), weight decay and epoch chosen by JSD on an inner actor
    split (15%) of the four parts, then refitted on all four parts.
  - The five teachers also predict the outer held-out actors; their mean is the teacher reference for the analysis
    (an ensemble reference, reported as such).
Outputs per fold: teacher_v_f<k>.csv (uid, role = train / heldout, the six probabilities in LABELS order) and a
quality report of the cross-fitted teacher on the training clips (JSD, partial Spearman, entropy) and of the
ensemble on the held-out clips.

Usage (from the repository root; needs torch):
    python analysis/distill_teacher.py --out results/distill_teacher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from combined_head import hsemotion_pooled
from cremad_human import counts, jsd
from hsemotion_head import INNER_VAL_SHARE, MAX_EPOCHS, WEIGHT_DECAYS, train
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, SPLIT_SEED, norm_entropy, outer_folds, partial_spearman

N_CF, N_BOOT = 5, 1000


def cf_parts(actor_sex: pd.Series, k: int) -> dict[int, int]:
    rng = np.random.default_rng(SPLIT_SEED + 300 + k)
    out = {}
    for _, group in actor_sex.groupby(actor_sex):
        for j, a in enumerate(rng.permutation(group.index.to_numpy())):
            out[int(a)] = j % N_CF
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=Path("results/hsemotion_features/features.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/distill_teacher"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    hse = hsemotion_pooled(args.features)
    uids = [u for u in df.index if df.loc[u, "v_n"] > 0 and u in hse]
    d = df.loc[uids]
    X = np.stack([hse[u] for u in uids]).astype(np.float32)
    P = counts(d, "v")
    P = (P / P.sum(1, keepdims=True)).astype(np.float32)
    folds, actors = d.fold.to_numpy(), d.actor.to_numpy()
    choices, quality = [], []
    for k in range(N_OUTER):
        tr_all, ho = folds != k, folds == k
        parts = d.actor.map(cf_parts(d[tr_all].groupby("actor")["Sex"].first(), k)).to_numpy()
        q_train = np.full_like(P, np.nan)
        q_ho = []
        for j in range(N_CF):
            fit_all, pred = tr_all & (parts != j), tr_all & (parts == j)
            fa = sorted(set(actors[fit_all]))
            inner = set(np.random.default_rng(SPLIT_SEED + 400 + 10 * k + j).permutation(fa)[:max(1, round(INNER_VAL_SHARE * len(fa)))])
            val = fit_all & np.isin(actors, list(inner))
            fit = fit_all & ~val
            mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
            best = None
            for kind in ("linear", "mlp"):
                for wd in WEIGHT_DECAYS:
                    _, curve = train(kind, (X[fit] - mu) / sd, P[fit], wd, MAX_EPOCHS, (X[val] - mu) / sd, P[val])
                    ep = int(np.argmin(curve)) + 1
                    if best is None or curve[ep - 1] < best[0]:
                        best = (curve[ep - 1], kind, wd, ep)
            score, kind, wd, ep = best
            mu, sd = X[fit_all].mean(0), X[fit_all].std(0) + 1e-6
            head, _ = train(kind, (X[fit_all] - mu) / sd, P[fit_all], wd, ep)
            with torch.no_grad():
                q_train[pred] = torch.softmax(head(torch.from_numpy((X[pred] - mu) / sd)), -1).numpy()
                q_ho.append(torch.softmax(head(torch.from_numpy((X[ho] - mu) / sd)), -1).numpy())
            choices.append({"outer fold": k, "part": j, "head": kind, "weight decay": wd, "epoch": ep, "inner JSD": round(score, 4),
                            "fit actors": len(fa), "predicted clips": int(pred.sum())})
        assert not np.isnan(q_train[tr_all]).any()
        q_ho = np.mean(q_ho, 0)
        rows = [{"uid": u, "role": "train", **dict(zip(LABELS, q))} for u, q in zip(np.array(uids)[tr_all], q_train[tr_all])]
        rows += [{"uid": u, "role": "heldout", **dict(zip(LABELS, q))} for u, q in zip(np.array(uids)[ho], q_ho)]
        out = pd.DataFrame(rows)
        assert np.allclose(out[LABELS].sum(1), 1, atol=1e-5) and out.uid.is_unique
        out.to_csv(args.out / f"teacher_v_f{k}.csv", index=False, float_format="%.8f")
        for role, sel, q in (("cross-fitted, outer-train clips", tr_all, q_train[tr_all]), ("5-teacher mean, outer held-out clips", ho, q_ho)):
            Pv, hp, emo = P[sel], norm_entropy(P[sel]), d.intended.to_numpy()[sel]
            b = Bootstrap(actors[sel], N_BOOT)
            hq = norm_entropy(q)
            quality.append({"outer fold": k, "clips": role, "n": int(sel.sum()), "JSD V": b(lambda i: jsd(q[i], Pv[i]).mean()),
                            "Spearman(H(q), A^V)": b(lambda i: spearman(hq[i], hp[i])),
                            "partial given emotion": b(lambda i: partial_spearman(hq[i], hp[i], emo[i])),
                            "mean H(q)": round(float(hq.mean()), 3), "human mean H": round(float(hp.mean()), 3)})
    tables = {"quality": pd.DataFrame(quality), "choices": pd.DataFrame(choices)}
    (args.out / "meta.json").write_text(json.dumps({"labels": LABELS, "n_cf": N_CF, "features": "HSEmotion mean+SD (A2 crops)",
                                                    "target": "video-only votes"}, indent=1))
    lines = ["# Distillation teacher (HSE-only, cross-fitted within each outer-train set)\n",
             "Generated by `analysis/distill_teacher.py`; see its docstring.\n"]
    for name, t in tables.items():
        t.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(t.astype(object).where(t.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
