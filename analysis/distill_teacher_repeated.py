"""Repeated cross-fitting of the HSE-only teacher (locked 16/11/2026, before running).

Three 5-fold actor partitions of every outer-train set: partition 0 is the one of analysis/distill_teacher.py (seeds
SPLIT_SEED + 300 + k for the parts and SPLIT_SEED + 400 + 10k + j for the inner split); partitions r = 1, 2 add
1000 * r to both seeds. Within each partition the teacher recipe of distill_teacher.py is unchanged (head family,
weight decay and epoch chosen on an inner actor split of the four fitting parts, refit on them). Each training clip
gets one out-of-sample prediction per partition; the three-teacher target of a training clip is their mean. Every
head also predicts the outer held-out actors (never seen by any head of that outer fold).

Outputs (results/distill_teacher_ens/):
  teacher_v_ens3_f<k>.csv   train rows: mean of the clip's three out-of-sample predictions; heldout rows: mean of all
                            15 heads (a reference only, not the training-equivalent source)
  members_f<k>.csv          held-out predictions of each head (partition, part) and the out-of-sample train predictions
                            per partition (role = train, part = the clip's own part)
  choices.csv               the selected head family, weight decay and epoch per head
Partition 0 must reproduce results/distill_teacher/teacher_v_f<k>.csv (checked).

Usage (from the repository root; needs torch):
    python analysis/distill_teacher_repeated.py --out results/distill_teacher_ens
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from combined_head import hsemotion_pooled
from cremad_human import counts
from hsemotion_head import INNER_VAL_SHARE, MAX_EPOCHS, WEIGHT_DECAYS, train
from preprocess_cremad import LABELS, clip_table
from visual_probe_analysis import N_OUTER, SPLIT_SEED, outer_folds

N_CF, N_REP = 5, 3


def parts_of(actor_sex: pd.Series, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    out = {}
    for _, group in actor_sex.groupby(actor_sex):
        for j, a in enumerate(rng.permutation(group.index.to_numpy())):
            out[int(a)] = j % N_CF
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=Path("results/hsemotion_features/features.npz"))
    parser.add_argument("--reference", type=Path, default=Path("results/distill_teacher"))
    parser.add_argument("--out", type=Path, default=Path("results/distill_teacher_ens"))
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
    folds, actors, U = d.fold.to_numpy(), d.actor.to_numpy(), np.array(uids)
    choices = []
    for k in range(N_OUTER):
        tr_all, ho = folds != k, folds == k
        sexes = d[tr_all].groupby("actor")["Sex"].first()
        oof = np.full((N_REP,) + P.shape, np.nan, dtype=np.float32)
        mem = []
        for r in range(N_REP):
            parts = d.actor.map(parts_of(sexes, SPLIT_SEED + 300 + k + 1000 * r)).to_numpy()
            for j in range(N_CF):
                fit_all, pred = tr_all & (parts != j), tr_all & (parts == j)
                fa = sorted(set(actors[fit_all]))
                seed = SPLIT_SEED + 400 + 10 * k + j + 1000 * r
                inner = set(np.random.default_rng(seed).permutation(fa)[:max(1, round(INNER_VAL_SHARE * len(fa)))])
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
                    oof[r][pred] = torch.softmax(head(torch.from_numpy((X[pred] - mu) / sd)), -1).numpy()
                    qh = torch.softmax(head(torch.from_numpy((X[ho] - mu) / sd)), -1).numpy()
                mem.append(pd.DataFrame({"uid": U[ho], "role": "heldout", "partition": r, "part": j, **dict(zip(LABELS, qh.T))}))
                choices.append({"outer fold": k, "partition": r, "part": j, "head": kind, "weight decay": wd, "epoch": ep, "inner JSD": round(score, 4)})
            mem.append(pd.DataFrame({"uid": U[tr_all], "role": "train", "partition": r, "part": parts[tr_all], **dict(zip(LABELS, oof[r][tr_all].T))}))
        assert not np.isnan(oof[:, tr_all]).any()
        # partition 0 must reproduce the saved single-partition teacher
        ref = pd.read_csv(args.reference / f"teacher_v_f{k}.csv").query("role == 'train'").set_index("uid")[LABELS]
        diff = np.abs(ref.loc[U[tr_all]].to_numpy() - oof[0][tr_all]).max()
        assert diff < 1e-6, f"partition 0 does not reproduce the saved teacher (max diff {diff})"
        M = pd.concat(mem)
        M.to_csv(args.out / f"members_f{k}.csv", index=False, float_format="%.8f")
        hom = M[M.role == "heldout"].groupby("uid")[LABELS].mean()
        out = pd.concat([pd.DataFrame({"uid": U[tr_all], "role": "train", **dict(zip(LABELS, oof[:, tr_all].mean(0).T))}),
                         pd.DataFrame({"uid": hom.index, "role": "heldout", **{l: hom[l].to_numpy() for l in LABELS}})])
        assert np.allclose(out[LABELS].sum(1), 1, atol=1e-5)
        out.to_csv(args.out / f"teacher_v_ens3_f{k}.csv", index=False, float_format="%.8f")
        print(f"outer fold {k}: partition 0 reproduces the saved teacher (max diff {diff:.2e}); {N_REP * N_CF} heads")
    pd.DataFrame(choices).to_csv(args.out / "choices.csv", index=False)


if __name__ == "__main__":
    main()
