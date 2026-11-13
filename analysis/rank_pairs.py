"""Ranking-transfer pilot, step 1 (locked 13/11/2026): the shared pair set and how much the two ranking sources disagree.

For every outer fold k and every intended emotion, over the outer-train clips: u_T = percentile of the normalised
entropy of the cross-fitted HSE-only teacher (results/distill_teacher/teacher_v_f<k>.csv, train rows), u_H =
percentile of the normalised entropy of the human video-only votes. Percentiles use midranks for ties:
u = (midrank - 0.5) / n. A pair (i, j) of the same emotion is valid when |u_T(i) - u_T(j)| >= 0.3 and
|u_H(i) - u_H(j)| >= 0.3; both arms (rank_teacher, rank_votes) use exactly these pairs and differ only in the sign
r_ij (from u_T or from u_H). Pairs where the two sources order the clips differently are kept.

Reports, per fold: share of same-emotion pairs that are valid, share of valid pairs on which the two sources disagree,
and, simulating the training queue (the 256 most recent training clips in total, filtered to the same emotion; the
loss of a clip is computed before it enters the queue), the mean number of valid pairs per clip and the share of
clips without any pair. Writes results/distill_teacher/rank_u_f<k>.csv (uid, intended, u_T, u_H) for the notebook.

Usage (from the repository root):
    python analysis/rank_pairs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cremad_human import counts
from preprocess_cremad import LABELS, clip_table
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds

MIN_DU, QUEUE = 0.3, 256


def pct(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 0.5) / len(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    args = parser.parse_args()
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    rows = []
    for k in range(N_OUTER):
        t = pd.read_csv(args.teacher / f"teacher_v_f{k}.csv")
        t = t[t.role == "train"].set_index("uid")
        d = df.loc[t.index]
        P = counts(d, "v")
        P = P / P.sum(1, keepdims=True)
        hT, hH = norm_entropy(t[LABELS].to_numpy()), norm_entropy(P)
        u = pd.DataFrame({"intended": d.intended.to_numpy(), "hT": hT, "hH": hH}, index=t.index)
        u["u_T"] = u.groupby("intended")["hT"].transform(lambda s: pct(s.to_numpy()))
        u["u_H"] = u.groupby("intended")["hH"].transform(lambda s: pct(s.to_numpy()))
        u[["intended", "u_T", "u_H"]].reset_index().to_csv(args.teacher / f"rank_u_f{k}.csv", index=False, float_format="%.6f")
        n_pairs = n_valid = n_dis = 0
        for _, g in u.groupby("intended"):
            a, b = g.u_T.to_numpy(), g.u_H.to_numpy()
            dT, dH = a[:, None] - a[None, :], b[:, None] - b[None, :]
            iu = np.triu_indices(len(g), 1)
            dT, dH = dT[iu], dH[iu]
            valid = (np.abs(dT) >= MIN_DU) & (np.abs(dH) >= MIN_DU)
            n_pairs += len(dT)
            n_valid += int(valid.sum())
            n_dis += int((np.sign(dT[valid]) != np.sign(dH[valid])).sum())
        # queue simulation with a random clip order (the notebook's order is a permutation of the training clips)
        order = np.random.default_rng(0).permutation(len(u))
        emo, uT, uH = u.intended.to_numpy()[order], u.u_T.to_numpy()[order], u.u_H.to_numpy()[order]
        per_clip = []
        for i in range(len(order)):
            lo = max(0, i - QUEUE)
            same = emo[lo:i] == emo[i]
            v = same & (np.abs(uT[lo:i] - uT[i]) >= MIN_DU) & (np.abs(uH[lo:i] - uH[i]) >= MIN_DU)
            per_clip.append(int(v.sum()))
        per_clip = np.array(per_clip)
        rows.append({"outer fold": k, "training clips": len(u), "same-emotion pairs": n_pairs,
                     "valid share": round(n_valid / n_pairs, 4), "valid pairs": n_valid,
                     "sources disagree (share of valid)": round(n_dis / n_valid, 4),
                     "Spearman(u_T, u_H) within emotion": round(float(u.groupby("intended").apply(lambda g: g.u_T.corr(g.u_H, method="spearman")).mean()), 3),
                     "queue: valid pairs per clip (mean)": round(float(per_clip[QUEUE:].mean()), 1),
                     "queue: clips without a pair": round(float((per_clip[QUEUE:] == 0).mean()), 4)})
    rep = pd.DataFrame(rows)
    rep.to_csv(args.teacher / "rank_pairs_report.csv", index=False)
    print(rep.to_string())


if __name__ == "__main__":
    main()
