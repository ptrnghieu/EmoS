"""Control for the V distillation pilot (locked 12/11/2026, before training): the cross-fitted teacher distributions
of results/distill_teacher, permuted between training clips.

Within every outer-train set, whole teacher probability vectors are permuted among clips that share the intended
emotion and the cross-fitting part of analysis/distill_teacher.py (so every permuted target still comes from a teacher
that never saw that clip's actor). The permutation is a fixed-seed cyclic shift of a shuffled order inside each group,
so no clip keeps its own vector (unless its group has one clip). It keeps the set of teacher distributions, including
their entropy distribution, per emotion and part; it breaks the match between teacher distribution and clip. Held-out
rows are copied unchanged (not used in training).
Outputs: results/distill_teacher/teacher_v_perm_f<k>.csv and perm_report.csv (fixed points, entropy of the mixed
targets 2/3 votes + 1/3 teacher with the true and the permuted teacher).

Usage (from the repository root):
    python analysis/distill_teacher_perm.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import counts
from distill_teacher import cf_parts
from preprocess_cremad import LABELS, clip_table
from visual_probe_analysis import N_OUTER, SPLIT_SEED, norm_entropy, outer_folds

ALPHA = 1 / 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    args = parser.parse_args()
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    rows = []
    for k in range(N_OUTER):
        t = pd.read_csv(args.teacher / f"teacher_v_f{k}.csv")
        assert list(t.columns[2:]) == LABELS
        tr = t[t.role == "train"].set_index("uid")
        d = df.loc[tr.index]
        part = d.actor.map(cf_parts(df[df.fold != k].groupby("actor")["Sex"].first(), k))
        rng = np.random.default_rng(SPLIT_SEED + 500 + k)
        new = tr[LABELS].copy()
        fixed = 0
        for (emo, j), g in d.groupby([d.intended, part]):
            order = rng.permutation(g.index.to_numpy())
            donor = np.roll(order, -1)
            new.loc[order, LABELS] = tr.loc[donor, LABELS].to_numpy()
            fixed += int(len(order) == 1)
        out = pd.concat([new.reset_index().assign(role="train")[["uid", "role"] + LABELS],
                         t[t.role == "heldout"][["uid", "role"] + LABELS]])
        path = args.teacher / f"teacher_v_perm_f{k}.csv"
        out.to_csv(path, index=False, float_format="%.8f")
        P = counts(d, "v")
        P = P / P.sum(1, keepdims=True)
        h_true = norm_entropy((1 - ALPHA) * P + ALPHA * tr[LABELS].to_numpy())
        h_perm = norm_entropy((1 - ALPHA) * P + ALPHA * new[LABELS].to_numpy())
        rows.append({"outer fold": k, "training clips": len(tr), "groups (emotion x part)": d.groupby([d.intended, part]).ngroups,
                     "clips keeping their own vector": fixed + int((new[LABELS].to_numpy() == tr[LABELS].to_numpy()).all(1).sum() - fixed),
                     "teacher entropy mean (unchanged)": round(float(norm_entropy(tr[LABELS].to_numpy()).mean()), 4),
                     "mixed target entropy, true teacher": round(float(h_true.mean()), 4),
                     "mixed target entropy, permuted teacher": round(float(h_perm.mean()), 4),
                     "mean |entropy change| per clip": round(float(np.abs(h_perm - h_true).mean()), 4),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]})
    rep = pd.DataFrame(rows)
    rep.to_csv(args.teacher / "perm_report.csv", index=False)
    print(rep.to_string())


if __name__ == "__main__":
    main()
