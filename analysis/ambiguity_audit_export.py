"""Exporter and repository add-on for analysis/ambiguity_audit (17/11/2026; descriptive, no model fitting).

1. Writes the JSONL input of analysis/ambiguity_audit/audit_predictions.py from exactly the V sources of step 0
   (analysis/kl_decomposition.py, the same files, the same clip filter and the same calibration):
     cont      distill_cont_f<k>_s0, one global temperature per head fold chosen by JSD on the other two head folds
               (GRID), applied to the third (the calibration used in step 0)
     s1_noI    stage1_s1_noI_f<k>_s0, softmax of the logits, uncalibrated (as in step 0)
     hse_mlp   results/hsemotion_head/mlp, softmax of the logits
     teacher   the five-teacher mean of the cross-fitted HSE-only teachers on the held-out actors (teacher_v_f<k>.csv)
   Class order: preprocess_cremad.LABELS. Folds: the Stage 1 outer folds by actor.
2. rho_within with the project's evaluator (mean over intended emotions of Spearman(H(q), H(p)), as in
   analysis/merger_probe.py) and partial Spearman given the intended emotion, for baseline and comparator on the same
   clips, in the groups all / baseline_correct / both_correct (the audit's definitions: unique human majority, unique
   model top, equal), paired bootstrap over actors (rq3_analysis.Bootstrap, N_BOOT). Descriptive and conditional, like
   the audit's other subgroups.

Usage (from the repository root):
    python analysis/ambiguity_audit_export.py --out results/ambiguity_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from av_complement_head import N_BOOT, N_HEAD, head_folds
from cremad_human import counts, jsd
from distill_calibration import GRID
from kl_decomposition import read
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, softmax
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman

SOURCES = ("cont", "s1_noI", "hse_mlp", "teacher")


def unique_correct(p, q):
    return ((p == p.max(1, keepdims=True)).sum(1) == 1) & ((q == q.max(1, keepdims=True)).sum(1) == 1) & (p.argmax(1) == q.argmax(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    parser.add_argument("--hse", type=Path, default=Path("results/hsemotion_head"))
    parser.add_argument("--out", type=Path, default=Path("results/ambiguity_audit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    hse = read(args.hse / "mlp" / "scores.jsonl")
    lines, R = [], []
    for k in range(N_OUTER):
        zc = read(args.runs / f"distill_cont_f{k}_s0" / "scores.jsonl")
        zr = read(args.runs / f"distill_rank_teacher_f{k}_s0" / "scores.jsonl")  # only for step 0's clip filter
        z1 = read(args.stage1 / f"stage1_s1_noI_f{k}_s0" / "scores.jsonl")
        t = pd.read_csv(args.teacher / f"teacher_v_f{k}.csv").query("role == 'heldout'").set_index("uid")[LABELS]
        hl = read(args.hse / "linear" / "scores.jsonl")  # only for step 0's clip filter
        uids = [u for u in zc.index if u in zr.index and u in z1.index and u in t.index and u in hse.index and u in hl.index
                and u in df.index and df.loc[u, "fold"] == k and df.loc[u, "v_n"] > 0]
        d = df.loc[uids]
        C = counts(d, "v")
        P = C / C.sum(1, keepdims=True)
        hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
        Z = zc.loc[uids].to_numpy(float)
        cal = np.zeros_like(P)
        for j in range(N_HEAD):
            te, tr = hf == j, hf != j
            tt = float(GRID[int(np.argmin([jsd(softmax(Z[tr], x), P[tr]).mean() for x in GRID]))])
            cal[te] = softmax(Z[te], tt)
        src = {"cont": cal, "s1_noI": softmax(z1.loc[uids].to_numpy(float)), "hse_mlp": softmax(hse.loc[uids].to_numpy(float)),
               "teacher": t.loc[uids].to_numpy(float)}
        for name, q in src.items():
            q = q / q.sum(1, keepdims=True)
            for i, u in enumerate(uids):
                lines.append(json.dumps({"clip_id": u, "actor_id": str(int(d.actor.iloc[i])), "fold": str(k), "condition": "V", "model": name,
                                         "classes": list(LABELS), "counts": [int(c) for c in C[i]], "probs": [float(x) for x in q[i]]}))
        R.append(pd.DataFrame({"uid": uids, "actor": d.actor.to_numpy(), "emotion": d.intended.to_numpy(), "C": list(C),
                               **{n: list(q) for n, q in src.items()}}))
    (args.out / "predictions_V.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    R = pd.concat(R, ignore_index=True)
    C = np.stack(R.C).astype(float)
    P = C / C.sum(1, keepdims=True)
    hp, emo, act = norm_entropy(P), R.emotion.to_numpy(), R.actor.to_numpy()
    Q = {n: np.stack(R[n]) for n in SOURCES}
    H = {n: norm_entropy(q) for n, q in Q.items()}
    ok = {n: unique_correct(P, q) for n, q in Q.items()}
    EMOS = sorted(set(emo))

    def within(h, idx):
        return float(np.nanmean([spearman(h[idx][emo[idx] == e], hp[idx][emo[idx] == e]) for e in EMOS]))

    rows = []
    for base in ("cont", "s1_noI"):
        for comp in ("hse_mlp", "teacher"):
            for group, mask in (("all", np.ones(len(P), bool)), ("baseline_correct", ok[base]), ("both_correct", ok[base] & ok[comp])):
                sel = np.flatnonzero(mask)
                boot = Bootstrap(act[sel], N_BOOT)
                hb, hc, hpp, em = H[base][sel], H[comp][sel], hp[sel], emo[sel]
                w = lambda h, i: float(np.nanmean([spearman(h[i][em[i] == e], hpp[i][em[i] == e]) for e in EMOS]))
                rows.append({"baseline": base, "comparator": comp, "group": group, "clips": len(sel), "actors": len(set(act[sel])),
                             "rho_within baseline": boot(lambda i: w(hb, i)), "rho_within comparator": boot(lambda i: w(hc, i)),
                             "delta rho_within (comparator - baseline)": boot(lambda i: w(hc, i) - w(hb, i)),
                             "partial baseline": boot(lambda i: partial_spearman(hb[i], hpp[i], em[i])),
                             "partial comparator": boot(lambda i: partial_spearman(hc[i], hpp[i], em[i])),
                             "delta partial": boot(lambda i: partial_spearman(hc[i], hpp[i], em[i]) - partial_spearman(hb[i], hpp[i], em[i]))})
    tb = pd.DataFrame(rows)
    tb.to_csv(args.out / "rho_within.csv", index=False)
    rep = ["# Ambiguity audit add-on: rho_within and partial Spearman with the project evaluator (V)\n",
           f"Generated by `analysis/ambiguity_audit_export.py`; {len(P)} clips. Descriptive, conditional subgroups.\n",
           markdown(tb.astype(object).where(tb.notna(), ""))]
    (args.out / "rho_within.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
