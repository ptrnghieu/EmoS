"""Diagnostic before any further GPU run (15/11/2026): does the teacher's between-class structure bring the V prediction
closer to the votes when the level of disagreement (entropy) is held fixed?

On the held-out actors of the three outer folds, for a student S (main: `cont` after the calibration protocol of
rank_compare.py; also `cont` at T = 1 and `rank_teacher` calibrated) and the teacher T (mean of the five cross-fitted
HSE-only teachers on the held-out actors), four predictions per clip:
    S                       student structure, student entropy
    T                       teacher structure, teacher entropy
    T-structure, S-entropy  the teacher's logits (log t) at a per-clip temperature giving the student's entropy
    S-structure, T-entropy  the student's logits at a per-clip temperature giving the teacher's entropy
A per-clip temperature keeps the order of the classes (and the top label) of its structure source and sets the
normalised entropy; it is found by bisection on log temperature in [-6, 6]; clips where the target entropy is out of
reach are counted and reported (the bound is used).
Not a deployable model, not a ceiling, and no evidence that a student could learn the teacher's structure. With the
entropy held fixed, partial Spearman cannot change by the structure swap itself (the entropies are identical); the
diagnostic quantity is JSD V (and the top label among the raters' top labels). Added 15/11: the same comparison within
the clips where teacher and student share the top label and where they do not (groups, not a causal decomposition).

Usage (from the repository root):
    python analysis/structure_entropy_swap.py --out results/structure_entropy_swap
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
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, fmt, softmax
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman


def read(path: Path) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == "V") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    z = s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
    return z.sub(z.mean(1), axis=0)


def at_entropy(z: np.ndarray, target: np.ndarray):
    """Per row, softmax(z / tau) with normalised entropy equal to target (bisection on log tau)."""
    lo, hi = np.full(len(z), -6.0), np.full(len(z), 6.0)
    ent = lambda lt: norm_entropy(softmax(z / np.exp(lt)[:, None]))
    out_lo, out_hi = target < ent(lo), target > ent(hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        up = ent(mid) < target
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return softmax(z / np.exp((lo + hi) / 2)[:, None]), int(out_lo.sum() + out_hi.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    parser.add_argument("--out", type=Path, default=Path("results/structure_entropy_swap"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    rows = []
    for k in range(N_OUTER):
        z = {a: read(args.runs / f"distill_{a}_f{k}_s0" / "scores.jsonl") for a in ("cont", "rank_teacher")}
        t = pd.read_csv(args.teacher / f"teacher_v_f{k}.csv").query("role == 'heldout'").set_index("uid")[LABELS]
        uids = [u for u in z["cont"].index if u in z["rank_teacher"].index and u in t.index and u in df.index
                and df.loc[u, "fold"] == k and df.loc[u, "v_n"] > 0]
        d = df.loc[uids]
        P = counts(d, "v")
        P = P / P.sum(1, keepdims=True)
        hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
        S = {"cont (T=1)": softmax(z["cont"].loc[uids].to_numpy(float))}
        for a in ("cont", "rank_teacher"):
            Z = z[a].loc[uids].to_numpy(float)
            cal = np.zeros_like(P)
            for j in range(N_HEAD):
                te, tr = hf == j, hf != j
                tt = float(GRID[int(np.argmin([jsd(softmax(Z[tr], x), P[tr]).mean() for x in GRID]))])
                cal[te] = softmax(Z[te], tt)
            S[f"{a} (calibrated)"] = cal
        for i, u in enumerate(uids):
            rows.append({"uid": u, "fold": k, "actor": int(d.actor.iloc[i]), "emotion": d.intended.iloc[i], "p": P[i],
                         "teacher": t.loc[u].to_numpy(float), **{n: q[i] for n, q in S.items()}})
    R = pd.DataFrame(rows)
    Pv, T = np.stack(R.p), np.stack(R.teacher)
    hp, emo, act = norm_entropy(Pv), R.emotion.to_numpy(), R.actor.to_numpy()
    top = Pv == Pv.max(1, keepdims=True)
    boot = Bootstrap(act, N_BOOT)
    logT = np.log(np.clip(T, 1e-12, None))
    res, pair_rows, infeas = [], [], []
    for sname in ("cont (calibrated)", "cont (T=1)", "rank_teacher (calibrated)"):
        Sq = np.stack(R[sname])
        logS = np.log(np.clip(Sq, 1e-12, None))
        tS, nA = at_entropy(logT, norm_entropy(Sq))
        sT, nB = at_entropy(logS, norm_entropy(T))
        infeas.append({"student": sname, "T-structure, S-entropy: target out of reach": nA, "S-structure, T-entropy: target out of reach": nB, "clips": len(R)})
        preds = {"S": Sq, "T": T, "T-structure, S-entropy": tS, "S-structure, T-entropy": sT}
        for n, q in preds.items():
            j, hq, hit = jsd(q, Pv), norm_entropy(q), top[np.arange(len(q)), q.argmax(1)]
            res.append({"student": sname, "prediction": n, "JSD V": boot(lambda i: j[i].mean()), "top label among raters' top": boot(lambda i: hit[i].mean()),
                        "partial V": boot(lambda i: partial_spearman(hq[i], hp[i], emo[i])), "mean H(q)": round(float(hq.mean()), 4)})
        for a, b, role in (("T-structure, S-entropy", "S", "teacher structure at the student's entropy minus student"),
                           ("T", "S-structure, T-entropy", "teacher minus student structure at the teacher's entropy"),
                           ("S-structure, T-entropy", "S", "student structure at the teacher's entropy minus student"),
                           ("T", "T-structure, S-entropy", "teacher minus teacher structure at the student's entropy")):
            ja, jb = jsd(preds[a], Pv), jsd(preds[b], Pv)
            ta, tb = top[np.arange(len(R)), preds[a].argmax(1)], top[np.arange(len(R)), preds[b].argmax(1)]
            pair_rows.append({"student": sname, "comparison": role, "JSD V": boot(lambda i: ja[i].mean() - jb[i].mean()),
                              "top label": boot(lambda i: ta[i].mean() - tb[i].mean())})
        if sname == "cont (calibrated)":
            # groups fixed by the top labels of the two models (descriptive, not a causal decomposition)
            same = T.argmax(1) == Sq.argmax(1)
            ja, jb = jsd(tS, Pv), jsd(Sq, Pv)
            grp = []
            for gname, gsel in (("same top label (teacher = student)", same), ("different top label", ~same)):
                sel = np.flatnonzero(gsel)
                be = Bootstrap(act[sel], N_BOOT)
                ts, ss = top[sel, T[sel].argmax(1)], top[sel, Sq[sel].argmax(1)]
                grp.append({"group": gname, "clips": len(sel), "share": round(len(sel) / len(R), 3),
                            "JSD: T-structure at S-entropy minus S": be(lambda i: ja[sel][i].mean() - jb[sel][i].mean()),
                            "student JSD in group": round(float(jb[sel].mean()), 4),
                            "teacher top among raters' top": round(float(ts.mean()), 3), "student top among raters' top": round(float(ss.mean()), 3)})
            by_top = pd.DataFrame(grp)
            er = []
            ja, jb = jsd(tS, Pv), jsd(Sq, Pv)
            for e in sorted(set(emo)):
                sel = np.flatnonzero(emo == e)
                be = Bootstrap(act[sel], N_BOOT)
                er.append({"intended emotion": e, "clips": len(sel), "JSD: T-structure at S-entropy minus S": be(lambda i: ja[sel][i].mean() - jb[sel][i].mean())})
            by_emotion = pd.DataFrame(er)
    tables = {"predictions": pd.DataFrame(res), "paired": pd.DataFrame(pair_rows), "by_emotion_cont_calibrated": by_emotion, "by_top_label_agreement_cont_calibrated": by_top,
              "out_of_reach": pd.DataFrame(infeas)}
    lines = ["# Structure vs entropy swap (diagnostic, existing predictions)\n",
             "Generated by `analysis/structure_entropy_swap.py`; definitions and limits in its docstring.\n"]
    for name, tb in tables.items():
        tb.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(tb.astype(object).where(tb.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
