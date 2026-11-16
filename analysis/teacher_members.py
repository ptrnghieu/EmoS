"""Single cross-fitted teachers vs their five-teacher mean on the held-out actors (16/11/2026; std_kd follow-up).

The structure-swap diagnostic used the mean of the five cross-fitted HSE-only teachers on the held-out actors, while
every training clip of std_kd / distill / rank_teacher received the prediction of one teacher (the one that did not
see its actor). All five teachers of an outer fold never saw its held-out actors, so each can be evaluated there.
Reports, against the V votes: JSD, top label among the raters' top labels, partial Spearman (given the intended
emotion) and mean normalised entropy, for each teacher, the mean of the five single-teacher scores, and the
five-teacher mean prediction; the paired difference ensemble minus mean single teacher (bootstrap over actors); and
the structure swap (teacher structure at the calibrated cont student's entropy minus the student) with a single
teacher instead of the mean.
Teacher files: results/distill_teacher/teacher_v_members_f<k>.csv (analysis/distill_teacher.py --members; the rerun
reproduces the saved teacher files exactly).

Usage (from the repository root):
    python analysis/teacher_members.py --out results/teacher_members
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
from structure_entropy_swap import at_entropy, read
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    parser.add_argument("--out", type=Path, default=Path("results/teacher_members"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    rows = []
    for k in range(N_OUTER):
        m = pd.read_csv(args.teacher / f"teacher_v_members_f{k}.csv")
        M = {j: g.set_index("uid")[LABELS] for j, g in m.groupby("member")}
        zc = read(args.runs / f"distill_cont_f{k}_s0" / "scores.jsonl")
        uids = [u for u in zc.index if all(u in M[j].index for j in M) and u in df.index and df.loc[u, "fold"] == k and df.loc[u, "v_n"] > 0]
        d = df.loc[uids]
        P = counts(d, "v")
        P = P / P.sum(1, keepdims=True)
        hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
        Z = zc.loc[uids].to_numpy(float)
        cal = np.zeros_like(P)
        for j in range(N_HEAD):
            te, tr = hf == j, hf != j
            t = float(GRID[int(np.argmin([jsd(softmax(Z[tr], x), P[tr]).mean() for x in GRID]))])
            cal[te] = softmax(Z[te], t)
        for i, u in enumerate(uids):
            rows.append({"uid": u, "fold": k, "actor": int(d.actor.iloc[i]), "emotion": d.intended.iloc[i], "p": P[i], "cont": cal[i],
                         **{f"teacher {j}": M[j].loc[u].to_numpy(float) for j in M}})
    R = pd.DataFrame(rows)
    Pv = np.stack(R.p)
    hp, emo, act = norm_entropy(Pv), R.emotion.to_numpy(), R.actor.to_numpy()
    top = Pv == Pv.max(1, keepdims=True)
    boot = Bootstrap(act, N_BOOT)
    T = {j: np.stack(R[f"teacher {j}"]) for j in range(5)}
    ens = np.mean([T[j] for j in range(5)], 0)
    S = np.stack(R.cont)

    def met(q):
        j, hq, hit = jsd(q, Pv), norm_entropy(q), top[np.arange(len(q)), q.argmax(1)]
        return {"JSD V": lambda i: j[i].mean(), "top label": lambda i: hit[i].mean(),
                "partial V": lambda i: partial_spearman(hq[i], hp[i], emo[i]), "mean H(q)": lambda i: hq[i].mean()}

    mt = {j: met(T[j]) for j in range(5)}
    me = met(ens)
    rows1 = [{"predictor": f"single teacher {j}", **{n: round(float(f(boot.all)), 4) for n, f in mt[j].items()}} for j in range(5)]
    rows1.append({"predictor": "mean of the five single-teacher scores", **{n: boot(lambda i, n=n: np.mean([mt[j][n](i) for j in range(5)])) for n in me}})
    rows1.append({"predictor": "five-teacher mean prediction (ensemble)", **{n: boot(f) for n, f in me.items()}})
    rows1.append({"predictor": "ensemble minus mean single teacher", **{n: boot(lambda i, n=n: me[n](i) - np.mean([mt[j][n](i) for j in range(5)])) for n in me}})
    rows1.append({"predictor": "cont (calibrated), reference", **{n: boot(f) for n, f in met(S).items()}})
    tables = {"single_vs_ensemble": pd.DataFrame(rows1)}

    logS = np.log(np.clip(S, 1e-12, None))
    hS = norm_entropy(S)
    rows2 = []
    js = jsd(S, Pv)
    swaps = {}
    for name, q in [(f"single teacher {j}", T[j]) for j in range(5)] + [("ensemble", ens)]:
        sw, nout = at_entropy(np.log(np.clip(q, 1e-12, None)), hS)
        swaps[name] = jsd(sw, Pv)
        rows2.append({"teacher": name, "JSD: teacher structure at the student's entropy minus student": boot(lambda i, a=swaps[name]: a[i].mean() - js[i].mean()),
                      "target out of reach": nout})
    single_mean = np.mean([swaps[f"single teacher {j}"] for j in range(5)], 0)
    rows2.append({"teacher": "mean over the five single-teacher swaps", "JSD: teacher structure at the student's entropy minus student": boot(lambda i: single_mean[i].mean() - js[i].mean()), "target out of reach": ""})
    tables["structure_swap_single_vs_ensemble"] = pd.DataFrame(rows2)
    lines = ["# Single cross-fitted teachers vs their five-teacher mean (held-out actors)\n",
             "Generated by `analysis/teacher_members.py`; definitions in its docstring.\n"]
    for name, tb in tables.items():
        tb.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(tb.astype(object).where(tb.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
