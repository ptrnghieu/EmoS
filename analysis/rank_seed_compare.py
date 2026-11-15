"""Ranking-transfer pilot: replication with a second training seed (agreed 14/11/2026, before the seed-1 runs).

Arms cont and rank_teacher, continued one epoch from the same s1_noI checkpoints (seed 0) with training seed 0 (the
pilot) and seed 1 (new clip order and dropout; both arms of a seed share the order). Calibration as
analysis/rank_compare.py (one global temperature per model and condition, chosen by JSD on two head folds of the
held-out actors, applied to the third). Main quantity: partial V after calibration, rank_teacher minus cont, for
each seed and as the mean of the two seeds' differences (bootstrap over actors, the same resample for both seeds).
Also JSD V and the AV metrics with the margins of the pilot (JSD < +0.002, partial AV > -0.02), and per fold.
This checks the stability of the continued-training phase, not of the whole pipeline (one s1_noI checkpoint).

Usage (from the repository root):
    python analysis/rank_seed_compare.py --runs results/distill --out results/rank_seed_compare
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

ARMS, SEEDS = ("cont", "rank_teacher"), (0, 1)
KEY = {"AV": "av", "V": "v"}


def read(path: Path, cond: str) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == cond) & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    z = s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
    return z.sub(z.mean(1), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--out", type=Path, default=Path("results/rank_seed_compare"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    for a in ARMS:
        for s in SEEDS:
            for k in range(N_OUTER):
                st = json.loads((args.runs / f"distill_{a}_f{k}_s{s}" / "state.json").read_text())
                assert st["train_done"] and (st["arm"], st["fold"], st["seed"]) == (a, k, s) and st["start"] == f"stage1_s1_noI_f{k}_s0", st
    tables, rows_all = {}, []
    for c in ("V", "AV"):
        rows = []
        for k in range(N_OUTER):
            z = {(a, s): read(args.runs / f"distill_{a}_f{k}_s{s}" / "scores.jsonl", c) for a in ARMS for s in SEEDS}
            uids = [u for u in z[("cont", 0)].index if all(u in v.index for v in z.values()) and u in df.index
                    and df.loc[u, "fold"] == k and df.loc[u, f"{KEY[c]}_n"] > 0]
            d = df.loc[uids]
            P = counts(d, KEY[c])
            P = P / P.sum(1, keepdims=True)
            hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
            pred = {}
            for (a, s), zz in z.items():
                Z = zz.loc[uids].to_numpy(float)
                cal = np.zeros_like(P)
                for j in range(N_HEAD):
                    te, tr = hf == j, hf != j
                    t = float(GRID[int(np.argmin([jsd(softmax(Z[tr], x), P[tr]).mean() for x in GRID]))])
                    cal[te] = softmax(Z[te], t)
                pred[f"{a} s{s}"] = cal
            for i, u in enumerate(uids):
                rows.append({"uid": u, "fold": k, "actor": int(d.actor.iloc[i]), "emotion": d.intended.iloc[i], "p": P[i], **{n: q[i] for n, q in pred.items()}})
        R = pd.DataFrame(rows)
        Pv = np.stack(R.p)
        hp, emo, act, fo = norm_entropy(Pv), R.emotion.to_numpy(), R.actor.to_numpy(), R.fold.to_numpy()
        boot = Bootstrap(act, N_BOOT)
        Q = {n: np.stack(R[n]) for n in R.columns if " s" in n}
        J = {n: jsd(q, Pv) for n, q in Q.items()}
        H = {n: norm_entropy(q) for n, q in Q.items()}
        dj = lambda s: (lambda i: J[f"rank_teacher s{s}"][i].mean() - J[f"cont s{s}"][i].mean())
        dp = lambda s: (lambda i: partial_spearman(H[f"rank_teacher s{s}"][i], hp[i], emo[i]) - partial_spearman(H[f"cont s{s}"][i], hp[i], emo[i]))
        for name, f in (("seed 0", 0), ("seed 1", 1), ("mean of the two seeds", None)):
            if f is None:
                fj = lambda i: (dj(0)(i) + dj(1)(i)) / 2
                fp = lambda i: (dp(0)(i) + dp(1)(i)) / 2
            else:
                fj, fp = dj(f), dp(f)
            sj = np.array([fj(i) for i in boot.samples])
            sp = np.array([fp(i) for i in boot.samples])
            pj, pp = fj(boot.all), fp(boot.all)
            rec = {"condition": c, "seed": name, "JSD (rank_teacher - cont)": fmt(pj, sj), "partial (rank_teacher - cont)": fmt(pp, sp)}
            jhi = np.nanpercentile(sj, 97.5)
            plo = np.nanpercentile(sp, 2.5)
            if c == "V":
                rec["partial V clearly higher"] = bool(plo > 0)
                rec["JSD V margin (+0.002)"] = "meets" if jhi < 0.002 else "not shown"
            else:
                rec["JSD AV margin (+0.002)"] = "meets" if jhi < 0.002 else "not shown"
                rec["partial AV margin (-0.02)"] = "meets" if plo > -0.02 else "not shown"
            for k in range(N_OUTER):
                sel = np.flatnonzero(fo == k)
                if f is not None:
                    rec[f"partial fold {k}"] = round(float(partial_spearman(H[f"rank_teacher s{f}"][sel], hp[sel], emo[sel]) - partial_spearman(H[f"cont s{f}"][sel], hp[sel], emo[sel])), 4)
            rows_all.append(rec)
        # how similar are the two seeds of the same arm
        if c == "V":
            tables["seed_agreement_V"] = pd.DataFrame([{"arm": a, "JSD seed 0": round(float(J[f'{a} s0'].mean()), 4), "JSD seed 1": round(float(J[f'{a} s1'].mean()), 4),
                                                        "partial seed 0": round(partial_spearman(H[f'{a} s0'], hp, emo), 4), "partial seed 1": round(partial_spearman(H[f'{a} s1'], hp, emo), 4),
                                                        "corr of clip entropies across seeds": round(float(np.corrcoef(H[f'{a} s0'], H[f'{a} s1'])[0, 1]), 4)} for a in ARMS])
    tables["rank_teacher_minus_cont_calibrated"] = pd.DataFrame(rows_all)
    lines = ["# Ranking-transfer pilot: second training seed (cont and rank_teacher)\n",
             "Generated by `analysis/rank_seed_compare.py`; protocol in its docstring.\n"]
    for name, t in tables.items():
        t.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(t.astype(object).where(t.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
