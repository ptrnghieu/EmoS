"""Ranking-transfer pilot: robustness and mechanism checks on the existing predictions (CPU, 14/11/2026, after the
locked result of analysis/rank_compare.py was seen; descriptive).

1. rank_teacher minus distill before calibration (paired, bootstrap over actors): JSD V, partial V, mean entropy.
2. Main comparison (rank_teacher minus cont, partial V and JSD V after calibration, protocol of rank_compare.py) per
   outer fold, per actor (JSD only: about 80 clips per actor are too few for a partial correlation) and leaving out
   one actor at a time (range of the pooled estimate).
3. Ordered pairs on a fixed pair set of the held-out actors: within each outer fold and intended emotion, percentiles
   (midranks) of the teacher entropy (mean of the five cross-fitted teachers on the held-out actors) and of the human
   V-vote entropy; a pair is valid when both percentile gaps are >= 0.3 (the training rule). For every checkpoint
   (s1_noI start, cont, distill, rank_teacher, rank_votes) the share of valid pairs whose V-output entropy (T = 1) is
   ordered as the teacher orders them, and as the votes order them. Bootstrap over actors (a pair is weighted by the
   product of its two actors' multiplicities).

Usage (from the repository root):
    python analysis/rank_diagnostics.py --out results/rank_diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from av_complement_head import N_BOOT, N_HEAD, head_folds
from cremad_human import counts, jsd
from distill_calibration import GRID
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, fmt, softmax
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman

ARMS = ("cont", "distill", "rank_teacher", "rank_votes")
MIN_DU = 0.3


def read(path: Path) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == "V") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    z = s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
    return z.sub(z.mean(1), axis=0)


def pct(x):
    return (rankdata(x, method="average") - 0.5) / len(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    parser.add_argument("--out", type=Path, default=Path("results/rank_diagnostics"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    rows = []
    for k in range(N_OUTER):
        z = {a: read(args.runs / f"distill_{a}_f{k}_s0" / "scores.jsonl") for a in ARMS}
        z["s1_noI"] = read(args.stage1 / f"stage1_s1_noI_f{k}_s0" / "scores.jsonl")
        t = pd.read_csv(args.teacher / f"teacher_v_f{k}.csv").query("role == 'heldout'").set_index("uid")[LABELS]
        uids = [u for u in z["cont"].index if all(u in z[a].index for a in z) and u in t.index and u in df.index
                and df.loc[u, "fold"] == k and df.loc[u, "v_n"] > 0]
        d = df.loc[uids]
        P = counts(d, "v")
        P = P / P.sum(1, keepdims=True)
        hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
        rec = {n: softmax(z[n].loc[uids].to_numpy(float)) for n in z}
        for a in ("cont", "rank_teacher"):
            Z = z[a].loc[uids].to_numpy(float)
            cal = np.zeros_like(P)
            for j in range(N_HEAD):
                te, tr = hf == j, hf != j
                tt = float(GRID[int(np.argmin([jsd(softmax(Z[tr], x), P[tr]).mean() for x in GRID]))])
                cal[te] = softmax(Z[te], tt)
            rec[f"{a} (calibrated)"] = cal
        for i, u in enumerate(uids):
            rows.append({"uid": u, "fold": k, "actor": int(d.actor.iloc[i]), "emotion": d.intended.iloc[i], "p": P[i],
                         "teacher": t.loc[u].to_numpy(float), **{n: q[i] for n, q in rec.items()}})
    R = pd.DataFrame(rows)
    Pv = np.stack(R.p)
    hp, emo, act, fo = norm_entropy(Pv), R.emotion.to_numpy(), R.actor.to_numpy(), R.fold.to_numpy()
    Q = {n: np.stack(R[n]) for n in R.columns if n not in ("uid", "fold", "actor", "emotion", "p")}
    boot = Bootstrap(act, N_BOOT)
    tables = {}

    # 1. rank_teacher vs distill before calibration
    def diff(a, b, sel=None):
        sel = np.arange(len(R)) if sel is None else sel
        ja, jb = jsd(Q[a][sel], Pv[sel]), jsd(Q[b][sel], Pv[sel])
        ha, hb = norm_entropy(Q[a][sel]), norm_entropy(Q[b][sel])
        bb = boot if len(sel) == len(R) else Bootstrap(act[sel], N_BOOT)
        return {"JSD V": bb(lambda i: ja[i].mean() - jb[i].mean()),
                "partial V": bb(lambda i: partial_spearman(ha[i], hp[sel][i], emo[sel][i]) - partial_spearman(hb[i], hp[sel][i], emo[sel][i])),
                "mean H(q)": bb(lambda i: ha[i].mean() - hb[i].mean())}

    tables["1_rank_teacher_vs_distill_uncalibrated"] = pd.DataFrame([{"comparison": "rank_teacher minus distill (T=1)", **diff("rank_teacher", "distill")},
                                                                    {"comparison": "rank_teacher minus cont (T=1)", **diff("rank_teacher", "cont")},
                                                                    {"comparison": "distill minus cont (T=1)", **diff("distill", "cont")}])
    # 2. per fold, per actor, leave one actor out (calibrated main comparison)
    a, b = "rank_teacher (calibrated)", "cont (calibrated)"
    tables["2a_per_fold"] = pd.DataFrame([{"outer fold": k, "clips": int((fo == k).sum()), **diff(a, b, np.flatnonzero(fo == k))} for k in range(N_OUTER)])
    ja, jb = jsd(Q[a], Pv), jsd(Q[b], Pv)
    ha, hb = norm_entropy(Q[a]), norm_entropy(Q[b])
    per_actor = pd.DataFrame({"actor": act, "d": ja - jb}).groupby("actor")["d"].agg(["mean", "size"]).reset_index()
    loo = []
    for x in np.unique(act):
        s = act != x
        loo.append({"left out": int(x), "partial V diff": partial_spearman(ha[s], hp[s], emo[s]) - partial_spearman(hb[s], hp[s], emo[s]),
                    "JSD V diff": float(ja[s].mean() - jb[s].mean())})
    loo = pd.DataFrame(loo)
    full_p = partial_spearman(ha, hp, emo) - partial_spearman(hb, hp, emo)
    tables["2b_actors"] = pd.DataFrame([{
        "actors": len(per_actor), "partial V diff (all actors)": round(full_p, 4),
        "leave-one-actor-out partial V diff: min / max": f"{loo['partial V diff'].min():.4f} / {loo['partial V diff'].max():.4f}",
        "actor whose removal moves partial most": int(loo.loc[(loo['partial V diff'] - full_p).abs().idxmax(), "left out"]),
        "JSD V diff per actor: share < 0": round(float((per_actor["mean"] < 0).mean()), 3),
        "JSD V diff per actor: min / median / max": f"{per_actor['mean'].min():.4f} / {per_actor['mean'].median():.4f} / {per_actor['mean'].max():.4f}",
        "leave-one-actor-out JSD V diff: min / max": f"{loo['JSD V diff'].min():.5f} / {loo['JSD V diff'].max():.5f}"}])
    loo.to_csv(args.out / "leave_one_actor_out.csv", index=False)
    per_actor.to_csv(args.out / "per_actor_jsd.csv", index=False)

    # 3. ordered pairs on a fixed held-out pair set
    hT = norm_entropy(np.stack(R.teacher))
    models = ["s1_noI", "cont", "distill", "rank_teacher", "rank_votes"]
    H = {m: norm_entropy(Q[m]) for m in models}
    I, J = [], []
    for k in range(N_OUTER):
        for e in np.unique(emo):
            idx = np.flatnonzero((fo == k) & (emo == e))
            uT, uH = pct(hT[idx]), pct(hp[idx])
            ii, jj = np.triu_indices(len(idx), 1)
            ok = (np.abs(uT[ii] - uT[jj]) >= MIN_DU) & (np.abs(uH[ii] - uH[jj]) >= MIN_DU)
            I.append(idx[ii[ok]])
            J.append(idx[jj[ok]])
    I, J = np.concatenate(I), np.concatenate(J)
    sT, sH = np.sign(hT[I] - hT[J]), np.sign(hp[I] - hp[J])
    agree = {(m, src): (np.sign(H[m][I] - H[m][J]) == s).astype(float) for m in models for src, s in (("teacher", sT), ("votes", sH))}
    actors_u = np.unique(act)
    pos = {x: n for n, x in enumerate(actors_u)}
    ai, aj = np.array([pos[x] for x in act[I]]), np.array([pos[x] for x in act[J]])
    rng = np.random.default_rng(0)
    W = []
    for _ in range(N_BOOT):
        c = np.bincount(rng.integers(0, len(actors_u), len(actors_u)), minlength=len(actors_u)).astype(float)
        W.append(c[ai] * c[aj])
    stat = lambda v: fmt(float(v.mean()), np.array([(w * v).sum() / w.sum() for w in W]))
    rows3 = []
    for src in ("teacher", "votes"):
        for m in models:
            rows3.append({"order source": src, "checkpoint": m, "share of pairs ordered as the source": stat(agree[(m, src)]),
                          "minus s1_noI": stat(agree[(m, src)] - agree[("s1_noI", src)]), "minus cont": stat(agree[(m, src)] - agree[("cont", src)])})
    tables["3_ordered_pairs"] = pd.DataFrame(rows3)
    tables["3_pair_set"] = pd.DataFrame([{"held-out valid pairs": len(I), "sources disagree (share)": round(float((sT != sH).mean()), 4)}])

    lines = ["# Ranking-transfer pilot: robustness and mechanism checks (descriptive, after the locked result)\n",
             "Generated by `analysis/rank_diagnostics.py`; definitions in its docstring.\n"]
    for name, tb in tables.items():
        tb.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(tb.astype(object).where(tb.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
