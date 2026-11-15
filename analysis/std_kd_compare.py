"""Logit-standardised distillation baseline (std_kd): reading fixed before the runs (15/11/2026).

Adapted from analysis/rank_compare.py (same calibration protocol and margins). Arms: cont, distill, rank_teacher and
std_kd (cont + 0.1 * tau^2 * KL between the softmaxes of the standardised teacher and student V logits, tau = 1).
Main comparison of this round (registered before the runs): JSD V after calibration, std_kd minus cont.
    JSD V clearly lower, partial V clearly higher than cont, AV margins met -> progress on both distribution and ranking
    JSD V clearly lower, partial V not clearly higher                      -> structure transfer helps; no further
                                                                              progress on the ranking target
    JSD V lower but partial V clearly lower                                -> trade-off; not the joint goal
    JSD V not clearly lower                                                -> this configuration did not show a transfer
                                                                              of the structural advantage
To say that it improves on the current ranking method, std_kd must be compared directly with rank_teacher; a CI that
contains 0 on partial does not show preservation.

Original docstring of the ranking pilot follows for the shared protocol.

Ranking-transfer pilot: reading fixed before the runs (13/11/2026).

Arms (results/distill/distill_<arm>_f<k>_s0, kaggle/distill_cremad.ipynb, all continued one epoch from s1_noI of fold
k with 1/2 CE_AV + 1/2 CE_V on the human votes): cont; distill (V target 2/3 votes + 1/3 teacher); rank_teacher and
rank_votes (cont + 0.1 * pairwise ranking loss on the entropy of the V output, pairs shared, sign from the teacher's
or from the human V-vote entropy).

Calibration, the same protocol for every arm and condition (as analysis/distill_calibration.py): within the held-out
actors of each outer fold, 3 head folds by actor; one global temperature per model and condition chosen by mean JSD
on the other two head folds (grid in [0.5, 2] containing 1), applied to the third. Results are reported before and
after calibration, per fold and per emotion; the A reading (not trained) is reported uncalibrated as a drift check.

Main metric of this round (changed from the earlier JSD-first rule, before the runs): partial V after calibration.
Non-inferiority, after calibration, rank_teacher minus cont: JSD V upper bound < +0.002; AV: JSD upper bound < +0.002
and partial lower bound > -0.02.
    rank_teacher better than cont on partial V and meeting the margins -> the pilot improves the V ranking within the
                                                                          trade-off limits set
    and also better than rank_votes                                     -> further evidence for the teacher's ranking
    better than cont, not better than rank_votes                        -> no advantage of the teacher over a ranking
                                                                          from the human votes shown (read
                                                                          rank_votes minus cont before saying more)
    not better than cont                                                -> no benefit of this configuration found
Not meeting a margin does not by itself show a degradation. Against distill the question is whether the trade-off
between JSD V and partial V is improved.

Usage (from the repository root):
    python analysis/std_kd_compare.py --runs results/distill --out results/std_kd_compare
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
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman

ARMS = ("cont", "distill", "rank_teacher", "std_kd")
KEY = {"AV": "av", "V": "v", "A": "a"}
M_JSD, M_PARTIAL = 0.002, -0.02


def read(path: Path, cond: str) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == cond) & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    z = s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
    return z.sub(z.mean(1), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--out", type=Path, default=Path("results/std_kd_compare"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [k for k in range(N_OUTER) if all((args.runs / f"distill_{a}_f{k}_s0" / "scores.jsonl").exists() for a in ARMS)]
    assert folds, "no fold with all four arms"
    status = "three folds pooled" if folds == list(range(N_OUTER)) else f"folds {folds} only: not the locked result"
    for a in ARMS:
        for k in folds:
            st = json.loads((args.runs / f"distill_{a}_f{k}_s0" / "state.json").read_text())
            assert st["train_done"] and st["arm"] == a and st["fold"] == k and st["start"] == f"stage1_s1_noI_f{k}_s0", st
            if a.startswith("rank"):
                assert abs(st.get("lambda_rank", 0) - 0.1) < 1e-12, st
            if a == "std_kd":
                assert abs(st.get("lambda_kd", 0) - 0.1) < 1e-12 and st.get("kd_tau") == 1.0, st
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    Q, meta, temps = {}, {}, []
    for c in ("V", "AV", "A"):
        rows = []
        for k in folds:
            z = {a: read(args.runs / f"distill_{a}_f{k}_s0" / "scores.jsonl", c) for a in ARMS}
            uids = [u for u in z["cont"].index if all(u in z[a].index for a in ARMS) and u in df.index
                    and df.loc[u, "fold"] == k and df.loc[u, f"{KEY[c]}_n"] > 0]
            d = df.loc[uids]
            P = counts(d, KEY[c])
            P = P / P.sum(1, keepdims=True)
            hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
            pred = {}
            for a in ARMS:
                Z = z[a].loc[uids].to_numpy(float)
                pred[f"{a} (T=1)"] = softmax(Z)
                if c == "A":
                    continue
                cal = np.zeros_like(P)
                for j in range(N_HEAD):
                    te, tr = hf == j, hf != j
                    t = float(GRID[int(np.argmin([jsd(softmax(Z[tr], t), P[tr]).mean() for t in GRID]))])
                    cal[te] = softmax(Z[te], t)
                    temps.append({"condition": c, "arm": a, "fold": k, "head fold": j, "temperature": t})
                pred[f"{a} (calibrated)"] = cal
            for i, u in enumerate(uids):
                rows.append({"uid": u, "fold": k, "actor": int(d.actor.iloc[i]), "emotion": d.intended.iloc[i], "p": P[i],
                             **{n: q[i] for n, q in pred.items()}})
        R = pd.DataFrame(rows)
        Pv = np.stack(R.p)
        meta[c] = {"P": Pv, "hp": norm_entropy(Pv), "emo": R.emotion.to_numpy(), "act": R.actor.to_numpy(), "fold": R.fold.to_numpy(),
                   "boot": Bootstrap(R.actor.to_numpy(), N_BOOT)}
        Q[c] = {n: np.stack(R[n]) for n in R.columns if n.endswith(")")}

    def met(c, q):
        m = meta[c]
        j, hq = jsd(q, m["P"]), norm_entropy(q)
        return {"JSD": lambda i: j[i].mean(), "partial": lambda i: partial_spearman(hq[i], m["hp"][i], m["emo"][i]),
                "Spearman(H(q), A)": lambda i: spearman(hq[i], m["hp"][i]), "mean H(q)": lambda i: hq[i].mean()}

    tables = {"readings": pd.DataFrame([{"condition": c, "predictor": n, **{k: meta[c]["boot"](f) for k, f in met(c, q).items()},
                                         **{f"JSD fold {k}": round(float(jsd(q, meta[c]["P"])[meta[c]["fold"] == k].mean()), 4) for k in folds}}
                                        for c in ("V", "AV", "A") for n, q in Q[c].items()])}
    st, prow = {}, []
    for c in ("V", "AV", "A"):
        ver = ("calibrated", "T=1") if c != "A" else ("T=1",)
        for v in ver:
            for a, b in (("std_kd", "cont"), ("std_kd", "rank_teacher"), ("std_kd", "distill"), ("rank_teacher", "cont"), ("distill", "cont")):
                na, nb = f"{a} ({v})", f"{b} ({v})"
                ma, mb = met(c, Q[c][na]), met(c, Q[c][nb])
                rec = {"condition": c, "version": v, "comparison": f"{a} minus {b}"}
                for m in ma:
                    s = np.array([ma[m](i) - mb[m](i) for i in meta[c]["boot"].samples])
                    p = ma[m](meta[c]["boot"].all) - mb[m](meta[c]["boot"].all)
                    lo, hi = np.nanpercentile(s, [2.5, 97.5])
                    rec[m] = fmt(p, s)
                    st[(c, v, a, b, m)] = (p, lo, hi)
                rec.update({f"JSD fold {k}": round(float(jsd(Q[c][na], meta[c]["P"])[meta[c]["fold"] == k].mean()
                                                         - jsd(Q[c][nb], meta[c]["P"])[meta[c]["fold"] == k].mean()), 4) for k in folds})
                prow.append(rec)
    tables["paired"] = pd.DataFrame(prow)

    g = lambda c, a, b, m: st[(c, "calibrated", a, b, m)]
    fmt3 = lambda t: "{:.4f} [{:.4f}, {:.4f}]".format(*t)
    _, lj, hj = g("V", "std_kd", "cont", "JSD")
    _, lp, hpv = g("V", "std_kd", "cont", "partial")
    ni = {"JSD AV": "meets" if g("AV", "std_kd", "cont", "JSD")[2] < M_JSD else "beyond" if g("AV", "std_kd", "cont", "JSD")[1] > M_JSD else "not shown",
          "partial AV": "meets" if g("AV", "std_kd", "cont", "partial")[1] > M_PARTIAL else "beyond" if g("AV", "std_kd", "cont", "partial")[2] < M_PARTIAL else "not shown"}
    if hj >= 0:
        overall = "JSD V not clearly lower than cont: this configuration did not show a transfer of the structural advantage"
    elif hpv < 0:
        overall = "JSD V lower but partial V clearly lower: trade-off, not the joint goal"
    elif lp > 0 and all(v == "meets" for v in ni.values()):
        overall = "JSD V clearly lower and partial V clearly higher than cont, AV margins met: progress on both distribution and ranking"
    else:
        overall = "JSD V clearly lower; partial V not clearly higher (or an AV margin not shown): structure transfer helps, no further progress on ranking shown"
    tables["reading"] = pd.DataFrame([{
        "JSD V: std_kd - cont (main)": fmt3(g("V", "std_kd", "cont", "JSD")), "partial V: std_kd - cont": fmt3(g("V", "std_kd", "cont", "partial")),
        "JSD V: std_kd - rank_teacher": fmt3(g("V", "std_kd", "rank_teacher", "JSD")), "partial V: std_kd - rank_teacher": fmt3(g("V", "std_kd", "rank_teacher", "partial")),
        "JSD V: std_kd - distill": fmt3(g("V", "std_kd", "distill", "JSD")), "partial V: std_kd - distill": fmt3(g("V", "std_kd", "distill", "partial")),
        "JSD AV": fmt3(g("AV", "std_kd", "cont", "JSD")), "JSD AV margin": ni["JSD AV"],
        "partial AV": fmt3(g("AV", "std_kd", "cont", "partial")), "partial AV margin": ni["partial AV"],
        "overall": overall, "status": status}]).T.reset_index().rename(columns={"index": "item", 0: "value"})

    erows = []
    for c in ("V", "AV"):
        m = meta[c]
        for a, b in (("std_kd", "cont"), ("std_kd", "rank_teacher")):
            hn, hr = norm_entropy(Q[c][f"{a} (calibrated)"]), norm_entropy(Q[c][f"{b} (calibrated)"])
            jn, jr = jsd(Q[c][f"{a} (calibrated)"], m["P"]), jsd(Q[c][f"{b} (calibrated)"], m["P"])
            for e in sorted(set(m["emo"])):
                sel = np.flatnonzero(m["emo"] == e)
                be = Bootstrap(m["act"][sel], N_BOOT)
                erows.append({"condition": c, "comparison": f"{a} minus {b} (calibrated)", "intended emotion": e, "clips": len(sel),
                              "JSD difference": be(lambda i: jn[sel][i].mean() - jr[sel][i].mean()),
                              "within-class Spearman difference": be(lambda i: spearman(hn[sel][i], m["hp"][sel][i]) - spearman(hr[sel][i], m["hp"][sel][i]))})
    tables["by_emotion_paired"] = pd.DataFrame(erows)
    tables["temperatures"] = pd.DataFrame(temps).groupby(["condition", "arm"])["temperature"].agg(["mean", "min", "max"]).round(3).reset_index()
    lines = ["# Logit-standardised distillation (std_kd) vs continued training, distillation and the ranking method\n",
             f"Generated by `analysis/std_kd_compare.py` ({status}). Protocol and reading rules in its docstring (fixed before the runs).\n"]
    for name, t in tables.items():
        t.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(t.astype(object).where(t.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
