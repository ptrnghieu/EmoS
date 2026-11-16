"""Evaluation of the repeated cross-fitting teachers on the outer held-out actors (locked 16/11/2026, before running).

Sources (results/distill_teacher_ens/members_f<k>.csv; 3 partitions x 5 parts = 15 heads per outer fold, none of
which saw the held-out actors):
  single teacher            each of the 15 heads; metrics averaged over the 15
  three-teacher ensemble    the mean prediction of one head from each partition, the source a training clip receives;
                            metrics computed for every one of the 125 combinations and averaged (not the mean
                            prediction of all combinations)
  15-teacher ensemble       the mean of all heads: a reference for the teacher system, not a training-equivalent source
For an ensemble two entropies are reported: H(mean prediction) and the mean of the single-head entropies.
Metrics against the V votes: JSD, top label among the raters' top labels, partial Spearman of each entropy (given
the intended emotion), mean normalised entropy; and the structure swap (the source's structure at the calibrated cont
student's entropy minus the student). Paired differences with bootstrap over actors (the same resample for all
combinations). Ranking stability: within-emotion Spearman between the entropies of two sources that share no head
(single: heads of different partitions; three-teacher: combinations (j, j, j) and (j', j', j'), j != j'), averaged.

Decision table (fixed before running): structure at equal entropy clearly better -> repeat std_kd with the new target
(lambda, tau unchanged); ranking clearly better but structure not -> rank_teacher with the new source; neither ->
no basis for GPU on the ensemble. "Clearly better" = three-teacher ensemble minus single teacher, 95% interval
excluding 0 in the favourable direction.

Usage (from the repository root):
    python analysis/teacher_ensemble_eval.py --out results/teacher_ensemble_eval
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from av_complement_head import N_BOOT, N_HEAD, head_folds
from cremad_human import counts, jsd
from distill_calibration import GRID
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, fmt, softmax
from rq_analysis import markdown
from structure_entropy_swap import at_entropy, read
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman

N_CF, N_REP = 5, 3
COMBOS = list(itertools.product(range(N_CF), repeat=N_REP))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teachers", type=Path, default=Path("results/distill_teacher_ens"))
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--out", type=Path, default=Path("results/teacher_ensemble_eval"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    parts = []
    for k in range(N_OUTER):
        m = pd.read_csv(args.teachers / f"members_f{k}.csv").query("role == 'heldout'")
        H = {(r, j): g.set_index("uid")[LABELS] for (r, j), g in m.groupby(["partition", "part"])}
        zc = read(args.runs / f"distill_cont_f{k}_s0" / "scores.jsonl")
        uids = [u for u in zc.index if all(u in h.index for h in H.values()) and u in df.index and df.loc[u, "fold"] == k and df.loc[u, "v_n"] > 0]
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
        parts.append({"P": P, "emo": d.intended.to_numpy(), "act": d.actor.to_numpy(), "S": cal,
                      "H": {key: h.loc[uids].to_numpy(float) for key, h in H.items()}})
    Pv = np.concatenate([p["P"] for p in parts])
    emo = np.concatenate([p["emo"] for p in parts])
    act = np.concatenate([p["act"] for p in parts])
    fold_id = np.concatenate([np.full(len(p["P"]), k) for k, p in enumerate(parts)])
    S = np.concatenate([p["S"] for p in parts])
    head = {key: np.concatenate([p["H"][key] for p in parts]) for key in parts[0]["H"]}
    hp, top = norm_entropy(Pv), Pv == Pv.max(1, keepdims=True)
    hS, logS, jS = norm_entropy(S), None, jsd(S, Pv)
    boot = Bootstrap(act, N_BOOT)
    samples = boot.samples

    def per_source(q, h_alt=None):
        j, hq, hit = jsd(q, Pv), norm_entropy(q), top[np.arange(len(q)), q.argmax(1)]
        sw, _ = at_entropy(np.log(np.clip(q, 1e-12, None)), hS)
        jsw = jsd(sw, Pv)
        return {"JSD V": j, "top": hit, "H": hq, "H_alt": hq if h_alt is None else h_alt, "swap": jsw}

    single = [per_source(head[key]) for key in head]
    ens3 = []
    for c in COMBOS:
        qs = [head[(r, c[r])] for r in range(N_REP)]
        ens3.append(per_source(np.mean(qs, 0), np.mean([norm_entropy(q) for q in qs], 0)))
    allq = list(head.values())
    full = per_source(np.mean(allq, 0), np.mean([norm_entropy(q) for q in allq], 0))

    def stat(srcs, name, idx):
        if name == "partial H(mean)":
            return np.mean([partial_spearman(s["H"][idx], hp[idx], emo[idx]) for s in srcs])
        if name == "partial mean H":
            return np.mean([partial_spearman(s["H_alt"][idx], hp[idx], emo[idx]) for s in srcs])
        if name == "swap minus student":
            return np.mean([s["swap"][idx].mean() for s in srcs]) - jS[idx].mean()
        if name == "mean H":
            return np.mean([s["H"][idx].mean() for s in srcs])
        key = {"JSD V": "JSD V", "top label": "top"}[name]
        return np.mean([s[key][idx].mean() for s in srcs])

    metrics = ["JSD V", "top label", "partial H(mean)", "partial mean H", "mean H", "swap minus student"]
    groups = {"single teacher (mean over 15)": single, "three-teacher ensemble (mean over 125)": ens3, "15-teacher ensemble": [full]}
    allidx = np.arange(len(Pv))
    rows, diffs = [], []
    cache = {}
    for g, srcs in groups.items():
        rec = {"source": g}
        for mname in metrics:
            if g == "single teacher (mean over 15)" and mname == "partial mean H":
                rec[mname] = ""
                continue
            vals = np.array([stat(srcs, mname, i) for i in samples])
            cache[(g, mname)] = vals
            rec[mname] = fmt(stat(srcs, mname, allidx), vals)
        rows.append(rec)
    S_rec = {"source": "cont (calibrated), reference", "JSD V": round(float(jS.mean()), 4),
             "top label": round(float(top[np.arange(len(S)), S.argmax(1)].mean()), 4),
             "partial H(mean)": round(partial_spearman(hS, hp, emo), 4), "mean H": round(float(hS.mean()), 4)}
    rows.append(S_rec)
    for a, b in (("three-teacher ensemble (mean over 125)", "single teacher (mean over 15)"), ("15-teacher ensemble", "three-teacher ensemble (mean over 125)")):
        rec = {"comparison": f"{a} minus {b}"}
        for mname in metrics:
            if (a, mname) in cache and (b, mname) in cache:
                pt = stat(groups[a], mname, allidx) - stat(groups[b], mname, allidx)
                rec[mname] = fmt(pt, cache[(a, mname)] - cache[(b, mname)])
        diffs.append(rec)
    rec = {"comparison": "three-teacher ensemble: partial mean H minus partial H(mean)"}
    rec["partial"] = fmt(stat(ens3, "partial mean H", allidx) - stat(ens3, "partial H(mean)", allidx),
                         cache[("three-teacher ensemble (mean over 125)", "partial mean H")] - cache[("three-teacher ensemble (mean over 125)", "partial H(mean)")])
    diffs.append(rec)

    def within_emotion_spearman(a, b):
        out = []
        for k in range(N_OUTER):
            for e in np.unique(emo):
                sel = (fold_id == k) & (emo == e)
                out.append(spearmanr(a[sel], b[sel]).correlation)
        return float(np.mean(out))

    keys = list(head)
    s_pairs = [within_emotion_spearman(norm_entropy(head[x]), norm_entropy(head[y])) for x, y in itertools.combinations(keys, 2) if x[0] != y[0]]
    diag = [tuple([j] * N_REP) for j in range(N_CF)]
    ens_q = {c: np.mean([head[(r, c[r])] for r in range(N_REP)], 0) for c in diag}
    ens_h = {c: np.mean([norm_entropy(head[(r, c[r])]) for r in range(N_REP)], 0) for c in diag}
    e1 = [within_emotion_spearman(norm_entropy(ens_q[x]), norm_entropy(ens_q[y])) for x, y in itertools.combinations(diag, 2)]
    e2 = [within_emotion_spearman(ens_h[x], ens_h[y]) for x, y in itertools.combinations(diag, 2)]
    stab = pd.DataFrame([{"source": "single teacher (heads of different partitions)", "pairs": len(s_pairs), "mean within-emotion Spearman": round(np.mean(s_pairs), 4)},
                         {"source": "three-teacher ensemble, H(mean), disjoint combinations", "pairs": len(e1), "mean within-emotion Spearman": round(np.mean(e1), 4)},
                         {"source": "three-teacher ensemble, mean H, disjoint combinations", "pairs": len(e2), "mean within-emotion Spearman": round(np.mean(e2), 4)}])
    tables = {"sources": pd.DataFrame(rows), "paired": pd.DataFrame(diffs), "ranking_stability": stab}
    lines = ["# Repeated cross-fitting teachers on the held-out actors\n",
             "Generated by `analysis/teacher_ensemble_eval.py`; definitions and the decision table in its docstring.\n"]
    for name, tb in tables.items():
        tb.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(tb.astype(object).where(tb.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
