"""Step 0 (locked 16/11/2026): which part of the video-only error do better sources reduce?

For clip i with V votes p (N_i raters) and a prediction q, and a reference class y among the classes with the most
votes (ties: the decomposition is averaged over all tied classes, each clip keeps total weight one):
    E_ref = KL([p_y, 1 - p_y] || [q_y, 1 - q_y])
    E_alt = (1 - p_y) KL(p^- || q^-),  p^- = p_{-y} / (1 - p_y), q^- = q_{-y} / (1 - q_y)   (0 when p_y = 1)
    KL(p || q) = E_ref + E_alt   (checked numerically).
Main report: clip means of E_ref and E_alt; also the vote-weighted mean sum_i N_i E_alt,i / sum_i N_i (no further
weighting). For every better source, paired differences of both parts against the Qwen reference (bootstrap over
actors, the same resample for both parts); no share of the total is reported. A lower E_ref does not by itself mean
better top-label recognition: it also reflects the probability given to the reference class.

Sources (V reading, held-out actors of the three outer folds): Qwen cont (calibrated, main Qwen reference), s1_noI,
rank_teacher (calibrated), HSEmotion + MLP / linear head, the five-teacher mean of the cross-fitted HSE teachers.
Simulated human re-runs (sensitivity under assumptions, not a ceiling), 20 repetitions, reference class from the
observed votes: main, empirical Bayes: theta ~ Dirichlet(counts + alpha * pi), n' ~ Multinomial(N, theta), smoothed
group distribution q = (n' + alpha * pi) / (N + alpha) (posterior predictive under the fitted prior); sensitivity,
Jeffreys: theta ~ Dirichlet(counts + 0.5), q = (n' + 0.5) / (N + 3). They are "simulated groups with smoothing".

Usage (from the repository root):
    python analysis/kl_decomposition.py --out results/kl_decomposition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from av_complement_head import N_BOOT, N_HEAD, head_folds
from cremad_human import counts, fit_dirichlet, jsd
from distill_calibration import GRID
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, fmt, softmax
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, outer_folds

N_REP = 20


def kl_rows(p, q):
    p, q = np.clip(p, 0, None), np.clip(q, 1e-12, None)
    return np.where(p > 0, p * (np.log(np.where(p > 0, p, 1)) - np.log(q)), 0).sum(1)


def decompose(p, q):
    """Per clip E_ref, E_alt, averaged over tied reference classes; checks E_ref + E_alt = KL."""
    n, K = p.shape
    eref, ealt = np.zeros(n), np.zeros(n)
    tied = p >= p.max(1, keepdims=True) - 1e-12
    for y in range(K):
        sel = tied[:, y]
        if not sel.any():
            continue
        py, qy = p[sel, y], np.clip(q[sel, y], 1e-12, 1 - 1e-12)
        b = kl_rows(np.stack([py, 1 - py], 1), np.stack([qy, 1 - qy], 1))
        rest = [c for c in range(K) if c != y]
        pm = np.where((1 - py)[:, None] > 0, p[sel][:, rest] / np.clip(1 - py, 1e-12, None)[:, None], 0)
        qm = q[sel][:, rest] / (1 - qy)[:, None]
        a = (1 - py) * np.where(py < 1, kl_rows(pm, qm), 0)
        w = 1 / tied[sel].sum(1)
        eref[sel] += w * b
        ealt[sel] += w * a
    total = kl_rows(p, q)
    assert np.allclose(eref + ealt, total, atol=1e-8), np.abs(eref + ealt - total).max()
    return eref, ealt


def read(path: Path) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == "V") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    z = s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
    return z.sub(z.mean(1), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    parser.add_argument("--hse", type=Path, default=Path("results/hsemotion_head"))
    parser.add_argument("--out", type=Path, default=Path("results/kl_decomposition"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    hse = {n: read(args.hse / n / "scores.jsonl") for n in ("mlp", "linear")}
    rows = []
    for k in range(N_OUTER):
        z = {a: read(args.runs / f"distill_{a}_f{k}_s0" / "scores.jsonl") for a in ("cont", "rank_teacher")}
        z1 = read(args.stage1 / f"stage1_s1_noI_f{k}_s0" / "scores.jsonl")
        t = pd.read_csv(args.teacher / f"teacher_v_f{k}.csv").query("role == 'heldout'").set_index("uid")[LABELS]
        uids = [u for u in z["cont"].index if u in z["rank_teacher"].index and u in z1.index and u in t.index and all(u in h.index for h in hse.values())
                and u in df.index and df.loc[u, "fold"] == k and df.loc[u, "v_n"] > 0]
        d = df.loc[uids]
        C = counts(d, "v")
        P = C / C.sum(1, keepdims=True)
        hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
        src = {"s1_noI": softmax(z1.loc[uids].to_numpy(float)), "teacher (5-mean)": t.loc[uids].to_numpy(float),
               "HSEmotion + MLP head": softmax(hse["mlp"].loc[uids].to_numpy(float)), "HSEmotion + linear head": softmax(hse["linear"].loc[uids].to_numpy(float))}
        for a in ("cont", "rank_teacher"):
            Z = z[a].loc[uids].to_numpy(float)
            cal = np.zeros_like(P)
            for j in range(N_HEAD):
                te, tr = hf == j, hf != j
                tt = float(GRID[int(np.argmin([jsd(softmax(Z[tr], x), P[tr]).mean() for x in GRID]))])
                cal[te] = softmax(Z[te], tt)
            src[f"{a} (calibrated)"] = cal
        for i, u in enumerate(uids):
            rows.append({"uid": u, "actor": int(d.actor.iloc[i]), "emotion": d.intended.iloc[i], "C": C[i], **{n: q[i] for n, q in src.items()}})
    R = pd.DataFrame(rows)
    C = np.stack(R.C)
    N = C.sum(1)
    P = C / N[:, None]
    act, emo = R.actor.to_numpy(), R.emotion.to_numpy()
    boot = Bootstrap(act, N_BOOT)
    names = ["cont (calibrated)", "s1_noI", "rank_teacher (calibrated)", "HSEmotion + MLP head", "HSEmotion + linear head", "teacher (5-mean)"]
    E = {n: decompose(P, np.stack(R[n])) for n in names}
    # simulated human re-runs (smoothed groups)
    alpha, pi = fit_dirichlet(C)
    rng = np.random.default_rng(0)
    for label, prior, smooth in (("re-run, empirical Bayes (smoothed)", alpha * pi, lambda n_: (n_ + alpha * pi) / (N[:, None] + alpha)),
                                 ("re-run, Jeffreys (smoothed)", 0.5, lambda n_: (n_ + 0.5) / (N[:, None] + 3.0))):
        er, ea = [], []
        for _ in range(N_REP):
            g = rng.gamma(C + prior)
            n_ = rng.multinomial(N.astype(int), g / g.sum(1, keepdims=True)).astype(float)
            a_, b_ = decompose(P, smooth(n_))
            er.append(a_)
            ea.append(b_)
        E[label] = (np.mean(er, 0), np.mean(ea, 0))
    allnames = names + ["re-run, empirical Bayes (smoothed)", "re-run, Jeffreys (smoothed)"]
    tab = []
    for n in allnames:
        er, ea = E[n]
        tab.append({"source": n, "KL (clip mean)": boot(lambda i: (er[i] + ea[i]).mean()), "E_ref (clip mean)": boot(lambda i: er[i].mean()),
                    "E_alt (clip mean)": boot(lambda i: ea[i].mean()),
                    "E_alt (vote-weighted)": boot(lambda i: (N[i] * ea[i]).sum() / N[i].sum()),
                    "E_ref (vote-weighted)": boot(lambda i: (N[i] * er[i]).sum() / N[i].sum())})
    tables = {"components": pd.DataFrame(tab)}
    ref = "cont (calibrated)"
    pr = []
    for n in allnames[1:]:
        er, ea = E[n]
        r0, a0 = E[ref]
        pr.append({"comparison": f"{n} minus {ref}", "delta KL": boot(lambda i: (er[i] + ea[i] - r0[i] - a0[i]).mean()),
                   "delta E_ref": boot(lambda i: (er[i] - r0[i]).mean()), "delta E_alt": boot(lambda i: (ea[i] - a0[i]).mean()),
                   "delta E_alt (vote-weighted)": boot(lambda i: (N[i] * (ea[i] - a0[i])).sum() / N[i].sum())})
    tables["paired_vs_cont"] = pd.DataFrame(pr)
    er_rows = []
    for n in ("HSEmotion + MLP head", "teacher (5-mean)"):
        er, ea = E[n]
        r0, a0 = E[ref]
        for e in sorted(set(emo)):
            sel = np.flatnonzero(emo == e)
            be = Bootstrap(act[sel], N_BOOT)
            er_rows.append({"comparison": f"{n} minus {ref}", "intended emotion": e, "clips": len(sel),
                            "share of clips with p_y = 1": round(float((P[sel].max(1) >= 1 - 1e-12).mean()), 3),
                            "delta E_ref": be(lambda i: (er[sel][i] - r0[sel][i]).mean()), "delta E_alt": be(lambda i: (ea[sel][i] - a0[sel][i]).mean())})
    tables["by_emotion"] = pd.DataFrame(er_rows)
    tables["data"] = pd.DataFrame([{"clips": len(R), "share with p_y = 1": round(float((P.max(1) >= 1 - 1e-12).mean()), 4),
                                    "share with tied reference classes": round(float(((P >= P.max(1, keepdims=True) - 1e-12).sum(1) > 1).mean()), 4),
                                    "median raters": float(np.median(N)), "EB alpha": round(alpha, 3)}])
    lines = ["# Step 0: decomposition of KL(votes || prediction) into the reference class and the alternatives (V reading)\n",
             "Generated by `analysis/kl_decomposition.py`; definitions in its docstring.\n"]
    for name, tb in tables.items():
        tb.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(tb.astype(object).where(tb.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
