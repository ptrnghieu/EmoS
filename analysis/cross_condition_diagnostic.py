"""Does multi-condition soft-label SFT reproduce how human judgements change from one modality to audio-visual?
(Diagnostic before any new method, 02/11/2026; existing predictions only, no training.)

Inputs: the Stage 1 outer runs of m3 and s1_noI (results/stage1/stage1_<arm>_f<k>_s0, seed 0): each clip read under AV,
V and A by the adapter of the fold that held its actor out. Humans: CREMA-D vote counts per condition (about 10
different raters per condition). p^c = human vote distribution, q^c = softmax of the model's six label logits.

1. Each condition on its own: JSD(q^c, p^c), top-label agreement (the model's label is one of the raters' top labels),
   Spearman of normalised entropies. Without this an integration failure could just be a poor unimodal prediction.
2. Changes X -> AV for X in {V, A}, per clip:
       Delta_h = p^AV - p^X,  Delta_m = q^AV - q^X            (probability redistribution, 6-vectors)
       dH_h = H(p^AV) - H(p^X),  dH_m = H(q^AV) - H(q^X)      (entropy change, normalised entropy)
   redistribution: mean per-clip cosine(Delta_m, Delta_h) over clips with ||Delta_h|| >= 0.1; correlation pooled over
   clip x label; sign agreement on the labels humans move by >= 0.2; magnitude ratio mean||Delta_m|| / mean||Delta_h||
   entropy change: Spearman(dH_m, dH_h); sign agreement on clips with |dH_h| >= 0.15
3. Integration: on clips whose unimodal human distributions differ (||p^A - p^V||_2 >= 0.3), the audio weight w of the
   best mixture p^AV ~ (1 - w) p^V + w p^A (least squares, clipped to [0, 1]); the model's w from its own q^V, q^A, q^AV.
   Disambiguation: H(AV) < min(H(V), H(A)) - 0.05 (AV observers agree more than either unimodal group).
References:
   human re-run         every metric computed between the observed human data and a simulated re-run of the human
                        study (per clip and condition: theta ~ Dirichlet(counts + 0.5), the same number of new votes);
                        mean over 20 re-runs = how well another group of ~10 raters per condition would agree. A
                        reference, not an upper bound: a predictor of the underlying distribution, free of the re-run's
                        own sampling noise, can exceed it.
   emotion prior        for changes: the mean human Delta / dH of the intended emotion, from the other outer folds
   subset "both unimodal readings good": JSD(q^V, p^V) and JSD(q^A, p^A) both at or below their median (per model)
CIs: 95% bootstrap over actors (300 resamples). Failure cases (s1_noI): clips whose two unimodal readings are good and
whose human V -> AV change is large (||Delta_h|| >= 0.4), lowest cosine first, one per emotion where possible.

Usage (from the repository root; downloads the CREMA-D rating files; needs matplotlib):
    python analysis/cross_condition_diagnostic.py --out results/cross_condition_diagnostic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import counts, jsd
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, softmax
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds

K = len(LABELS)
CONDS = ("V", "A", "AV")
ARMS = ("m3", "s1_noI")
N_REP, N_BOOT = 20, 300


def load_arm(folder: Path, arm: str) -> dict[str, pd.DataFrame]:
    parts = []
    for f in range(N_OUTER):
        s = pd.DataFrame([json.loads(l) for l in open(folder / f"stage1_{arm}_f{f}_s0" / "scores.jsonl", encoding="utf-8") if l.strip()])
        parts.append(s[s.get("error", pd.Series(index=s.index, dtype=object)).isna()])
    s = pd.concat(parts).drop_duplicates(["uid", "condition"], keep="last")
    return {c: s[s.condition == c].set_index("uid")[[f"logit_{l}" for l in LABELS]] for c in CONDS}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def change_metrics(Dm, Dh, dHm, dHh):
    """Per-clip ingredients; each metric is a function of a clip index set (for the bootstrap)."""
    big = np.linalg.norm(Dh, axis=1) >= 0.1
    cos = cosine_rows(Dm, Dh)
    moved = np.abs(Dh) >= 0.2
    agree = np.sign(Dm) == np.sign(Dh)
    hchange = np.abs(dHh) >= 0.15
    return {
        "redistribution: cosine (||Dh|| >= 0.1)": lambda i: cos[i][big[i]].mean(),
        "redistribution: correlation over clip x label": lambda i: np.corrcoef(Dm[i].ravel(), Dh[i].ravel())[0, 1],
        "redistribution: sign agreement where humans move >= 0.2": lambda i: agree[i][moved[i]].mean(),
        "magnitude: mean||Dm|| / mean||Dh||": lambda i: np.linalg.norm(Dm[i], axis=1).mean() / np.linalg.norm(Dh[i], axis=1).mean(),
        "entropy change: Spearman(dH_m, dH_h)": lambda i: spearman(dHm[i], dHh[i]),
        "entropy change: sign agreement where |dH_h| >= 0.15": lambda i: (np.sign(dHm[i]) == np.sign(dHh[i]))[hchange[i]].mean(),
    }


def mixture_weight(pv, pa, pav):
    d = pa - pv
    return np.clip(((pav - pv) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/stage1"))
    parser.add_argument("--out", type=Path, default=Path("results/cross_condition_diagnostic"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    models = {arm: load_arm(args.runs, arm) for arm in ARMS}
    uids = [u for u in df.index if (df.loc[u, ["a_n", "v_n", "av_n"]] > 0).all()
            and all(u in models[a][c].index for a in ARMS for c in CONDS)]
    df = df.loc[uids]
    C = {c: counts(df, k) for c, k in (("V", "v"), ("A", "a"), ("AV", "av"))}
    n = {c: C[c].sum(1) for c in CONDS}
    P = {c: C[c] / n[c][:, None] for c in CONDS}
    Q = {arm: {c: softmax(models[arm][c].loc[uids].to_numpy(float)) for c in CONDS} for arm in ARMS}
    actors, folds, emo = df.actor.to_numpy(), df.fold.to_numpy(), df.intended.to_numpy()
    boot = Bootstrap(actors, N_BOOT)
    rng = np.random.default_rng(0)
    H = lambda p: norm_entropy(p)

    # simulated re-runs of the human study
    reps = []
    for _ in range(N_REP):
        rep = {}
        for c in CONDS:
            g = rng.gamma(C[c] + 0.5)
            theta = g / g.sum(1, keepdims=True)
            rep[c] = rng.multinomial(n[c].astype(int), theta) / n[c][:, None]
        reps.append(rep)

    def mean_over_reps(fn):
        return float(np.nanmean([fn(rep) for rep in reps]))

    tables = {}

    # ---- 1. each condition ------------------------------------------------------------------------------
    rows = []
    for c in CONDS:
        top = P[c] == P[c].max(1, keepdims=True)
        for arm in ARMS:
            q = Q[arm][c]
            j = jsd(q, P[c])
            hit = top[np.arange(len(q)), q.argmax(1)]
            rows.append({"condition": c, "predictor": arm, "JSD to votes (lower is better)": boot(lambda i: j[i].mean()),
                         "top label among raters' top": boot(lambda i: hit[i].mean()),
                         "Spearman(H(q), H(p))": boot(lambda i: spearman(H(q)[i], H(P[c])[i]))})
        rows.append({"condition": c, "predictor": "human re-run (reference)",
                     "JSD to votes (lower is better)": f"{mean_over_reps(lambda r: jsd(r[c], P[c]).mean()):.3f}",
                     "top label among raters' top": f"{mean_over_reps(lambda r: top[np.arange(len(r[c])), r[c].argmax(1)].mean()):.3f}",
                     "Spearman(H(q), H(p))": f"{mean_over_reps(lambda r: spearman(H(r[c]), H(P[c]))):.3f}"})
    tables["1_each_condition"] = pd.DataFrame(rows)

    # ---- 2. changes X -> AV ------------------------------------------------------------------------------
    good = {arm: (jsd(Q[arm]["V"], P["V"]) <= np.median(jsd(Q[arm]["V"], P["V"])))
            & (jsd(Q[arm]["A"], P["A"]) <= np.median(jsd(Q[arm]["A"], P["A"]))) for arm in ARMS}
    rows, cos_store = [], {}
    for x in ("V", "A"):
        Dh, dHh = P["AV"] - P[x], H(P["AV"]) - H(P[x])
        # emotion prior from the other folds
        Dprior, dHprior = np.zeros_like(Dh), np.zeros_like(dHh)
        for f in range(N_OUTER):
            tr, te = folds != f, folds == f
            for e in np.unique(emo):
                Dprior[te & (emo == e)] = Dh[tr & (emo == e)].mean(0)
                dHprior[te & (emo == e)] = dHh[tr & (emo == e)].mean()
        preds = {arm: (Q[arm]["AV"] - Q[arm][x], H(Q[arm]["AV"]) - H(Q[arm][x])) for arm in ARMS}
        preds["emotion prior (other folds)"] = (Dprior, dHprior)
        for name, (Dm, dHm) in preds.items():
            for subset, mask in (("all clips", np.ones(len(uids), bool)),) + ((("both unimodal readings good", good[name]),) if name in ARMS else ()):
                idx = np.flatnonzero(mask)
                m = change_metrics(Dm[idx], Dh[idx], dHm[idx], dHh[idx])
                b = Bootstrap(actors[idx], N_BOOT)
                rows.append({"change": f"{x} -> AV", "predictor": name, "clips": subset, "n": len(idx), **{k: b(f) for k, f in m.items()}})
            if name in ARMS:
                cos_store[(name, x)] = cosine_rows(Dm, Dh)
        for subset, mask in (("all clips", np.ones(len(uids), bool)), ("both unimodal readings good (s1_noI subset)", good["s1_noI"])):
            idx = np.flatnonzero(mask)
            ceil = {k: mean_over_reps(lambda r, k=k: change_metrics((r["AV"] - r[x])[idx], Dh[idx], (H(r["AV"]) - H(r[x]))[idx], dHh[idx])[k](np.arange(len(idx))))
                    for k in change_metrics(Dh, Dh, dHh, dHh)}
            rows.append({"change": f"{x} -> AV", "predictor": "human re-run (reference)", "clips": subset, "n": len(idx),
                         **{k: f"{v:.3f}" for k, v in ceil.items()}})
            if subset != "all clips":
                m = change_metrics(Dprior[idx], Dh[idx], dHprior[idx], dHh[idx])
                b = Bootstrap(actors[idx], N_BOOT)
                rows.append({"change": f"{x} -> AV", "predictor": "emotion prior (other folds)", "clips": subset, "n": len(idx),
                             **{k: b(f) for k, f in m.items()}})
    tables["2_changes_to_AV"] = pd.DataFrame(rows)

    # ---- 3. integration ----------------------------------------------------------------------------------
    differ = np.linalg.norm(P["A"] - P["V"], axis=1) >= 0.3
    idx = np.flatnonzero(differ)
    wh = mixture_weight(P["V"], P["A"], P["AV"])
    disamb = lambda pv, pa, pav: H(pav) < np.minimum(H(pv), H(pa)) - 0.05
    dh = disamb(P["V"], P["A"], P["AV"])
    rows = [{"predictor": "humans (observed)", "clips with differing unimodal votes": len(idx),
             "mean audio weight w": f"{wh[idx].mean():.3f}", "Spearman(w, human w)": "",
             "disambiguation rate (all clips)": f"{dh.mean():.3f}", "recall of human disambiguation": "",
             "precision": ""}]
    for arm in ARMS:
        q = Q[arm]
        wm = mixture_weight(q["V"], q["A"], q["AV"])
        dm = disamb(q["V"], q["A"], q["AV"])
        b = Bootstrap(actors[idx], N_BOOT)
        rows.append({"predictor": arm, "clips with differing unimodal votes": len(idx),
                     "mean audio weight w": b(lambda i: wm[idx][i].mean()),
                     "Spearman(w, human w)": b(lambda i: spearman(wm[idx][i], wh[idx][i])),
                     "disambiguation rate (all clips)": boot(lambda i: dm[i].mean()),
                     "recall of human disambiguation": boot(lambda i: dm[i][dh[i]].mean()),
                     "precision": boot(lambda i: dh[i][dm[i]].mean() if dm[i].any() else np.nan)})
    rows.append({"predictor": "human re-run (reference)", "clips with differing unimodal votes": len(idx),
                 "mean audio weight w": f"{mean_over_reps(lambda r: mixture_weight(r['V'], r['A'], r['AV'])[idx].mean()):.3f}",
                 "Spearman(w, human w)": f"{mean_over_reps(lambda r: spearman(mixture_weight(r['V'], r['A'], r['AV'])[idx], wh[idx])):.3f}",
                 "disambiguation rate (all clips)": f"{mean_over_reps(lambda r: disamb(r['V'], r['A'], r['AV']).mean()):.3f}",
                 "recall of human disambiguation": f"{mean_over_reps(lambda r: disamb(r['V'], r['A'], r['AV'])[dh].mean()):.3f}",
                 "precision": f"{mean_over_reps(lambda r: dh[disamb(r['V'], r['A'], r['AV'])].mean()):.3f}"})
    tables["3_integration"] = pd.DataFrame(rows)

    # ---- failure cases (s1_noI) --------------------------------------------------------------------------
    arm = "s1_noI"
    Dh = P["AV"] - P["V"]
    cand = np.flatnonzero(good[arm] & (np.linalg.norm(Dh, axis=1) >= 0.4))
    order = cand[np.argsort(cos_store[(arm, "V")][cand])]
    picked, seen = [], set()
    for i in order:
        if emo[i] not in seen:
            picked.append(i)
            seen.add(emo[i])
        if len(picked) == 6:
            break
    case_rows = []
    for i in picked:
        fmt = lambda p: " ".join(f"{l[:3]} {v:.2f}" for l, v in zip(LABELS, p) if v >= 0.05)
        case_rows.append({"clip": uids[i], "intended": emo[i], "votes V/A/AV": f"{int(n['V'][i])}/{int(n['A'][i])}/{int(n['AV'][i])}",
                          "human V": fmt(P["V"][i]), "model V": fmt(Q[arm]["V"][i]), "human A": fmt(P["A"][i]), "model A": fmt(Q[arm]["A"][i]),
                          "human AV": fmt(P["AV"][i]), "model AV": fmt(Q[arm]["AV"][i]),
                          "cosine(Dm, Dh) V->AV": round(float(cos_store[(arm, "V")][i]), 2)})
    tables["4_failure_cases_s1_noI"] = pd.DataFrame(case_rows)
    tables["subset_sizes"] = pd.DataFrame([{"model": a, "both unimodal readings good": int(good[a].sum()),
                                            "of which large human V->AV change (>= 0.4)": int((good[a] & (np.linalg.norm(Dh, axis=1) >= 0.4)).sum())}
                                           for a in ARMS])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(picked), 3, figsize=(11, 2.1 * len(picked)), sharey=True)
    x = np.arange(K)
    for r, i in enumerate(picked):
        for c_i, c in enumerate(CONDS):
            ax = axes[r, c_i]
            ax.bar(x - 0.2, P[c][i], 0.4, color="#9a9a9a", label="humans")
            ax.bar(x + 0.2, Q[arm][c][i], 0.4, color="#2f6fd6", label="s1_noI")
            ax.set_xticks(x, [l[:3] for l in LABELS], fontsize=8)
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title({"V": "video only", "A": "audio only", "AV": "audio-visual"}[c], fontsize=10)
            if c_i == 0:
                ax.set_ylabel(f"{uids[i]}\n(intended {emo[i]})", fontsize=8)
    axes[0, 2].legend(fontsize=8, loc="upper right")
    fig.suptitle("s1_noI: both unimodal readings close to the votes, V -> AV change missed (the 6 largest misses of 479, one per emotion; selected extremes)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "failure_cases.png", dpi=130)

    lines = ["# Cross-condition diagnostic: does soft-label SFT reproduce how judgements change to AV?\n",
             f"Generated by `analysis/cross_condition_diagnostic.py`. {len(uids)} clips, {len(set(actors))} actors, each read by "
             f"the fold model that never saw its actor (Stage 1, seed 0). CIs: 95% bootstrap over actors ({N_BOOT}). "
             f"Human re-run = agreement of {N_REP} simulated re-runs of the human study with the observed votes.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    lines += ["![failure cases](failure_cases.png)", ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
