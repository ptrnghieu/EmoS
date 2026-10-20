"""Bayesian causal inference (Koerding et al., 2007) on CH-SIMS v2 + EmoS: a test on the human data and a
training-free combiner over the models' own predictions.

1. Humans. Causal inference predicts that observers fuse cues that differ a little, treat cues that differ
   a lot as having separate causes, and are most uncertain about the causal structure in between, so
   disagreement should peak at an intermediate cue discrepancy instead of rising steadily. Discrepancy =
   the spread (max - min) of the T, A, V valence labels, 0 to 2. Per spread bin: share of clips where all
   3 annotators agree and the entropy of the 3 votes; then the ordinal mixed model of rq1_models.py with
   the spread as categories, and with spread + spread^2 (intensity, batch, random intercept per video).

2. Models. A combiner in the spirit of causal inference, built only from scores already computed: the
   T+A+V answer stands for the model's fused reading (it can also capture readings such as sarcasm that
   need all modalities together), the T, A and V answers for the separate readings. Per clip,
       pi = sigmoid(a + b * agreement),  agreement = 1 - mean pairwise JSD of the T, A, V distributions,
       Q  = pi * P_TAV + (1 - pi) * sum_m w_m * P_m,  then a temperature on Q.
   Parameters are fitted on the train and valid splits (minimising the JSD to the 3-vote distribution)
   and evaluated on the test split. Ablations: T+A+V with a temperature only; a fixed mixture (b = 0, so
   the weight of the fused reading does not depend on the clip); and an oracle that replaces the model's
   agreement by the humans' (1 - T/A/V valence spread / 2), to tell a weak combiner from unimodal readings
   that do not see the conflict. A diagnostic table asks the latter directly: does the disagreement among
   the model's T, A, V readings flag polarity conflict, follow the label spread, follow human entropy?
   Emotion-LLaMA is scored over its six labels against the humans' six-label distribution. 95%
   video-level bootstrap on the test split.

Usage (from the repository root; needs scipy and statsmodels):
    python analysis/bci_analysis.py --out results/bci_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rq1_models import CLMM, fmt_or, load_data
from rq2_compare_models import ALL_RUNS, SIX, SIX_LABEL_RUNS
from rq3_analysis import N_BOOT, Bootstrap, load_logits, softmax
from rq_analysis import LABELS, auroc, entropy, jsd, markdown, spearman

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6]  # lower edges; the last bin is 1.6-2.0


def spread_bins(spread: np.ndarray) -> np.ndarray:
    return np.array([f"{BINS[i]:.1f}" + ("-2.0" if i == len(BINS) - 1 else "-1.4" if BINS[i] == 1.2 else "")
                     for i in np.searchsorted(BINS, spread + 1e-9, side="right") - 1])


def human_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    spread = np.ptp(df[["label_T", "label_A", "label_V"]].to_numpy(), axis=1)
    df = df.assign(spread=spread, spread_bin=spread_bins(spread))
    rows = []
    for b, sub in df.groupby("spread_bin", sort=True):
        boot = Bootstrap(sub.video_id.to_numpy(), N_BOOT)
        agree, ent = sub.all_agree.to_numpy(), sub.human_entropy.to_numpy()
        rows.append({"T/A/V valence spread": b, "clips": len(sub), "conflict (polarity) share": round(sub.conflict.mean(), 2),
                     "mean intensity |label|": round(sub.intensity.mean(), 2),
                     "all 3 agree": boot(lambda i: agree[i].mean()), "entropy of the 3 votes": boot(lambda i: ent[i].mean())})
    out["bci_humans_by_spread"] = pd.DataFrame(rows)

    rows = []
    levels = sorted(df.spread_bin.unique())
    X = np.column_stack([(df.spread_bin == l).to_numpy(float) for l in levels[1:]] + [df.intensity, df.last_batch])
    m = CLMM(df.y.to_numpy(), X, df.video_id.to_numpy(), [f"spread {l}" for l in levels[1:]] + ["intensity", "batch"], True).fit()
    for _, r in m.coef().iterrows():
        if r.term.startswith("spread"):
            rows.append({"model": f"spread bins (reference {levels[0]})", "term": r.term,
                         "OR for more disagreement [95% CI]": fmt_or(r), "p": r.p})
    s = df.spread.to_numpy()
    X = np.column_stack([s, s ** 2, df.intensity, df.last_batch])
    m = CLMM(df.y.to_numpy(), X, df.video_id.to_numpy(), ["spread", "spread^2", "intensity", "batch"], True).fit()
    c = m.coef().set_index("term")
    for t in ("spread", "spread^2"):
        rows.append({"model": "spread + spread^2", "term": t, "OR for more disagreement [95% CI]": fmt_or(c.loc[t]), "p": c.loc[t].p})
    b1, b2 = np.log(c.loc["spread", "OR"]), np.log(c.loc["spread^2", "OR"])
    rows.append({"model": "spread + spread^2", "term": "spread at the peak (-b1 / 2 b2)",
                 "OR for more disagreement [95% CI]": f"{-b1 / (2 * b2):.2f}" if b2 < 0 else "no peak (b2 >= 0)", "p": np.nan})
    out["bci_humans_mixed_models"] = pd.DataFrame(rows)
    return out


# ---- combiner ----------------------------------------------------------------------------------------

def mean_pairwise_jsd(PT, PA, PV):
    return (jsd(PT, PA) + jsd(PT, PV) + jsd(PA, PV)) / 3


def combine(theta, P, agreement, variant):
    a, b, wT, wA, wV, log_t = theta
    if variant == "TAV + temperature":
        Q = P["TAV"]
    else:
        w = np.exp([wT, wA, wV]) / np.exp([wT, wA, wV]).sum()
        seg = w[0] * P["T"] + w[1] * P["A"] + w[2] * P["V"]
        pi = 1 / (1 + np.exp(-(a + (b if variant.startswith("causal inference") else 0.0) * agreement)))
        Q = pi[:, None] * P["TAV"] + (1 - pi[:, None]) * seg
    Q = np.clip(Q, 1e-12, None) ** (1 / np.exp(log_t))
    return Q / Q.sum(1, keepdims=True)


def fit(P, agreement, H, variant):
    loss = lambda th: jsd(combine(th, P, agreement, variant), H).mean()
    starts = [np.array([a, b, 0, 0, 0, 0.0]) for a in (-1.0, 1.0) for b in (-2.0, 2.0)]
    best = min((minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7})
                for x0 in starts), key=lambda r: r.fun)
    return best.x


def model_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    H7 = np.stack(df.human_dist)
    fit_rows, test = df["mode"].isin(["train", "valid"]).to_numpy(), (df["mode"] == "test").to_numpy()
    runs = [(n, p, LABELS) for n, p in ALL_RUNS.items() if Path(p).exists()] + [(n, p, SIX) for n, p in SIX_LABEL_RUNS.items()]
    variants = ["TAV + temperature", "fixed mixture", "causal inference", "causal inference, oracle conflict"]
    spread = np.ptp(df[["label_T", "label_A", "label_V"]].to_numpy(), axis=1)
    rows, diffs, params, diag = [], [], [], []
    for name, path, labels in runs:
        cols = [LABELS.index(l) for l in labels]
        keep = H7[:, cols].sum(1) > 0
        H = H7[keep][:, cols] / H7[keep][:, cols].sum(1, keepdims=True)
        sub = df[keep].reset_index(drop=True)
        logits = load_logits(path)[[f"logit_{l}" for l in labels]]
        P = {c: softmax(logits.loc[[(u, c) for u in sub.uid]].to_numpy(float)) for c in ("TAV", "T", "A", "V")}
        agreement = 1 - mean_pairwise_jsd(P["T"], P["A"], P["V"])
        oracle = 1 - spread[keep] / 2
        diag.append({"model": name, "AUROC: T/A/V readings disagree on polarity-conflict clips": round(auroc(1 - agreement, sub.conflict.to_numpy() == 1), 3),
                     "Spearman with the T/A/V label spread": round(spearman(1 - agreement, spread[keep]), 3),
                     "Spearman with human entropy": round(spearman(1 - agreement, entropy(H)), 3)})
        fr, te = fit_rows[keep], test[keep]
        Pf, Pt = {c: v[fr] for c, v in P.items()}, {c: v[te] for c, v in P.items()}
        Ht, ent_h = H[te], entropy(H[te])
        conflict, disagree = sub.conflict.to_numpy()[te] == 1, ent_h > 1e-9
        boot = Bootstrap(sub.video_id.to_numpy()[te], N_BOOT)

        def stats(Q):
            e = entropy(Q)
            j = jsd(Q, Ht)
            return {"JSD to humans": lambda i: j[i].mean(),
                    "Spearman(entropy, human entropy)": lambda i: spearman(e[i], ent_h[i]),
                    "AUROC: entropy flags disagreement": lambda i: auroc(e[i], disagree[i]),
                    "entropy change on conflict": lambda i: e[i][conflict[i]].mean() - e[i][~conflict[i]].mean()}

        Qs = {"T+A+V as is": Pt["TAV"]}
        for v in variants:
            g = oracle if v.endswith("oracle conflict") else agreement
            th = fit(Pf, g[fr], H[fr], v)
            Qs[v] = combine(th, Pt, g[te], v)
            if v != "TAV + temperature":
                w = np.exp(th[2:5]) / np.exp(th[2:5]).sum()
                pi = 1 / (1 + np.exp(-(th[0] + (th[1] if v.startswith("causal inference") else 0) * g[te])))
                params.append({"model": name, "variant": v, "a": round(th[0], 2), "b (agreement)": round(th[1], 2) if v.startswith("causal inference") else 0,
                               "w_T": round(w[0], 2), "w_A": round(w[1], 2), "w_V": round(w[2], 2), "temperature": round(np.exp(th[5]), 2),
                               "mean weight of the fused reading, no conflict": round(pi[~conflict].mean(), 2),
                               "mean weight of the fused reading, conflict": round(pi[conflict].mean(), 2)})
        for v, Q in Qs.items():
            s = stats(Q)
            pred = Q.argmax(1)
            rows.append({"model": name, "predictions": v, **{k: boot(f) for k, f in s.items()},
                         "accuracy vs majority": round(float((np.array(labels)[pred] == sub.ds_label.to_numpy()[te]).mean()), 3)})
        for a, b in (("causal inference", "TAV + temperature"), ("causal inference", "fixed mixture"),
                     ("causal inference, oracle conflict", "TAV + temperature")):
            sa, sb = stats(Qs[a]), stats(Qs[b])
            diffs.append({"model": name, "comparison": f"{a} minus {b}", **{k: boot(lambda i, k=k: sa[k](i) - sb[k](i)) for k in sa}})
    return {"bci_models_unimodal_readings_vs_conflict": pd.DataFrame(diag), "bci_models_test": pd.DataFrame(rows), "bci_models_paired_differences": pd.DataFrame(diffs),
            "bci_models_fitted_parameters": pd.DataFrame(params)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/bci_analysis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_data()
    tables = {**human_tables(df), **model_tables(df)}
    lines = ["# Bayesian causal inference: human data and a training-free combiner over model predictions\n",
             f"Generated by `analysis/bci_analysis.py`. {len(df)} clips; combiner fitted on train + valid, evaluated on "
             f"the test split; 95% video-level bootstrap ({N_BOOT} resamples).\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
