"""Diagnostic before any new module: does modality-intrinsic ambiguity, especially visual ambiguity, explain human
disagreement on the audio-visual clip better than cross-modal conflict, and is it more than low expression intensity?

Per clip i and condition m in {V, A, AV} (video only, audio only, audio-visual; different raters per condition,
about 10 each): p_i^m(k) = n_{i,m,k} / n_{i,m} over CREMA-D's K = 6 labels, and the normalised entropy
    A_i^m = -(1 / log K) sum_k p_i^m(k) log p_i^m(k)        (0 = unanimous, 1 = uniform)
A^AV is the target (human disagreement); A^V and A^A are the predictors of modality-intrinsic ambiguity.

Other predictors:
    C_bin     conflict: the audio-only and video-only top labels share no label (ties: every top label counts)
    C_jsd     graded conflict: Jensen-Shannon divergence between p^A and p^V
    I^V, I^A  perceived intensity: mean intensity rating (respLevel, 1-100) of the video-only / audio-only raters
    class     the intended emotion (from the file name), as indicator variables
    actor     (fixed effects, in the in-sample model only)

1. Out-of-sample comparison. Linear models of A^AV, fitted and evaluated with 10-fold cross-validation over actors
   (every model uses the same folds), out-of-fold predictions scored by Spearman with A^AV, AUROC for the
   high-disagreement clips (top quarter of A^AV) and mean absolute error. 95% bootstrap over actors, with paired
   differences between models.
2. In-sample model with actor fixed effects: standardised coefficients, cluster-robust (actor) standard errors, and
   the drop in R^2 when each predictor is removed (its unique contribution).
3. Ambiguity versus low intensity: correlation of A^V with I^V; A^AV by A^V tercile within I^V terciles; and, on the
   one sentence with an intended intensity level (IEO: low / medium / high), A^V and A^AV by level.
Sensitivity: everything in 1 is repeated with smoothed entropies (add 0.5 to each count) against sampling noise.

Usage (from the repository root; downloads the CREMA-D rating files; needs statsmodels):
    python analysis/cremad_ambiguity_diagnostic.py --out results/cremad_ambiguity_diagnostic
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cremad_human import counts, jsd
from preprocess_cremad import LABELS, RAW, clip_table, fetch
from rq3_analysis import N_BOOT, Bootstrap
from rq_analysis import auroc, markdown, spearman

K = len(LABELS)


def norm_entropy(c: np.ndarray, smooth: float = 0.0) -> np.ndarray:
    c = c + smooth
    p = c / c.sum(1, keepdims=True)
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0), 1) / np.log(K)


def intensity(responses: pd.DataFrame) -> pd.DataFrame:
    r = responses.drop_duplicates(["localid", "clipName", "queryType"]).copy()
    r["level"] = pd.to_numeric(r["respLevel"], errors="coerce")
    out = r.groupby(["clipName", "queryType"])["level"].mean().unstack()
    return out.rename(columns={1: "I_A", 2: "I_V", 3: "I_AV"})


def design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [np.ones((len(df), 1))]
    for c in cols:
        if c == "class":
            parts.append(pd.get_dummies(df["intended"], drop_first=True).to_numpy(float))
        else:
            parts.append(df[[c]].to_numpy(float))
    return np.hstack(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/cremad_ambiguity_diagnostic"))
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table()
    df = df[(df[["a_n", "v_n", "av_n"]] > 0).all(1)].reset_index(drop=True)
    responses = pd.read_csv(io.BytesIO(fetch(f"{RAW}/finishedResponses.csv")), low_memory=False)
    df = df.merge(intensity(responses), left_on="uid", right_index=True, how="left")
    A, V, AV = counts(df, "a"), counts(df, "v"), counts(df, "av")
    top = lambda c: c == c.max(1, keepdims=True)
    df["C_bin"] = (~(top(A) & top(V)).any(1)).astype(float)
    df["C_jsd"] = jsd(A / A.sum(1, keepdims=True), V / V.sum(1, keepdims=True))
    rng = np.random.default_rng(0)
    actors = rng.permutation(df.actor.unique())
    fold_of = {a: k % args.folds for k, a in enumerate(actors)}
    fold = df.actor.map(fold_of).to_numpy()
    boot = Bootstrap(df.video_id.to_numpy(), N_BOOT)

    models = {
        "controls: class + intensity (I^V, I^A)": ["class", "I_V", "I_A"],
        "conflict (C_bin, C_jsd)": ["C_bin", "C_jsd"],
        "conflict + controls": ["C_bin", "C_jsd", "class", "I_V", "I_A"],
        "A^V": ["A_V"],
        "A^A": ["A_A"],
        "A^V + A^A": ["A_V", "A_A"],
        "conflict + A^V": ["C_bin", "C_jsd", "A_V"],
        "conflict + A^V + A^A": ["C_bin", "C_jsd", "A_V", "A_A"],
        "all: conflict + A^V + A^A + controls": ["C_bin", "C_jsd", "A_V", "A_A", "class", "I_V", "I_A"],
        "all without A^V": ["C_bin", "C_jsd", "A_A", "class", "I_V", "I_A"],
    }
    pairs = [("conflict + A^V", "conflict (C_bin, C_jsd)"), ("A^V", "conflict (C_bin, C_jsd)"),
             ("all: conflict + A^V + A^A + controls", "all without A^V"),
             ("all: conflict + A^V + A^A + controls", "conflict + controls"), ("A^V", "A^A")]
    tables = {}

    for smooth, tag in ((0.0, ""), (0.5, "_smoothed")):
        df["A_V"], df["A_A"], df["A_AV"] = norm_entropy(V, smooth), norm_entropy(A, smooth), norm_entropy(AV, smooth)
        y = df.A_AV.to_numpy()
        high = y >= np.quantile(y, 0.75)
        preds = {}
        for name, cols in models.items():
            X = design(df, cols)
            pred = np.zeros(len(df))
            for k in range(args.folds):
                tr, te = fold != k, fold == k
                beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                pred[te] = X[te] @ beta
            preds[name] = pred
        stat = lambda p: {"Spearman with A^AV": lambda i: spearman(p[i], y[i]),
                          "AUROC: high disagreement (top 25%)": lambda i: auroc(p[i], high[i]),
                          "MAE": lambda i: np.abs(p[i] - y[i]).mean()}
        tables[f"1_out_of_sample{tag}"] = pd.DataFrame(
            [{"model (10-fold CV over actors)": n, **{k: boot(f) for k, f in stat(p).items()}} for n, p in preds.items()])
        rows = []
        for a, b in pairs:
            sa, sb = stat(preds[a]), stat(preds[b])
            rows.append({"comparison": f"{a} minus {b}", **{k: boot(lambda i, k=k: sa[k](i) - sb[k](i)) for k in sa}})
        tables[f"1_paired_differences{tag}"] = pd.DataFrame(rows)

    # ---- 2. in-sample model with actor fixed effects ----------------------------------------------
    df["A_V"], df["A_A"], df["A_AV"] = norm_entropy(V), norm_entropy(A), norm_entropy(AV)
    names = ["A_V", "A_A", "C_bin", "C_jsd", "I_V", "I_A"]
    z = df[names].apply(lambda s: (s - s.mean()) / s.std())
    dummies = pd.concat([pd.get_dummies(df.intended, prefix="class", drop_first=True),
                         pd.get_dummies(df.actor, prefix="actor", drop_first=True)], axis=1).astype(float)
    X = sm.add_constant(pd.concat([z, dummies], axis=1))
    groups = df.actor.to_numpy()
    full = sm.OLS(df.A_AV, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    rows = []
    for n in names:
        reduced = sm.OLS(df.A_AV, X.drop(columns=n)).fit()
        lo, hi = full.conf_int().loc[n]
        rows.append({"predictor (standardised)": n, "coefficient [95% CI, clustered by actor]": f"{full.params[n]:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                     "p": full.pvalues[n], "unique R² (drop when removed)": round(full.rsquared - reduced.rsquared, 4)})
    rows.append({"predictor (standardised)": "model R² (with class and actor fixed effects)", "unique R² (drop when removed)": round(full.rsquared, 3)})
    tables["2_in_sample_actor_fixed_effects"] = pd.DataFrame(rows)

    # ---- 3. ambiguity versus low intensity --------------------------------------------------------
    rows = [{"quantity": "Spearman(A^V, I^V): visual ambiguity vs perceived visual intensity", "value": boot(lambda i: spearman(df.A_V.to_numpy()[i], df.I_V.to_numpy()[i]))},
            {"quantity": "Spearman(A^AV, I^V)", "value": boot(lambda i: spearman(df.A_AV.to_numpy()[i], df.I_V.to_numpy()[i]))},
            {"quantity": "Spearman(A^AV, A^V)", "value": boot(lambda i: spearman(df.A_AV.to_numpy()[i], df.A_V.to_numpy()[i]))}]
    tables["3_ambiguity_vs_intensity"] = pd.DataFrame(rows)
    df["I_V tercile"] = pd.qcut(df.I_V, 3, labels=["low", "mid", "high"])
    df["A_V tercile"] = pd.qcut(df.A_V.rank(method="first"), 3, labels=["low", "mid", "high"])
    tables["3_A_AV_by_A_V_within_I_V"] = df.pivot_table(index="I_V tercile", columns="A_V tercile", values="A_AV",
                                                        aggfunc="mean", observed=False).round(3).reset_index()
    ieo = df[df.level.isin(["low", "medium", "high"])]
    tables["3_intended_level_IEO_sentence"] = ieo.groupby("level").agg(
        clips=("uid", "size"), A_V=("A_V", "mean"), A_A=("A_A", "mean"), A_AV=("A_AV", "mean"), I_V=("I_V", "mean")
    ).reindex(["low", "medium", "high"]).round(3).reset_index()

    lines = ["# Visual ambiguity, conflict and intensity as explanations of audio-visual disagreement (CREMA-D)\n",
             f"Generated by `analysis/cremad_ambiguity_diagnostic.py`. {len(df)} clips, {df.actor.nunique()} actors. "
             f"Normalised entropies (0-1). Bootstrap: 95%, over actors, {N_BOOT} resamples.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
