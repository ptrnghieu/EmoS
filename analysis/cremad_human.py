"""M1 on CREMA-D: how does cross-modal conflict, as people perceive it, relate to people's disagreement on the
audio-visual clip? (Plan approved 21/10/2026.)

CREMA-D clips are rated by about 10 people in each of three conditions, by different people per condition. So the
conflict measures (from the audio-only and video-only raters) and the target (the audio-visual raters'
disagreement) come from disjoint groups of people.

Target: entropy of the audio-visual votes (all raters, about 10 per clip); `disagree` = not unanimous.
Predictors:
    conflict, binary   the audio-only and video-only top labels share no label (ties: every top label counts)
    conflict, graded   Jensen-Shannon divergence between the audio-only and video-only vote distributions
    ambiguity A / V    entropy of the audio-only / video-only votes (ambiguity within one modality)

Average level: OLS of the audio-visual entropy on each predictor (standardised when continuous), standard errors
clustered by actor. Clip level: Spearman with the audio-visual entropy and AUROC for `disagree`, 95% bootstrap over
actors. Then all predictors together: a linear model fitted on the train actors, evaluated on the test actors.
Ceiling: a Dirichlet-multinomial fitted to the audio-visual votes gives the best Spearman any predictor could reach
against the observed (about 10-rater) entropy; the model is checked against an empirical split (about 7 raters
predict the other 3).

Usage (from the repository root; downloads the CREMA-D rating files, needs scipy and statsmodels):
    python analysis/cremad_human.py --out results/cremad_human
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import gammaln

from preprocess_cremad import LABELS, clip_table
from rq3_analysis import N_BOOT, Bootstrap
from rq_analysis import auroc, markdown, spearman


def counts(df: pd.DataFrame, cond: str) -> np.ndarray:
    return df[[f"{cond}_{l}" for l in LABELS]].to_numpy(float)


def entropy_counts(c: np.ndarray) -> np.ndarray:
    p = c / np.clip(c.sum(-1, keepdims=True), 1, None)
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0), -1)


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0), -1)
    return (kl(p, m) + kl(q, m)) / 2


def fit_dirichlet(C: np.ndarray) -> tuple[float, np.ndarray]:
    n = C.sum(1)

    def nll(x):
        a, p = np.exp(x[0]), np.exp(x[1:]) / np.exp(x[1:]).sum()
        ap = a * p
        return -(gammaln(a) - gammaln(a + n) + (gammaln(C + ap) - gammaln(ap)).sum(1)).sum()

    res = minimize(nll, np.concatenate([[0.0], np.log(C.mean(0) + 1e-3)]), method="L-BFGS-B")
    return float(np.exp(res.x[0])), np.exp(res.x[1:]) / np.exp(res.x[1:]).sum()


def ceiling(C: np.ndarray, sims: int = 200, seed: int = 0) -> dict:
    """Simulate clips from the fitted Dirichlet-multinomial with each clip's own number of raters."""
    alpha, p = fit_dirichlet(C)
    n = C.sum(1).astype(int)
    rng = np.random.default_rng(seed)
    perfect, split_sim = [], []
    for _ in range(sims):
        theta = rng.dirichlet(alpha * p, size=len(C))
        obs = rng.multinomial(n, theta).astype(float)
        # perfect predictor: expected entropy of n votes given theta (Monte Carlo)
        expected = np.mean([entropy_counts(rng.multinomial(n, theta).astype(float)) for _ in range(20)], 0)
        perfect.append(spearman(expected, entropy_counts(obs)))
        held = rng.multinomial(3, theta).astype(float)
        rest = rng.multinomial(n - 3, theta).astype(float)
        split_sim.append(spearman(entropy_counts(rest), entropy_counts(held)))
    return {"alpha": alpha, "perfect": np.array(perfect), "split_sim": np.array(split_sim)}


def empirical_split(C: np.ndarray, seed: int = 0) -> float:
    """Spearman between the entropy of about 7 raters and of the other 3, raters drawn at random per clip."""
    rng = np.random.default_rng(seed)
    held, rest = [], []
    for c in C.astype(int):
        votes = rng.permutation(np.repeat(np.arange(len(c)), c))
        held.append(np.bincount(votes[:3], minlength=len(c)))
        rest.append(np.bincount(votes[3:], minlength=len(c)))
    return spearman(entropy_counts(np.array(rest, float)), entropy_counts(np.array(held, float)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/cremad_human"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table()
    df = df[(df[["a_n", "v_n", "av_n"]] > 0).all(1)].reset_index(drop=True)
    A, V, AV = counts(df, "a"), counts(df, "v"), counts(df, "av")
    top = lambda c: c == c.max(1, keepdims=True)
    df["conflict"] = (~(top(A) & top(V)).any(1)).astype(float)
    df["conflict_graded"] = jsd(A / A.sum(1, keepdims=True), V / V.sum(1, keepdims=True))
    df["ambiguity_a"], df["ambiguity_v"] = entropy_counts(A), entropy_counts(V)
    df["av_entropy"] = entropy_counts(AV)
    df["disagree"] = (AV > 0).sum(1) > 1
    predictors = {"conflict, binary": "conflict", "conflict, graded (JSD audio vs video)": "conflict_graded",
                  "ambiguity, audio only (entropy)": "ambiguity_a", "ambiguity, video only (entropy)": "ambiguity_v"}
    boot = Bootstrap(df.video_id.to_numpy(), N_BOOT)
    y, dis = df.av_entropy.to_numpy(), df.disagree.to_numpy()
    tables = {}

    tables["data"] = pd.DataFrame([{
        "clips": len(df), "actors": df.actor.nunique(),
        "raters per clip, audio / video / audio-visual (median)": f"{df.a_n.median():.0f} / {df.v_n.median():.0f} / {df.av_n.median():.0f}",
        "clips with conflict": f"{int(df.conflict.sum())} ({df.conflict.mean():.0%})",
        "audio-visual clips not unanimous": f"{dis.mean():.0%}", "audio-visual entropy, mean": round(y.mean(), 3)}])

    # ---- average level: OLS, SEs clustered by actor ---------------------------------------------
    rows = []
    for name, col in predictors.items():
        x = df[col].to_numpy(float)
        if col != "conflict":
            x = (x - x.mean()) / x.std()
        fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="cluster", cov_kwds={"groups": df.actor.to_numpy()})
        lo, hi = fit.conf_int()[1]
        rows.append({"predictor": name, "unit": "conflict vs none" if col == "conflict" else "per SD",
                     "change in audio-visual entropy [95% CI]": f"{fit.params[1]:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                     "p": fit.pvalues[1], "R²": round(fit.rsquared, 3)})
    tables["average_level_ols"] = pd.DataFrame(rows)
    tables["entropy_by_conflict"] = df.groupby("conflict").agg(
        clips=("uid", "size"), av_entropy_mean=("av_entropy", "mean"), av_entropy_sd=("av_entropy", "std"),
        share_not_unanimous=("disagree", "mean")).reset_index()

    # ---- clip level ------------------------------------------------------------------------------
    rows = []
    for name, col in predictors.items():
        x = df[col].to_numpy(float)
        rows.append({"predictor": name, "Spearman with audio-visual entropy": boot(lambda i, x=x: spearman(x[i], y[i])),
                     "AUROC: flags disagreement": boot(lambda i, x=x: auroc(x[i], dis[i]))})
    train, test = (df["mode"] == "train").to_numpy(), (df["mode"] == "test").to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in predictors.values()])
    beta = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
    combined = X @ beta
    boot_test = Bootstrap(df.video_id.to_numpy()[test], N_BOOT)
    yt, dt, ct = y[test], dis[test], combined[test]
    rows.append({"predictor": "all four, linear (fitted on train actors; test actors only)",
                 "Spearman with audio-visual entropy": boot_test(lambda i: spearman(ct[i], yt[i])),
                 "AUROC: flags disagreement": boot_test(lambda i: auroc(ct[i], dt[i]))})
    tables["clip_level"] = pd.DataFrame(rows)

    # ---- ceiling ---------------------------------------------------------------------------------
    c = ceiling(AV)
    tables["ceiling"] = pd.DataFrame([
        {"quantity": "fitted Dirichlet concentration (alpha)", "value": round(c["alpha"], 3)},
        {"quantity": "check: about 7 raters predict the other 3, empirical", "value": round(empirical_split(AV), 3)},
        {"quantity": "check: the same, simulated from the fitted model (mean)", "value": round(c["split_sim"].mean(), 3)},
        {"quantity": "ceiling: Spearman of a perfect predictor with the observed audio-visual entropy (mean [2.5%, 97.5%])",
         "value": f"{c['perfect'].mean():.3f} [{np.percentile(c['perfect'], 2.5):.3f}, {np.percentile(c['perfect'], 97.5):.3f}]"}])

    lines = ["# M1 on CREMA-D: perceived cross-modal conflict and audio-visual disagreement\n",
             "Generated by `analysis/cremad_human.py`. Conflict measures come from the audio-only and video-only raters, "
             f"the target from the audio-visual raters (different people). CIs: 95% bootstrap over actors ({N_BOOT} resamples).\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
