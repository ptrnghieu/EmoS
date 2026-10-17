"""RQ1 models: does cross-modal conflict predict annotator disagreement once intensity, annotator
batch and the video are controlled for?

1. Cumulative-logit mixed model (ordinal regression with a random intercept per video) for
   unique_labels (1, 2 or 3 different Basic-7 labels among the 3 annotators):
       logit P(unique_labels <= k) = theta_k - (x'beta + u_video),  u_video ~ N(0, sigma^2)
   so exp(beta) > 1 means more disagreement. Fitted by maximum likelihood with Gauss-Hermite
   quadrature over the random intercept; 95% Wald intervals from the numerical Hessian.
   Model A uses binary conflict, model B the conflict type (none / T / A / V / several modalities
   deviating from the multimodal label).
2. Conflict types compared pairwise (Wald tests on model B), Holm correction within each family.
3. Linear mixed models (random intercept per video) for the continuous disagreement measures:
   entropy of the 3 Basic-7 votes, entropy of the pooled GoEmotions labels, 1 - ds_confidence.
4. Robustness: model A refitted per annotator batch, per CH-SIMS v2 split, and without clips that
   any annotator labelled Surprise.

Usage (from the repository root; needs scipy and statsmodels):
    python analysis/rq1_models.py --out results/rq1_models
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import approx_fprime, minimize
from scipy.special import expit, logsumexp
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.numdiff import approx_hess3
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from chsims_emos import CHSIMS_DIR, clip_key
from rq_analysis import entropy, load_humans, markdown

CONFLICT_TYPES = ["none", "T", "A", "V", "several"]
GOEMOTIONS = {  # batch -> (workbook, per-annotator columns), same annotators as the Basic-7 workbooks
    "first": ("First-2500/analysis_outputs/goemotion/goemotion_combined.xlsx", ["a1", "a2", "a3"]),
    "last": ("Last-2500/analysis_outputs_after2500/goemotion/goemotion_combined.xlsx", ["cp", "ljx", "wqf"]),
}


# ---- cumulative-logit mixed model ----------------------------------------------------------

class CLMM:
    """Ordinal regression with a random intercept per group. y: integer categories 0..K-1."""

    def __init__(self, y: np.ndarray, X: np.ndarray, groups: np.ndarray, names: list[str],
                 random_effect: bool = True, n_nodes: int = 40):
        self.y, self.X, self.names = y.astype(int), X.astype(float), names
        self.K, self.p = int(y.max()) + 1, X.shape[1]
        _, self.g = np.unique(groups, return_inverse=True)
        self.G = self.g.max() + 1
        self.random_effect = random_effect
        nodes, weights = np.polynomial.hermite.hermgauss(n_nodes if random_effect else 1)
        self.nodes, self.log_w = nodes, np.log(weights / np.sqrt(np.pi))

    def unpack(self, params):
        k = self.K - 1
        thetas = np.cumsum(np.concatenate([params[:1], np.exp(params[1:k])]))
        beta = params[k:k + self.p]
        sigma = np.exp(params[-1]) if self.random_effect else 0.0
        return thetas, beta, sigma

    def nll(self, params) -> float:
        thetas, beta, sigma = self.unpack(params)
        lin = (self.X @ beta)[:, None] + np.sqrt(2) * sigma * self.nodes[None, :]       # (n, J)
        cdf = expit(thetas[None, None, :] - lin[:, :, None])                              # (n, J, K-1)
        cdf = np.concatenate([np.zeros_like(cdf[..., :1]), cdf, np.ones_like(cdf[..., :1])], axis=-1)
        prob = np.take_along_axis(cdf, self.y[:, None, None] + 1, -1)[..., 0] - \
            np.take_along_axis(cdf, self.y[:, None, None], -1)[..., 0]
        per_group = np.zeros((self.G, len(self.nodes)))
        np.add.at(per_group, self.g, np.log(np.clip(prob, 1e-300, None)))
        return -logsumexp(per_group + self.log_w[None, :], axis=1).sum()

    def fit(self):
        counts = np.bincount(self.y, minlength=self.K).cumsum()[:-1] / len(self.y)
        start_thetas = np.log(counts / (1 - counts))
        start = np.concatenate([start_thetas[:1], np.log(np.diff(start_thetas)), np.zeros(self.p),
                                [np.log(0.5)] if self.random_effect else []])
        res = minimize(self.nll, start, method="BFGS", options={"gtol": 1e-6, "maxiter": 2000})
        # BFGS with finite-difference gradients often stops with "precision loss" at the optimum;
        # judge convergence by the gradient itself
        self.params = res.x
        self.max_grad = np.abs(approx_fprime(res.x, self.nll, 1e-6)).max()
        self.converged = bool(res.success or self.max_grad < 1e-2)
        self.cov = np.linalg.inv(approx_hess3(res.x, self.nll))
        self.loglik = -res.fun
        return self

    def coef(self) -> pd.DataFrame:
        k = self.K - 1
        b, se = self.params[k:k + self.p], np.sqrt(np.diag(self.cov))[k:k + self.p]
        z = b / se
        return pd.DataFrame({"term": self.names, "beta": b, "se": se, "OR": np.exp(b),
                             "OR low": np.exp(b - 1.96 * se), "OR high": np.exp(b + 1.96 * se),
                             "p": 2 * norm.sf(np.abs(z))})

    def contrast(self, a: str, b: str) -> tuple[float, float, float]:
        """beta_a - beta_b, its standard error and two-sided p (Wald)."""
        k = self.K - 1
        c = np.zeros(len(self.params))
        c[k + self.names.index(a)], c[k + self.names.index(b)] = 1, -1
        d, se = c @ self.params, np.sqrt(c @ self.cov @ c)
        return d, se, 2 * norm.sf(abs(d / se))

    def sigma(self) -> tuple[float, float, float]:
        """Random-intercept SD with a 95% interval (Wald on log sigma)."""
        log_s, se = self.params[-1], np.sqrt(self.cov[-1, -1])
        return np.exp(log_s), np.exp(log_s - 1.96 * se), np.exp(log_s + 1.96 * se)


# ---- data ----------------------------------------------------------------------------------

def load_goemotions() -> pd.DataFrame:
    """Per clip, the GoEmotions label sets of the 3 annotators. The two workbooks overlap at the batch
    boundary (rows left empty, or partly filled, in the second); joining on the batch as well keeps the
    row from the annotator group that also gave the clip its Basic-7 labels."""
    frames = []
    for batch, (path, annotators) in GOEMOTIONS.items():
        df = pd.read_excel(CHSIMS_DIR / path, dtype={"clip_id": str, "video_id": str})
        for i, a in enumerate(annotators):
            df[f"go{i + 1}"] = df[a].map(lambda s: frozenset(re.findall(r"'([^']+)'", str(s))))
        df["clip_key"] = df["clip_id"].map(clip_key)
        df["batch"] = batch
        frames.append(df[["video_id", "clip_key", "text", "batch", "go1", "go2", "go3"]])
    return pd.concat(frames, ignore_index=True)


def goemotions_entropy(sets: pd.DataFrame) -> np.ndarray:
    """Entropy of the pooled label counts of the 3 annotators (each annotator may pick several labels)."""
    labels = sorted(set().union(*sets.to_numpy().ravel()))
    counts = np.array([[sum(l in s for s in row) for l in labels] for row in sets.to_numpy()], float)
    return entropy(counts / counts.sum(axis=1, keepdims=True))


def load_data() -> pd.DataFrame:
    h = load_humans()
    h["conflict_type"] = np.select([h.n_deviating == 0, h.n_deviating >= 2, h.dev_T == 1, h.dev_A == 1],
                                   ["none", "several", "T", "A"], default="V")
    h["y"] = h["unique_labels"].astype(int) - 1
    h["last_batch"] = (h["batch"] == "last").astype(float)
    h["any_surprise"] = (h[["ann1", "ann2", "ann3"]] == "Surprise").any(axis=1)
    h["one_minus_ds_conf"] = 1 - h["ds_confidence"]

    meta_keys = h[["video_id", "clip_id", "text", "uid", "batch"]].assign(clip_key=h.clip_id.map(clip_key))
    go = meta_keys.merge(load_goemotions(), on=["video_id", "clip_key", "text", "batch"], how="inner", validate="one_to_one")
    go = go[go[["go1", "go2", "go3"]].apply(lambda r: all(len(s) for s in r), axis=1)]
    go["go_entropy"] = goemotions_entropy(go[["go1", "go2", "go3"]])
    return h.merge(go[["uid", "go_entropy"]], on="uid", how="left")


def design(df: pd.DataFrame, conflict: str, batch: bool = True) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    if conflict == "binary":
        cols.append(df.conflict.to_numpy(float)); names.append("conflict")
    else:
        for t in CONFLICT_TYPES[1:]:
            cols.append((df.conflict_type == t).to_numpy(float)); names.append(f"type {t}")
    cols.append(df.intensity.to_numpy(float)); names.append("intensity |label|")
    if batch:
        cols.append(df.last_batch.to_numpy()); names.append("last batch")
    return np.column_stack(cols), names


def fit(df: pd.DataFrame, conflict: str = "binary", batch: bool = True, random_effect: bool = True) -> CLMM:
    X, names = design(df, conflict, batch)
    return CLMM(df.y.to_numpy(), X, df.video_id.to_numpy(), names, random_effect).fit()


def fmt_or(row) -> str:
    return f"{row['OR']:.2f} [{row['OR low']:.2f}, {row['OR high']:.2f}]"


# ---- analyses ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/rq1_models"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_data()
    tables = {}

    # counts per conflict type
    tables["rq1_conflict_type_counts"] = (df.groupby("conflict_type").agg(
        clips=("uid", "size"), all_3_agree=("all_agree", "mean"), mean_unique_labels=("unique_labels", "mean"),
        mean_basic7_entropy=("human_entropy", "mean"), mean_goemotions_entropy=("go_entropy", "mean"))
        .reindex(CONFLICT_TYPES).reset_index())

    # model A (binary conflict) and model B (conflict type), each with and without the video effect
    rows = []
    models = {}
    for conflict in ("binary", "type"):
        for re_ in (True, False):
            m = fit(df, conflict, random_effect=re_)
            models[(conflict, re_)] = m
            for _, r in m.coef().iterrows():
                rows.append({"model": f"{'A: binary conflict' if conflict == 'binary' else 'B: conflict type'}",
                             "video random intercept": "yes" if re_ else "no", "term": r.term,
                             "OR [95% CI]": fmt_or(r), "p": r.p})
    coef = pd.DataFrame(rows)
    # Holm within the family "each conflict type vs none" (model B with the video effect)
    fam = (coef.model == "B: conflict type") & (coef["video random intercept"] == "yes") & coef.term.str.startswith("type")
    coef["p (Holm, conflict-type family)"] = np.nan
    coef.loc[fam, "p (Holm, conflict-type family)"] = multipletests(coef.loc[fam, "p"], method="holm")[1]
    tables["rq1_ordinal_mixed_models"] = coef

    fit_rows = []
    for (conflict, re_), m in models.items():
        row = {"model": "A: binary conflict" if conflict == "binary" else "B: conflict type",
               "video random intercept": "yes" if re_ else "no", "log-likelihood": m.loglik, "converged": m.converged}
        if re_:
            s, lo, hi = m.sigma()
            no_re = models[(conflict, False)]
            lr = 2 * (m.loglik - no_re.loglik)
            row.update({"video SD sigma [95% CI]": f"{s:.2f} [{lo:.2f}, {hi:.2f}]", "LR stat vs no video effect": lr,
                        # boundary test: chi-bar-square, half chi2(0) + half chi2(1)
                        "p (random effect)": 0.5 * 2 * norm.sf(np.sqrt(max(lr, 0)))})
        fit_rows.append(row)
    tables["rq1_model_fit"] = pd.DataFrame(fit_rows)

    # pairwise comparisons of conflict types (model B with the video effect), Holm-corrected
    mB = models[("type", True)]
    pairs = [(a, b) for i, a in enumerate(CONFLICT_TYPES[1:]) for b in CONFLICT_TYPES[1:][i + 1:]]
    rows = []
    for a, b in pairs:
        d, se, p = mB.contrast(f"type {a}", f"type {b}")
        rows.append({"comparison": f"{a}-deviating vs {b}-deviating",
                     "OR ratio [95% CI]": f"{np.exp(d):.2f} [{np.exp(d - 1.96 * se):.2f}, {np.exp(d + 1.96 * se):.2f}]",
                     "p": p})
    pw = pd.DataFrame(rows)
    pw["p (Holm)"] = multipletests(pw.p, method="holm")[1]
    tables["rq1_conflict_type_pairwise"] = pw

    # continuous disagreement measures: linear mixed models, random intercept per video
    rows = []
    measures = {"human_entropy": "entropy of the 3 Basic-7 votes", "go_entropy": "entropy of pooled GoEmotions labels",
                "one_minus_ds_conf": "1 - ds_confidence (Dawid-Skene)"}
    for col, label in measures.items():
        sub = df.dropna(subset=[col])
        for formula, kind in ((f"{col} ~ conflict + intensity + last_batch", "binary"),
                              (f"{col} ~ C(conflict_type, Treatment('none')) + intensity + last_batch", "type")):
            with warnings.catch_warnings():  # "MLE on the boundary": video variance ~0; reported below
                warnings.simplefilter("ignore", ConvergenceWarning)
                r = smf.mixedlm(formula, sub, groups=sub["video_id"]).fit(reml=True)
            ci = r.conf_int()
            for term in r.params.index:
                if term.startswith("conflict") or term.startswith("C(conflict_type"):
                    name = "conflict" if term == "conflict" else "type " + re.search(r"\[T\.(\w+)\]", term).group(1)
                    rows.append({"measure": label, "clips": len(sub), "model": kind, "term": name,
                                 "difference [95% CI]": f"{r.params[term]:+.3f} [{ci.loc[term, 0]:+.3f}, {ci.loc[term, 1]:+.3f}]",
                                 "p": r.pvalues[term], "video variance": float(r.cov_re.iloc[0, 0])})
    tables["rq1_continuous_measures"] = pd.DataFrame(rows)

    # robustness of model A
    subsets = {"all clips": (df, True)}
    for b in ("first", "last"):
        subsets[f"batch {b} only"] = (df[df.batch == b], False)
    for m in ("train", "valid", "test"):
        subsets[f"split {m} only"] = (df[df["mode"] == m], True)
    subsets["without clips with a Surprise vote"] = (df[~df.any_surprise], True)
    rows = []
    for name, (sub, batch) in subsets.items():
        m = fit(sub, "binary", batch=batch)
        r = m.coef().set_index("term").loc["conflict"]
        rows.append({"subset": name, "clips": len(sub), "videos": sub.video_id.nunique(),
                     "conflict OR [95% CI]": fmt_or(r), "p": r.p})
    tables["rq1_robustness"] = pd.DataFrame(rows)

    lines = ["# RQ1 models: conflict and annotator disagreement (CH-SIMS v2 + EmoS)\n",
             "Generated by `analysis/rq1_models.py`. Ordinal outcome: number of different Basic-7 labels among "
             "the 3 annotators; OR > 1 = more disagreement. Random intercept per video; 95% Wald intervals.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
