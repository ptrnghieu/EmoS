"""RQ1 + RQ2 analysis: cross-modal conflict, human disagreement and Qwen2.5-Omni-7B on CH-SIMS v2.

Inputs:  EmoS / CH-SIMS v2 labels (via chsims_emos) and the Kaggle run output
         (omni_runs/scores.jsonl, omni_runs/generations.jsonl).
Output:  <out>/report.md plus one CSV per table.

Confidence intervals are 95% percentile bootstraps that resample whole videos (clips from one
video are not independent).

Usage (from the repository root):
    python analysis/rq_analysis.py --runs results/omni_runs --out results/rq_analysis
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from chsims_emos import BASIC7, load_joined

LABELS = BASIC7
CONDITIONS = ["TAV", "T", "A", "V"]
POLARITY = {"Joy": 1, "Neutral": 0, "Anger": -1, "Disgust": -1, "Fear": -1, "Sadness": -1}  # Surprise: no polarity
RNG = np.random.default_rng(0)
N_BOOT = 1000


# ---- helpers -------------------------------------------------------------------------------

def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=-1)


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = (p + q) / 2
    kl = lambda a, b: (np.clip(a, 1e-12, 1) * np.log(np.clip(a, 1e-12, 1) / np.clip(b, 1e-12, 1))).sum(axis=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tvd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total variation distance; DistCE of Baan et al. (2022) when q is the human distribution."""
    return 0.5 * np.abs(p - q).sum(axis=-1)


def ent_ce(p: np.ndarray, human: np.ndarray) -> np.ndarray:
    """EntCE of Baan et al. (2022): model entropy minus human entropy (signed; > 0 = model less certain)."""
    return entropy(p) - entropy(human)


def rank_ok(p: np.ndarray, human: np.ndarray) -> np.ndarray:
    """Per-clip RankCS of Baan et al. (2022), with ties: True when every pair of labels that the
    humans rank differently is ranked the same way by the model. Baan et al. compare full argsorts and
    leave ties open (their footnote 4); with 3 votes over 7 labels at least 4 labels tie at 0, so
    pairs the humans tie are left unconstrained. Without ties this equals their definition."""
    hd = human[:, :, None] - human[:, None, :]
    md = p[:, :, None] - p[:, None, :]
    return ((hd <= 0) | (md > 0)).all(axis=(1, 2))


def loo_targets(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-annotator-out: for each clip and annotator i, the one-hot vote of i and the
    distribution of the other two votes. votes: (n, 3) label indices. Returns (n, 3, 7) arrays."""
    onehot = np.eye(len(LABELS))[votes]
    others = (onehot.sum(axis=1, keepdims=True) - onehot) / (votes.shape[1] - 1)
    return onehot, others


def spearman(x, y) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def auroc(score, positive) -> float:
    """P(score of a positive > score of a negative), ties count half (Mann-Whitney)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def boot(df: pd.DataFrame, stat, n: int = N_BOOT) -> tuple[float, float, float]:
    """Point estimate and 95% CI of stat(df), resampling videos with replacement."""
    groups = {v: g for v, g in df.groupby("video_id")}
    videos = np.array(list(groups))
    values = []
    for _ in range(n):
        sample = pd.concat([groups[v] for v in RNG.choice(videos, len(videos))], ignore_index=True)
        values.append(stat(sample))
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return stat(df), lo, hi


def fmt(est: tuple[float, float, float], digits: int = 3) -> str:
    return f"{est[0]:.{digits}f} [{est[1]:.{digits}f}, {est[2]:.{digits}f}]"


def markdown(table: pd.DataFrame) -> str:
    cell = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    rows = [list(table.columns)] + [[cell(v) for v in row] for row in table.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] +
                     ["| " + " | ".join(r) + " |" for r in rows[1:]])


def logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Logistic regression coefficients by Newton-Raphson (X includes the intercept column)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        w = p * (1 - p)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1]), X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def macro_f1(y_true, y_pred) -> float:
    f1s = []
    for label in LABELS:
        tp = ((y_pred == label) & (y_true == label)).sum()
        fp = ((y_pred == label) & (y_true != label)).sum()
        fn = ((y_pred != label) & (y_true == label)).sum()
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


# ---- data ----------------------------------------------------------------------------------

def odd_one_out(h: pd.DataFrame) -> pd.DataFrame:
    """Clips where exactly one modality's polarity differs from the multimodal label, with that modality."""
    single = h[h.n_deviating == 1].copy()
    single["odd"] = np.select([single.dev_T == 1, single.dev_A == 1], ["T", "A"], default="V")
    single["odd_sign"] = [row[f"sign_{row.odd}"] for _, row in single.iterrows()]
    single["majority_sign"] = np.sign(single["label"]).astype(int)
    return single


def following(pol: pd.Series, sub: pd.DataFrame, ci: bool = False) -> dict:
    """Share of answers (polarity `pol`) that follow the odd modality, split by which modality is odd."""
    sub = sub.assign(_pol=pol.to_numpy())
    sub = sub[sub._pol.notna()]
    row = {"answers": len(sub)}
    for m in "TAV":
        part = sub[sub.odd == m]
        rate = lambda d: (d._pol == d.odd_sign).mean()
        row[f"follows odd {m}"] = fmt(boot(part, rate)) if ci else rate(part)
    row["follows the other two"] = (sub._pol == sub.majority_sign).mean()
    row["answers Neutral"] = (sub._pol == 0).mean()
    return row


def load_humans() -> pd.DataFrame:
    df = load_joined()
    votes = df[["ann1", "ann2", "ann3"]].to_numpy()
    df["human_dist"] = list(np.stack([(votes == l).sum(axis=1) for l in LABELS], axis=1) / 3)
    df["human_entropy"] = entropy(np.stack(df["human_dist"]))
    df["all_agree"] = (df["unique_labels"] == 1).astype(int)
    df["intensity"] = df["label"].abs()
    for m in "TAV":
        df[f"sign_{m}"] = np.sign(df[f"label_{m}"]).astype(int)
    return df


def load_scores(path: Path) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[s["task"] == "score"]
    logits = s[[f"logit_{l}" for l in LABELS]].to_numpy()
    raw = np.exp(logits - s["lse"].to_numpy()[:, None])  # probability of each label's first token
    s["label_mass"] = raw.sum(axis=1)
    s["model_dist"] = list(raw / raw.sum(axis=1, keepdims=True))
    s["model_entropy"] = entropy(raw / raw.sum(axis=1, keepdims=True))
    s["model_maxp"] = (raw / raw.sum(axis=1, keepdims=True)).max(axis=1)
    s["pred"] = np.array(LABELS)[raw.argmax(axis=1)]
    return s[["uid", "condition", "label_mass", "model_dist", "model_entropy", "model_maxp", "pred"]]


def load_generations(path: Path) -> pd.DataFrame:
    g = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])

    def parse(text):
        match = re.search(r"\{.*\}", text or "", flags=re.S)
        try:
            return json.loads(match.group(0)) if match else None
        except (json.JSONDecodeError, ValueError):
            return None

    g["json"] = g["output"].map(parse)
    return g


def to_dist(counts) -> np.ndarray | None:
    if not isinstance(counts, dict):
        return None
    v = np.array([float(counts.get(l, 0) or 0) for l in LABELS])
    return v / v.sum() if v.sum() > 0 else None


# ---- analyses ------------------------------------------------------------------------------

def rq1(h: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    rows = []
    for name, sub in [("no conflict", h[h.conflict == 0]), ("conflict", h[h.conflict == 1])]:
        rows.append({"group": name, "clips": len(sub), "all 3 agree": sub.all_agree.mean(),
                     "mean human entropy": sub.human_entropy.mean()})
    out["rq1_agreement_by_conflict"] = pd.DataFrame(rows)

    def or_conflict(df):
        X = np.column_stack([np.ones(len(df)), df.conflict, df.intensity, (df.batch == "last").astype(float)])
        return float(np.exp(logistic(X, df.all_agree.to_numpy(float))[1]))

    def diff(df):
        return df[df.conflict == 1].all_agree.mean() - df[df.conflict == 0].all_agree.mean()

    out["rq1_effect"] = pd.DataFrame([
        {"quantity": "difference in P(all 3 agree), conflict minus no conflict", "estimate [95% CI]": fmt(boot(h, diff))},
        {"quantity": "odds ratio of all 3 agreeing for conflict, adjusted for intensity |label| and annotator batch",
         "estimate [95% CI]": fmt(boot(h, or_conflict))},
    ])

    # how does each annotator resolve a conflict? On clips where exactly one modality disagrees with
    # the other two, count how often the annotator's label has the polarity of the odd one out.
    single = odd_one_out(h)
    rows = []
    for batch, sub in single.groupby("batch"):
        for i in (1, 2, 3):
            pol = sub[f"ann{i}"].map(POLARITY)
            rows.append({"annotator": f"{batch}-{i}", **following(pol, sub)})
    out["rq1_annotator_follows_odd_modality"] = pd.DataFrame(rows)
    return out


def rq2(h: pd.DataFrame, s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    df = s.merge(h, on="uid")

    rows = []
    for cond in CONDITIONS:
        for name, sub in [("all", df[df.condition == cond]), ("no conflict", df[(df.condition == cond) & (df.conflict == 0)]),
                          ("conflict", df[(df.condition == cond) & (df.conflict == 1)])]:
            rows.append({"condition": cond, "clips": name, "n": len(sub),
                         "accuracy vs ds_label": (sub.pred == sub.ds_label).mean(),
                         "macro-F1": macro_f1(sub.ds_label.to_numpy(), sub.pred.to_numpy()),
                         "share predicted Neutral": (sub.pred == "Neutral").mean(),
                         "mean label mass": sub.label_mass.mean()})
    out["rq2_accuracy"] = pd.DataFrame(rows)

    # H2b: does the model become less certain on conflict clips, as humans do?
    rows = []
    for cond in CONDITIONS:
        sub = df[df.condition == cond]
        model_d = lambda d: d[d.conflict == 1].model_entropy.mean() - d[d.conflict == 0].model_entropy.mean()
        human_d = lambda d: d[d.conflict == 1].human_entropy.mean() - d[d.conflict == 0].human_entropy.mean()
        rows.append({"condition": cond,
                     "model entropy, no conflict": sub[sub.conflict == 0].model_entropy.mean(),
                     "model entropy, conflict": sub[sub.conflict == 1].model_entropy.mean(),
                     "model change [95% CI]": fmt(boot(sub, model_d)),
                     "human change [95% CI]": fmt(boot(sub, human_d)),
                     "model max prob, no conflict": sub[sub.conflict == 0].model_maxp.mean(),
                     "model max prob, conflict": sub[sub.conflict == 1].model_maxp.mean()})
    out["rq2_uncertainty_by_conflict"] = pd.DataFrame(rows)

    # alignment of model uncertainty with human disagreement, clip by clip
    rows = []
    for cond in CONDITIONS:
        sub = df[df.condition == cond]
        rows.append({"condition": cond,
                     "Spearman(model entropy, human entropy) [95% CI]": fmt(boot(sub, lambda d: spearman(d.model_entropy, d.human_entropy))),
                     "AUROC: model entropy flags human disagreement [95% CI]": fmt(boot(sub, lambda d: auroc(d.model_entropy, d.all_agree == 0)))})
    out["rq2_alignment"] = pd.DataFrame(rows)

    # distance to the human distribution, with reference points
    tav = df[df.condition == "TAV"].copy()
    H = np.stack(tav.human_dist)
    prior = H.mean(axis=0)
    refs = {
        "model (T+A+V)": np.stack(tav.model_dist),
        "model's own answer, one-hot": np.stack([np.eye(len(LABELS))[LABELS.index(l)] for l in tav.pred]),
        "majority label, one-hot (a perfectly accurate but fully confident model)":
            np.stack([np.eye(len(LABELS))[LABELS.index(l)] for l in tav.ds_label]),
        "label prior of the dataset, same for every clip": np.tile(prior, (len(tav), 1)),
        "uniform": np.full_like(H, 1 / len(LABELS)),
    }
    rows = []
    for name, P in refs.items():
        tav["_jsd"] = jsd(P, H)
        rows.append({"distribution": name,
                     "JSD to humans, all": tav._jsd.mean(),
                     "no conflict": tav[tav.conflict == 0]._jsd.mean(),
                     "conflict": tav[tav.conflict == 1]._jsd.mean()})
    out["rq2_jsd_to_humans"] = pd.DataFrame(rows)

    # calibration to human disagreement (Baan et al., 2022): DistCE = TVD to the 3-vote distribution,
    # EntCE = model entropy minus human entropy, RankCS = share of clips with the human label ranking
    rows = []
    for cond in CONDITIONS:
        for name, sub in [("all", df[df.condition == cond]), ("no conflict", df[(df.condition == cond) & (df.conflict == 0)]),
                          ("conflict", df[(df.condition == cond) & (df.conflict == 1)])]:
            P, Hs = np.stack(sub.model_dist), np.stack(sub.human_dist)
            e = ent_ce(P, Hs)
            rows.append({"condition": cond, "clips": name, "n": len(sub), "mean DistCE": tvd(P, Hs).mean(),
                         "mean |EntCE|": np.abs(e).mean(), "mean EntCE (signed)": e.mean(),
                         "RankCS": rank_ok(P, Hs).mean()})
    out["rq2_human_calibration"] = pd.DataFrame(rows)

    # ceiling: hold out one annotator and predict the other two. The held-out person's own vote is
    # the human reference; the model predicts the same two-vote targets, so the numbers compare.
    votes = np.stack([tav[f"ann{i}"].map(LABELS.index).to_numpy() for i in (1, 2, 3)], axis=1)
    held_out, others = loo_targets(votes)
    P = np.stack(tav.model_dist)
    predictors = {
        "one held-out annotator (human ceiling)": held_out,
        "model (T+A+V)": np.repeat(P[:, None], 3, axis=1),
        "model's own answer, one-hot": np.repeat(np.eye(len(LABELS))[[LABELS.index(l) for l in tav.pred]][:, None], 3, axis=1),
        "label prior of the dataset": np.broadcast_to(prior, others.shape),
    }
    conflict = tav.conflict.to_numpy() == 1
    rows = []
    for name, Q in predictors.items():
        j, t = jsd(Q, others).mean(axis=1), tvd(Q, others).mean(axis=1)
        rows.append({"predictor of the other two annotators": name, "JSD, all": j.mean(), "JSD, no conflict": j[~conflict].mean(),
                     "JSD, conflict": j[conflict].mean(), "DistCE, all": t.mean()})
    out["rq2_ceiling_leave_one_out"] = pd.DataFrame(rows)

    # H2a: on clips where exactly one modality disagrees with the other two, how often does the
    # answer take the odd modality's polarity? Humans (all six annotators pooled) vs the model.
    # Two subsets: all such clips, and clips where the odd modality has the opposite non-zero polarity
    # (a real flip). In the first, an odd modality is often neutral (0), so a model that answers
    # Neutral a lot "follows" it without preferring that modality; the second removes that.
    single = odd_one_out(h)
    flips = single[(single.odd_sign != 0) & (single.majority_sign != 0)]
    rows = []
    for subset, clips_ in [("all clips with one odd modality", single), ("odd modality flips polarity", flips)]:
        pooled = pd.concat([clips_.assign(_pol=clips_[f"ann{i}"].map(POLARITY)) for i in (1, 2, 3)], ignore_index=True)
        rows.append({"subset": subset, "who": "humans (6 annotators pooled)", **following(pooled["_pol"], pooled, ci=True)})
        for cond in CONDITIONS:
            pred = df[df.condition == cond].set_index("uid").pred
            sub = clips_.assign(_pol=clips_.uid.map(pred).map(POLARITY))
            rows.append({"subset": subset, "who": f"model, {cond}", **following(sub["_pol"], sub, ci=True)})
    out["rq2_follows_odd_modality"] = pd.DataFrame(rows)

    # conflict clips push both humans and the model towards Neutral; compare how far
    rows = []
    tav_all = df[df.condition == "TAV"]
    for name, sub in [("no conflict", tav_all[tav_all.conflict == 0]), ("conflict", tav_all[tav_all.conflict == 1])]:
        rows.append({"clips": name, "n": len(sub),
                     "humans: majority label is Neutral": (sub.ds_label == "Neutral").mean(),
                     "humans: share of Neutral votes": np.stack(sub.human_dist)[:, LABELS.index("Neutral")].mean(),
                     "model T+A+V: predicts Neutral": (sub.pred == "Neutral").mean()})
    out["rq2_neutral_shift"] = pd.DataFrame(rows)

    # model-internal: does the T+A+V answer agree with the model's own single-modality answers?
    wide = df.pivot(index="uid", columns="condition", values="pred").join(h.set_index("uid")[["conflict"]])
    rows = []
    for name, sub in [("all", wide), ("no conflict", wide[wide.conflict == 0]), ("conflict", wide[wide.conflict == 1])]:
        rows.append({"clips": name, "n": len(sub), **{f"T+A+V = {m}-only": (sub.TAV == sub[m]).mean() for m in "TAV"}})
    out["rq2_model_internal_following"] = pd.DataFrame(rows)
    return out


def generation_tables(h: pd.DataFrame, g: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    hi = h.set_index("uid")

    v = g[g.task == "verbalized"].copy()
    v["dist"] = v.json.map(to_dist)
    v = v[v.dist.notna()]
    keys = v.dist.map(lambda d: tuple(np.round(d, 3)))
    vv = v.join(hi, on="uid")
    P, H = np.stack(vv.dist), np.stack(vv.human_dist)
    out["gen_verbalized"] = pd.DataFrame([{
        "clips": len(v),
        "share of clips with the single most common output": keys.value_counts().iloc[0] / len(v),
        "distinct outputs": keys.nunique(),
        "mean entropy": entropy(P).mean(),
        "JSD to humans": jsd(P, H).mean(),
        "Spearman(entropy, human entropy)": spearman(entropy(P), vv.human_entropy),
    }])

    p = g[g.task == "per_modality"].copy()
    p = p[p.json.map(lambda j: isinstance(j, dict))].join(hi, on="uid")
    p["flag"] = p.json.map(lambda j: bool(j.get("conflict")) if isinstance(j.get("conflict"), (bool, str)) else None)
    p["dist"] = p.json.map(lambda j: to_dist(j.get("distribution")))
    for m, key in zip("TAV", ("text", "audio", "video")):
        p[f"label_{key}"] = p.json.map(lambda j, k=key: (j.get(k) or {}).get("label") if isinstance(j.get(k), dict) else None)
        p[f"pol_{m}"] = p[f"label_{key}"].map(POLARITY)
    labels = p[["label_text", "label_audio", "label_video"]]
    rows = []
    for name, sub in [("all", p), ("no conflict", p[p.conflict == 0]), ("conflict", p[p.conflict == 1])]:
        lab = labels.loc[sub.index]
        dist = sub.dist.dropna()
        row = {"clips": name, "n": len(sub),
               "model says conflict": sub.flag.mean(),
               "same label for text, audio and video": (lab.nunique(axis=1) == 1).mean(),
               "distribution puts >= 90 on one label": np.mean([d.max() >= 0.9 for d in dist]),
               "JSD to humans": jsd(np.stack(dist), np.stack(sub.loc[dist.index].human_dist)).mean()}
        for m in "TAV":
            keep = sub[f"pol_{m}"].notna()
            row[f"{m} label polarity matches human {m} label"] = (sub.loc[keep, f"pol_{m}"] == sub.loc[keep, f"sign_{m}"]).mean()
        rows.append(row)
    out["gen_per_modality"] = pd.DataFrame(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/omni_runs"))
    parser.add_argument("--out", type=Path, default=Path("results/rq_analysis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    humans = load_humans()
    tables = rq1(humans)
    tables.update(rq2(humans, load_scores(args.runs / "scores.jsonl")))
    if (args.runs / "generations.jsonl").exists():  # only the Qwen run has the generation tasks
        tables.update(generation_tables(humans, load_generations(args.runs / "generations.jsonl")))

    with open(args.runs / "scores.jsonl", encoding="utf-8") as f:
        model = json.loads(f.readline())["model"]
    lines = [f"# RQ1 + RQ2 results ({model}, CH-SIMS v2 + EmoS)\n",
             "Generated by `analysis/rq_analysis.py`. CIs: 95% video-level bootstrap.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
