"""Stage 1, outer evaluation on CREMA-D: does per-condition supervision make the AV entropy track human disagreement
better than AV soft labels alone? Criteria fixed before the runs (proposal, 24/10/2026).

Runs: results/stage1/stage1_<arm>_f<fold>_s<seed>/ (scores.jsonl, state.json) for arm in m3, s1, s1_noI, s1_avlab,
3 outer folds over actors. Each clip is scored by the model of the fold that held its actor out, so the three folds
together cover every clip once.

U^AV = H(q_AV) / log 6 (the model's AV prediction); A^AV = normalised entropy of the AV raters' votes.

    primary      Spearman(U^AV, A^AV): per fold and pooled over the folds; each arm minus m3, paired bootstrap
                 over actors (pooled) or over the fold's actors (per fold)
    stop rule    s1_noI: pooled difference to m3 with its 95% CI above 0 AND positive in all three folds
    secondary    AUROC for the clips in the top quarter of A^AV, threshold from each fold's training actors;
                 Spearman within each intended emotion and within each sentence (with clip counts);
                 the V and A readings against the video-only and audio-only raters
    references   two priors fitted on each fold's training actors and scored on its test actors:
                 emotion class + human perceived intensity (I^V, I^A), and the mean A^AV per (sentence, emotion)

Usage (from the repository root; downloads the CREMA-D rating files):
    python analysis/stage1_outer.py --runs results/stage1 --out results/stage1_outer
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import counts
from cremad_ambiguity_diagnostic import intensity
from preprocess_cremad import LABELS, RAW, clip_table, fetch
from rq3_analysis import N_BOOT, Bootstrap, softmax
from rq_analysis import auroc, markdown, spearman

K = len(LABELS)
ARMS = ["m3", "s1_avlab", "s1_noI", "s1"]
NAME = re.compile(r"stage1_(?P<arm>m3|s1|s1_noI|s1_avlab)_f(?P<fold>\d)_s(?P<seed>\d)$")


def norm_entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0), 1) / np.log(K)


def load_run(path: Path) -> tuple[dict, pd.DataFrame]:
    state = json.loads((path / "state.json").read_text())
    s = pd.DataFrame([json.loads(l) for l in open(path / "scores.jsonl", encoding="utf-8") if l.strip()])
    s = s[s.get("error", pd.Series(index=s.index, dtype=object)).isna()].drop_duplicates(["uid", "condition"], keep="last")
    return state, s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/stage1"))
    parser.add_argument("--out", type=Path, default=Path("results/stage1_outer"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table()
    df = df.merge(intensity(pd.read_csv(io.BytesIO(fetch(f"{RAW}/finishedResponses.csv")), low_memory=False)),
                  left_on="uid", right_index=True, how="left").set_index("uid")
    human = {c: norm_entropy(counts(df, k) / np.clip(counts(df, k).sum(1, keepdims=True), 1, None))
             for c, k in (("AV", "av"), ("V", "v"), ("A", "a"))}
    df["A_AV"], df["A_V"], df["A_A"] = human["AV"], human["V"], human["A"]

    # ---- load and check runs ------------------------------------------------------------------------
    runs, checks = {}, []
    for path in sorted(p for p in args.runs.iterdir() if NAME.match(p.name)):
        m = NAME.match(path.name)
        state, s = load_run(path)
        n_by = s.groupby("condition")["uid"].nunique().to_dict()
        expected = state.get("sizes", [None, None])[1]
        ok = (state.get("phase") == "outer" and bool(state.get("train_done"))
              and all(n_by.get(c) == expected for c in ("AV", "V", "A")))
        checks.append({"run": path.name, "lambdas V/A/I": state.get("lambdas"), "train_done": state.get("train_done"),
                       "expected clips": expected, **{f"scored {c}": n_by.get(c, 0) for c in ("AV", "V", "A")},
                       "OOM in training": state.get("oom"), "complete": ok})
        if ok:
            u = {}
            for c in ("AV", "V", "A"):
                part = s[s.condition == c].set_index("uid")
                u[c] = pd.Series(norm_entropy(softmax(part[[f"logit_{l}" for l in LABELS]].to_numpy(float))), index=part.index)
            runs[(m["arm"], int(m["fold"]), int(m["seed"]))] = u
    tables = {"run_checks": pd.DataFrame(checks)}
    folds = sorted({f for _, f, _ in runs})
    seeds = sorted({s for _, _, s in runs})
    fold_uids = {f: runs[next(k for k in runs if k[1] == f)]["AV"].index for f in folds}
    fold_of = pd.Series({u: f for f, us in fold_uids.items() for u in us})

    def pooled(arm, seed, c="AV"):
        if not all((arm, f, seed) in runs for f in folds):
            return None
        return pd.concat([runs[(arm, f, seed)][c] for f in folds])

    def m3_for(seed):
        return pooled("m3", seed) if pooled("m3", seed) is not None else pooled("m3", 0)

    # ---- references fitted on each fold's training actors ------------------------------------------
    ref = {"prior: emotion class + human intensity (I^V, I^A)": pd.Series(dtype=float),
           "prior: mean A^AV per (sentence, emotion)": pd.Series(dtype=float)}
    for f in folds:
        test = df.loc[fold_uids[f]]
        train = df.drop(index=fold_uids[f])
        X = lambda d: np.column_stack([np.ones(len(d)), pd.get_dummies(d["intended"]).reindex(columns=sorted(df.intended.unique()), fill_value=0).to_numpy(float)[:, 1:],
                                       d["I_V"].to_numpy(float), d["I_A"].to_numpy(float)])
        beta = np.linalg.lstsq(X(train), train["A_AV"].to_numpy(), rcond=None)[0]
        ref["prior: emotion class + human intensity (I^V, I^A)"] = pd.concat(
            [ref["prior: emotion class + human intensity (I^V, I^A)"], pd.Series(X(test) @ beta, index=test.index)])
        cell = train.groupby(["sentence_code", "intended"])["A_AV"].mean()
        pred = [cell.get((sc, em), train["A_AV"].mean()) for sc, em in zip(test.sentence_code, test.intended)]
        ref["prior: mean A^AV per (sentence, emotion)"] = pd.concat(
            [ref["prior: mean A^AV per (sentence, emotion)"], pd.Series(pred, index=test.index)])
    threshold = {f: df.drop(index=fold_uids[f])["A_AV"].quantile(0.75) for f in folds}

    uids = fold_of.index
    sub = df.loc[uids]
    y = sub["A_AV"].to_numpy()
    high = np.array([a >= threshold[f] for a, f in zip(y, fold_of.loc[uids])])
    boot = Bootstrap(sub["video_id"].to_numpy(), N_BOOT)

    def score_rows(name, u):
        u = u.reindex(uids).to_numpy()
        row = {"predictor": name, "Spearman(U^AV, A^AV), pooled": boot(lambda i: spearman(u[i], y[i])),
               "AUROC top-25% (train threshold)": boot(lambda i: auroc(u[i], high[i]))}
        for f in folds:
            sel = (fold_of.loc[uids] == f).to_numpy()
            row[f"fold {f}"] = round(spearman(u[sel], y[sel]), 3)
        return row

    # ---- primary --------------------------------------------------------------------------------------
    rows, diffs = [], []
    for seed in seeds:
        for arm in ARMS:
            u = pooled(arm, seed)
            if u is None:
                continue
            rows.append({"seed": seed, **score_rows(arm, u)})
            if arm == "m3":
                continue
            base = m3_for(seed).reindex(uids).to_numpy()
            ua = u.reindex(uids).to_numpy()
            d = {"comparison": f"{arm} minus m3", "seed": seed,
                 "pooled [95% CI]": boot(lambda i: spearman(ua[i], y[i]) - spearman(base[i], y[i]))}
            point = spearman(ua, y) - spearman(base, y)
            lo = np.percentile([spearman(ua[i], y[i]) - spearman(base[i], y[i]) for i in boot.samples], 2.5)
            per = []
            for f in folds:
                sel = np.flatnonzero((fold_of.loc[uids] == f).to_numpy())
                fb = Bootstrap(sub["video_id"].to_numpy()[sel], N_BOOT)
                d[f"fold {f} [95% CI]"] = fb(lambda i, sel=sel: spearman(ua[sel][i], y[sel][i]) - spearman(base[sel][i], y[sel][i]))
                per.append(spearman(ua[sel], y[sel]) - spearman(base[sel], y[sel]))
            if arm == "s1_noI":
                d["stop rule met (CI above 0 and all folds positive)"] = bool(lo > 0 and all(x > 0 for x in per))
            d["_point"] = point
            diffs.append(d)
    for name, u in ref.items():
        rows.append({"seed": "", **score_rows(name, u)})
    tables["primary_spearman"] = pd.DataFrame(rows)
    tables["primary_differences_vs_m3"] = pd.DataFrame(diffs).drop(columns="_point", errors="ignore")

    # ---- within emotion and within sentence (seed 0) --------------------------------------------------
    for key, label in (("intended", "emotion"), ("sentence_code", "sentence")):
        rows = []
        for g, part in sub.groupby(key):
            row = {label: g, "clips": len(part)}
            for arm in ARMS:
                u = pooled(arm, 0)
                if u is not None:
                    row[arm] = round(spearman(u.reindex(part.index).to_numpy(), part["A_AV"].to_numpy()), 3)
            rows.append(row)
        tables[f"within_{label}_seed0"] = pd.DataFrame(rows)

    # ---- V and A readings against their own raters (seed 0) -------------------------------------------
    rows = []
    for arm in ARMS:
        row = {"arm": arm}
        for c in ("V", "A"):
            u = pooled(arm, 0, c)
            if u is not None:
                uu = u.reindex(uids).to_numpy()
                h = sub[f"A_{c}"].to_numpy()
                row[f"Spearman(U^{c}, A^{c})"] = boot(lambda i, uu=uu, h=h: spearman(uu[i], h[i]))
        if len(row) > 1:
            rows.append(row)
    tables["readings_vs_own_raters_seed0"] = pd.DataFrame(rows)

    lines = ["# Stage 1 outer evaluation (CREMA-D, 3 folds over actors)\n",
             f"Generated by `analysis/stage1_outer.py`. {len(uids)} clips, {sub.actor.nunique()} actors. Complete runs: "
             f"{', '.join(f'{a} f{f} s{s}' for a, f, s in sorted(runs))}. CIs: 95% bootstrap over actors ({N_BOOT} resamples).\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
