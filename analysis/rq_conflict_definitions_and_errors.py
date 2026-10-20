"""Two follow-up analyses after the related-work review (20/10/2026).

A. Does the RQ1 result depend on how conflict is defined? Our definition: the per-modality polarity
   labels disagree with each other. Du et al. (2025, 2026) compare each modality's label with the
   multimodal label instead. Same cumulative-logit mixed model as rq1_models.py (intensity, annotator
   batch, random intercept per video), with four conflict measures:
       ours            any two per-modality polarities differ
       Du, hard        any modality's polarity differs from the multimodal label's
       Du, count       number of modalities whose polarity differs from the multimodal label's (0-3)
       Du, distance    mean |label_m - label| over T, A, V (CH-SIMS v2 labels lie in [-1, 1]), per SD

B. Are the models' "errors" on conflict clips answers a human gave? For each model (T+A+V): accuracy
   against ds_label, the share of answers that match at least one of the 3 annotators ("plausible"),
   and the share of errors that are plausible, on clips without and with conflict. Unanimous clips
   cannot have a plausible error, and conflict clips are less often unanimous, so the last table
   repeats the comparison on clips where the annotators disagree. Emotion-LLaMA answers over its six
   labels (no Disgust). 95% video-level bootstrap intervals.

Usage (from the repository root; needs scipy and statsmodels):
    python analysis/rq_conflict_definitions_and_errors.py --out results/rq_conflict_definitions_and_errors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rq1_models import CLMM, fmt_or, load_data
from rq2_compare_models import ALL_RUNS, SIX, SIX_LABEL_RUNS
from rq3_analysis import N_BOOT, Bootstrap, load_logits
from rq_analysis import LABELS, markdown

LORA_RUNS = {"LoRA hard (seed 0)": "results/rq3/rq3_hard/scores.jsonl",
             "LoRA soft (seed 0)": "results/rq3/rq3_soft/scores.jsonl"}


def conflict_measures(df: pd.DataFrame) -> dict[str, np.ndarray]:
    distance = np.mean([np.abs(df[f"label_{m}"] - df["label"]) for m in "TAV"], axis=0)
    return {"ours: per-modality polarities differ": df.conflict.to_numpy(float),
            "Du, hard: a modality's polarity differs from the multimodal label": (df.n_deviating >= 1).to_numpy(float),
            "Du, count: modalities differing from the multimodal label (per modality)": df.n_deviating.to_numpy(float),
            "Du, distance: mean |label_m - label| (per SD)": distance / distance.std()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/rq_conflict_definitions_and_errors"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_data()
    tables = {}

    # ---- A. conflict definitions -----------------------------------------------------------------
    rows = []
    for name, x in conflict_measures(df).items():
        X = np.column_stack([x, df.intensity.to_numpy(float), df.last_batch.to_numpy()])
        m = CLMM(df.y.to_numpy(), X, df.video_id.to_numpy(), ["conflict", "intensity |label|", "last batch"], True).fit()
        r = m.coef().set_index("term").loc["conflict"]
        binary = set(np.unique(x)) <= {0, 1}
        rows.append({"conflict measure": name,
                     "clips flagged, or mean": f"{int(x.sum())} clips" if binary else f"mean {x.mean():.2f}",
                     "OR for more disagreement [95% CI]": fmt_or(r), "p": r.p})
    tables["a_conflict_definitions_rq1"] = pd.DataFrame(rows)
    tables["a_ours_vs_du_hard_crosstab"] = pd.crosstab(
        df.conflict.map({0: "ours: no conflict", 1: "ours: conflict"}),
        (df.n_deviating >= 1).map({False: "Du hard: no", True: "Du hard: yes"})).reset_index()

    # ---- B. errors that match an annotator ------------------------------------------------------------
    votes = df[["ann1", "ann2", "ann3"]].to_numpy()
    gold = df.ds_label.to_numpy()
    conflict = df.conflict.to_numpy() == 1
    split = df.all_agree.to_numpy() == 0
    test = df["mode"].to_numpy() == "test"

    def answers(path: str, labels: list[str]) -> np.ndarray:
        z = load_logits(path)[[f"logit_{l}" for l in labels]]
        z = z.loc[[(u, "TAV") for u in df.uid if (u, "TAV") in z.index]]
        pred = pd.Series(np.array(labels)[z.to_numpy(float).argmax(1)], index=[u for u, _ in z.index])
        return pred.reindex(df.uid).to_numpy()

    preds = {name: answers(path, LABELS) for name, path in ALL_RUNS.items() if Path(path).exists()}
    preds.update({name: answers(path, SIX) for name, path in SIX_LABEL_RUNS.items()})
    lora = {name: answers(path, LABELS) for name, path in LORA_RUNS.items() if Path(path).exists()}
    if lora:  # LoRA runs: test split only (their valid-split scores served temperature scaling)
        preds["Qwen2.5-Omni-7B zero-shot (test split)"] = np.where(test, preds["Qwen2.5-Omni-7B"], None)
        preds.update({name: np.where(test, pred, None) for name, pred in lora.items()})

    rows, rows_split = [], []
    for name, pred in preds.items():
        have = np.array([p is not None and p == p for p in pred])  # LoRA runs cover the test split only
        scope = "test split" if not have.all() else "all clips"
        if scope == "test split":
            assert (have == test).all(), name
        idx = np.flatnonzero(have)
        sub = df.iloc[idx]
        boot = Bootstrap(sub.video_id.to_numpy(), N_BOOT)
        p, g, v, c, s = pred[idx], gold[idx], votes[idx], conflict[idx], split[idx]
        correct = p == g
        plausible = (v == p[:, None]).any(1)
        err_plausible = lambda i, sel: plausible[i][sel[i] & ~correct[i]].mean()
        row = {"model (T+A+V)": name, "clips": scope}
        for label, sel in (("no conflict", ~c), ("conflict", c)):
            row[f"accuracy, {label}"] = round(float(correct[sel].mean()), 3)
            row[f"matches an annotator, {label}"] = round(float(plausible[sel].mean()), 3)
            row[f"errors matching an annotator, {label}"] = boot(lambda i, sel=sel: err_plausible(i, sel))
        row["errors matching, conflict minus no conflict"] = boot(lambda i: err_plausible(i, c) - err_plausible(i, ~c))
        rows.append(row)
        row = {"model (T+A+V)": name, "clips": scope + ", annotators not unanimous"}
        for label, sel in (("no conflict", ~c & s), ("conflict", c & s)):
            row[f"errors, {label}"] = int((sel & ~correct).sum())
            row[f"errors matching an annotator, {label}"] = boot(lambda i, sel=sel: err_plausible(i, sel))
        row["conflict minus no conflict"] = boot(lambda i: err_plausible(i, c & s) - err_plausible(i, ~c & s))
        rows_split.append(row)
    tables["b_errors_matching_an_annotator"] = pd.DataFrame(rows)
    tables["b_errors_matching_an_annotator_non_unanimous"] = pd.DataFrame(rows_split)

    lines = ["# Conflict definitions (RQ1) and model errors that match an annotator (RQ2, RQ3)\n",
             f"Generated by `analysis/rq_conflict_definitions_and_errors.py`. {len(df)} clips "
             f"({int(conflict.sum())} with conflict by our definition). Ordinal mixed model as in rq1_models.py; "
             f"bootstrap: 95%, by video, {N_BOOT} resamples.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
