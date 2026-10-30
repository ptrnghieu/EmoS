"""Stage 3, outer evaluation on CREMA-D: does the human-ambiguity-aware method beat m3, and how much of it is the
4 fps input? Criteria fixed before the runs (approved 28/10/2026).

Runs (3 outer folds over actors, seed 0; each clip scored by the model of the fold that held its actor out):
    m3_A0      results/stage1/stage1_m3_f<k>_s0       AV soft labels, 2 fps (Stage 1 baseline)
    s1_noI_A0  results/stage1/stage1_s1_noI_f<k>_s0   AV + V + A soft labels, 2 fps (best Stage 1 arm)
    m3_A1      results/stage3/stage3_m3_A1_f<k>_s0    AV soft labels, 4 fps (the controlled baseline)
    s3_A1      results/stage3/stage3_s3_A1_f<k>_s0    the method; any further stage3 arm is read the same way

Readouts: U^AV = normalised entropy of q_AV for every arm; for arms with an ambiguity head, its score `amb_score`
(the method's pre-registered prediction). Target A^AV = normalised entropy of the AV raters' votes.

    primary      Spearman(readout, A^AV), pooled over the folds
    secondary    partial Spearman given emotion and given emotion x sentence; AUROC for the top quarter of A^AV
                 (threshold from the other folds' clips); per fold
    stop rule    s3 amb_score minus m3_A1 entropy: Spearman and partial given emotion x sentence both above 0 with
                 their 95% paired CI (bootstrap over actors), and the Spearman difference positive in all three folds
    also         the V reading against the video-only raters (U^V vs A^V), to see what 4 fps does to the video reading

Usage (from the repository root; downloads the CREMA-D rating files):
    python analysis/stage3_outer.py --out results/stage3_outer
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import counts
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import N_BOOT, Bootstrap, softmax
from rq_analysis import auroc, markdown, spearman
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman

SOURCES = {"stage1": re.compile(r"stage1_(?P<arm>m3|s1_noI)_f(?P<fold>\d)_s0$"),
           "stage3": re.compile(r"stage3_(?P<arm>[a-z0-9_]+?)_(?P<view>A1)_f(?P<fold>\d)_s0$")}


def load(path: Path):
    state = json.loads((path / "state.json").read_text())
    s = pd.DataFrame([json.loads(l) for l in open(path / "scores.jsonl", encoding="utf-8") if l.strip()])
    s = s[s.get("error", pd.Series(index=s.index, dtype=object)).isna()].drop_duplicates(["uid", "condition"], keep="last")
    return state, s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--stage3", type=Path, default=Path("results/stage3"))
    parser.add_argument("--out", type=Path, default=Path("results/stage3_outer"))
    parser.add_argument("--boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table().set_index("uid")
    for c, k in (("AV", "av"), ("V", "v")):
        C = counts(df, k)
        df[f"A_{c}"] = norm_entropy(C / np.clip(C.sum(1, keepdims=True), 1, None))
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))

    # ---- load and check -------------------------------------------------------------------------------
    parts, checks = {}, []
    for source, folder in (("stage1", args.stage1), ("stage3", args.stage3)):
        for path in sorted(p for p in folder.iterdir() if SOURCES[source].match(p.name)):
            m = SOURCES[source].match(path.name)
            arm = f"{m['arm']}_{'A0' if source == 'stage1' else m['view']}"
            state, s = load(path)
            n_by = s.groupby("condition")["uid"].nunique().to_dict()
            expected = state["sizes"][1]
            ok = state.get("phase") == "outer" and bool(state.get("train_done")) and n_by.get("AV") == expected
            checks.append({"run": path.name, "arm": arm, "fold": int(m["fold"]), "expected": expected,
                           **{f"scored {c}": n_by.get(c, 0) for c in ("AV", "V")}, "complete": ok})
            if ok:
                parts.setdefault(arm, []).append(s)
    tables = {"run_checks": pd.DataFrame(checks)}
    arms = {a: pd.concat(p) for a, p in parts.items() if len(p) == N_OUTER}
    assert arms, "no arm with all three outer folds"
    common = sorted(set.intersection(*(set(s[s.condition == "AV"].uid) for s in arms.values())))
    sub = df.loc[common]
    y, yv = sub.A_AV.to_numpy(), sub.A_V.to_numpy()
    emo, emo_sent = sub.intended.to_numpy(), (sub.intended + sub.sentence_code).to_numpy()
    folds = sub.fold.to_numpy()
    high = np.zeros(len(common), bool)
    for f in range(N_OUTER):
        high[folds == f] = y[folds == f] >= np.quantile(df.A_AV.to_numpy()[df.fold.to_numpy() != f], 0.75)
    boot = Bootstrap(sub.actor.to_numpy(), args.boot)

    readouts, v_readouts = {}, {}
    for arm, s in arms.items():
        for cond, store in (("AV", readouts), ("V", v_readouts)):
            part = s[s.condition == cond].set_index("uid")
            if not set(common) <= set(part.index):
                continue
            part = part.loc[common]
            store[f"{arm}: entropy"] = norm_entropy(softmax(part[[f"logit_{l}" for l in LABELS]].to_numpy(float)))
            if cond == "AV" and "amb_score" in part and part["amb_score"].notna().all():
                store[f"{arm}: amb_score"] = part["amb_score"].to_numpy(float)

    stats = lambda u: {"Spearman(readout, A^AV)": lambda i: spearman(u[i], y[i]),
                       "partial given emotion": lambda i: partial_spearman(u[i], y[i], emo[i]),
                       "partial given emotion x sentence": lambda i: partial_spearman(u[i], y[i], emo_sent[i]),
                       "AUROC top-25% A^AV": lambda i: auroc(u[i], high[i])}
    tables["readouts"] = pd.DataFrame([{"readout": n, **{k: boot(f) for k, f in stats(u).items()},
                                        **{f"fold {f}": round(spearman(u[folds == f], y[folds == f]), 3) for f in range(N_OUTER)}}
                                       for n, u in readouts.items()])

    pairs = [("m3_A1: entropy", "m3_A0: entropy", "4 fps input (m3)")]
    for name in readouts:
        if name.startswith("s3") or (name.split(":")[0] not in ("m3_A0", "s1_noI_A0", "m3_A1")):
            pairs += [(name, "m3_A1: entropy", "vs m3_A1 (stop rule for s3_A1: amb_score)"),
                      (name, "m3_A0: entropy", "vs m3_A0"), (name, "s1_noI_A0: entropy", "vs s1_noI_A0")]
    rows = []
    for a, b, label in pairs:
        if a not in readouts or b not in readouts:
            continue
        sa, sb = stats(readouts[a]), stats(readouts[b])
        rows.append({"comparison": f"{a} minus {b}", "note": label,
                     **{k: boot(lambda i, k=k: sa[k](i) - sb[k](i)) for k in sa},
                     **{f"fold {f}": round(spearman(readouts[a][folds == f], y[folds == f]) - spearman(readouts[b][folds == f], y[folds == f]), 3)
                        for f in range(N_OUTER)}})
    tables["paired"] = pd.DataFrame(rows)
    tables["v_reading_vs_video_only_raters"] = pd.DataFrame(
        [{"reading": n, "Spearman(U^V, A^V)": boot(lambda i, u=u: spearman(u[i], yv[i])),
          "partial given emotion": boot(lambda i, u=u: partial_spearman(u[i], yv[i], emo[i]))} for n, u in v_readouts.items()])

    lines = ["# Stage 3 outer evaluation (CREMA-D, seed 0)\n",
             f"Generated by `analysis/stage3_outer.py`. {len(common)} clips scored by every arm "
             f"({', '.join(arms)}). CIs: 95% bootstrap over actors ({args.boot} resamples), paired for differences.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
