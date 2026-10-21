"""M2 on CREMA-D: do zero-shot MLLMs track, clip by clip, what people perceive and where people disagree?

Inputs: results/cremad_qwen/scores.jsonl and results/cremad_minicpm/scores.jsonl (kaggle/cremad_qwen.ipynb,
kaggle/cremad_minicpm.ipynb): six label logits per clip and condition (AV audio-visual, A audio only, V video
only), scored as in the CH-SIMS runs. The model's distribution is the softmax over the six logits. Humans: the
vote distribution of the raters of the same condition (about 10 per clip; different people per condition).

Tables (all clips; 95% bootstrap over actors):
    per condition   accuracy against the same condition's human majority; JSD to the same condition's votes;
                    share of the next-token probability on the six labels (label mass)
    clip level      Spearman between the model's entropy and the humans' entropy, per condition (does the model's
                    uncertainty under A or V follow how ambiguous people find the audio or the face alone?);
                    AUROC for clips the AV raters do not agree on
    conflict        perceived conflict (audio-only and video-only raters' top labels share no label): entropy
                    change on conflict clips, model AV vs human AV; does the model's own A-vs-V disagreement
                    (JSD between its A and V readings) flag perceived conflict?
    following       on conflict clips, share of AV answers that take the audio-only raters' top label vs the
                    video-only raters' top label, humans (AV raters pooled) vs each model

Usage (from the repository root; downloads the CREMA-D rating files):
    python analysis/cremad_models.py --out results/cremad_models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import counts, entropy_counts, jsd
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import N_BOOT, Bootstrap, softmax
from rq_analysis import auroc, markdown, spearman

RUNS = {"Qwen2.5-Omni-7B": "results/cremad_qwen/scores.jsonl", "MiniCPM-o 2.6": "results/cremad_minicpm/scores.jsonl"}
CONDS = {"AV": "av", "A": "a", "V": "v"}


def load_run(path: str) -> tuple[pd.DataFrame, pd.Series]:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s["task"] == "score") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    s = s.drop_duplicates(["uid", "condition"], keep="last").set_index(["uid", "condition"])
    z = s[[f"logit_{l}" for l in LABELS]].to_numpy(float)
    mass = pd.Series(np.exp(z - s["lse"].to_numpy()[:, None]).sum(1), index=s.index)
    return s[[f"logit_{l}" for l in LABELS]], mass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/cremad_models"))
    parser.add_argument("--runs", nargs="*", help="name=path pairs, overriding RUNS")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = dict(r.split("=", 1) for r in args.runs) if args.runs else {k: v for k, v in RUNS.items() if Path(v).exists()}
    assert runs, "no score files found"

    df = clip_table()
    loaded = {name: load_run(path) for name, path in runs.items()}
    keep = set(df.uid)
    for logits, _ in loaded.values():  # clips scored under all three conditions by every model
        keep &= {u for u in logits.index.get_level_values(0) if all((u, c) in logits.index for c in CONDS)}
    df = df[df.uid.isin(keep) & (df[["a_n", "v_n", "av_n"]] > 0).all(1)].reset_index(drop=True)
    boot = Bootstrap(df.video_id.to_numpy(), N_BOOT)
    H = {c: counts(df, k) for c, k in CONDS.items()}
    Hp = {c: h / h.sum(1, keepdims=True) for c, h in H.items()}
    Hent = {c: entropy_counts(h) for c, h in H.items()}
    top = {c: h == h.max(1, keepdims=True) for c, h in H.items()}
    conflict = ~(top["A"] & top["V"]).any(1)
    disagree = (H["AV"] > 0).sum(1) > 1

    P, mass = {}, {}
    for name, (logits, m) in loaded.items():
        P[name] = {c: softmax(logits.loc[[(u, c) for u in df.uid]].to_numpy()) for c in CONDS}
        mass[name] = {c: m.loc[[(u, c) for u in df.uid]].to_numpy() for c in CONDS}
    ent = lambda p: -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    tables = {}

    # ---- per condition ---------------------------------------------------------------------------
    rows = []
    for name in P:
        for c in CONDS:
            pred = P[name][c].argmax(1)
            hit = top[c][np.arange(len(df)), pred]  # the model's label is (one of) the raters' top label(s)
            d = jsd(P[name][c], Hp[c])
            rows.append({"model": name, "condition": c, "accuracy vs same-condition human majority": boot(lambda i: hit[i].mean()),
                         "JSD to same-condition votes": boot(lambda i: d[i].mean()), "label mass": mass[name][c].mean(),
                         "model mean entropy": ent(P[name][c]).mean(), "human mean entropy": Hent[c].mean()})
    tables["per_condition"] = pd.DataFrame(rows)

    # ---- clip level --------------------------------------------------------------------------------
    rows = []
    for name in P:
        for c in CONDS:
            e = ent(P[name][c])
            rows.append({"model": name, "condition": c,
                         "Spearman(model entropy, human entropy, same condition)": boot(lambda i: spearman(e[i], Hent[c][i])),
                         "Spearman(model entropy, human AV entropy)": boot(lambda i: spearman(e[i], Hent["AV"][i])),
                         "AUROC: flags AV disagreement": boot(lambda i: auroc(e[i], disagree[i]))})
    rows.append({"model": "reference: video-only raters' entropy (M1)", "condition": "V",
                 "Spearman(model entropy, human AV entropy)": boot(lambda i: spearman(Hent["V"][i], Hent["AV"][i])),
                 "AUROC: flags AV disagreement": boot(lambda i: auroc(Hent["V"][i], disagree[i]))})
    tables["clip_level"] = pd.DataFrame(rows)

    # ---- conflict ----------------------------------------------------------------------------------
    change = lambda e: (lambda i: e[i][conflict[i]].mean() - e[i][~conflict[i]].mean())
    rows = [{"who": "humans (AV raters)", "AV entropy change on perceived conflict": boot(change(Hent["AV"]))}]
    for name in P:
        own = jsd(P[name]["A"], P[name]["V"])
        rows.append({"who": name, "AV entropy change on perceived conflict": boot(change(ent(P[name]["AV"]))),
                     "AUROC: its A-vs-V disagreement (JSD) flags perceived conflict": boot(lambda i: auroc(own[i], conflict[i])),
                     "share of clips where its A and V answers differ": (P[name]["A"].argmax(1) != P[name]["V"].argmax(1)).mean()})
    tables["conflict"] = pd.DataFrame(rows)

    # ---- following on conflict clips ---------------------------------------------------------------
    a_top, v_top = H["A"].argmax(1), H["V"].argmax(1)  # ties to the first label; conflict clips share no top label
    idx = np.flatnonzero(conflict)
    votes_av = H["AV"][idx]
    rows = [{"who": "humans (AV raters pooled)", "conflict clips": len(idx),
             "takes the audio-only top label": votes_av[np.arange(len(idx)), a_top[idx]].sum() / votes_av.sum(),
             "takes the video-only top label": votes_av[np.arange(len(idx)), v_top[idx]].sum() / votes_av.sum()}]
    for name in P:
        pred = P[name]["AV"].argmax(1)[idx]
        rows.append({"who": name, "conflict clips": len(idx), "takes the audio-only top label": (pred == a_top[idx]).mean(),
                     "takes the video-only top label": (pred == v_top[idx]).mean()})
    tables["following_on_conflict"] = pd.DataFrame(rows)

    lines = ["# M2 on CREMA-D: zero-shot MLLMs against what people perceive\n",
             f"Generated by `analysis/cremad_models.py`. {len(df)} clips, {df.actor.nunique()} actors "
             f"({int(conflict.sum())} with perceived conflict). Models: {', '.join(P)}. "
             f"CIs: 95% bootstrap over actors ({N_BOOT} resamples).\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
