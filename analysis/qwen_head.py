"""Readout control for the HSEmotion comparison: the same head family, trained the same way, on the hidden state of the
Qwen v_only checkpoint. Can a learned readout close the gap between the two pipelines? (Design approved 03/11/2026.)

Input: results/visual_probe/vprobe_v_only_A0_f<k>_all/ (kaggle/probe_visual.ipynb with MODEL_KIND = v_only): every clip
read under V (full frame, A0 timestamps, the v_only prompt) by the v_only adapter of outer fold k. For fold k, the head is
fitted on that fold's training actors and applied to its held-out actors, all with fold k's own checkpoint: rows from
different checkpoints are never mixed.

Feature (locked): `norm` = the residual stream at the answer position after the final norm, i.e. the vector the
language-model head reads; no pooling. Head, loss, standardisation, weight-decay grid, epoch selection on the Stage 1 inner
split and refit: exactly as analysis/hsemotion_head.py (linear main, MLP 128 secondary).

Not independent: the v_only adapter of fold k was trained on all of fold k's training actors, including the inner
validation actors used to choose the head, so their features were shaped by their own labels. The inner choice is
therefore optimistic, and the train features may separate the votes better than the held-out features do (a bias against
this readout). Reported as such.

Check: the six label logits saved with the features must match results/stage1/stage1_v_only_f<k>_s0/scores.jsonl on the
held-out actors (same checkpoint, same input).

Usage (from the repository root; needs torch):
    python analysis/qwen_head.py --runs results/visual_probe --out results/qwen_head
then vonly_compare.py --extra qwen_linear=results/qwen_head/linear/scores.jsonl qwen_mlp=results/qwen_head/mlp/scores.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cremad_human import counts, jsd
from hsemotion_head import INNER_VAL_SHARE, MAX_EPOCHS, WEIGHT_DECAYS, train
from preprocess_cremad import LABELS, clip_table
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, SPLIT_SEED, load_run, outer_folds

FEATURE = "norm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/visual_probe"))
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--out", type=Path, default=Path("results/qwen_head"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    preds = {kind: {} for kind in ("linear", "mlp")}
    grid, checks = [], []
    for fo in range(N_OUTER):
        uids, arrays, meta = load_run(args.runs / f"vprobe_v_only_A0_f{fo}_all")
        assert meta.get("model_kind") == "v_only" and meta.get("fold") == fo and meta.get("all_clips"), meta
        keep = [i for i, u in enumerate(uids) if u in df.index and df.loc[u, "v_n"] > 0]
        uids = [uids[i] for i in keep]
        X = arrays[FEATURE][keep].astype(np.float32)
        d = df.loc[uids]
        P = counts(d, "v")
        P = (P / P.sum(1, keepdims=True)).astype(np.float32)
        folds, actors = d.fold.to_numpy(), d.actor.to_numpy()
        tr_all, te = folds != fo, folds == fo
        # consistency with the v_only scores of the same checkpoint on its held-out actors
        s = pd.DataFrame([json.loads(l) for l in open(args.stage1 / f"stage1_v_only_f{fo}_s0" / "scores.jsonl", encoding="utf-8") if l.strip()])
        s = s[s.condition == "V"].drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
        test_uids = [u for u, t in zip(uids, te) if t and u in s.index]
        lg = arrays["logits"][keep][[uids.index(u) for u in test_uids]]
        ref = s.loc[test_uids].to_numpy()
        checks.append({"fold": fo, "clips read": len(uids), "training-actor clips": int(tr_all.sum()), "held-out clips": int(te.sum()),
                       "median |logit diff| vs v_only scores": round(float(np.median(np.abs(lg - ref))), 4),
                       "same top label": round(float((lg.argmax(1) == ref.argmax(1)).mean()), 4)})
        train_actors = sorted(set(actors[tr_all]))
        inner = set(np.random.default_rng(SPLIT_SEED + 1 + fo).permutation(train_actors)[:max(1, round(INNER_VAL_SHARE * len(train_actors)))])
        val = tr_all & np.isin(actors, list(inner))
        fit = tr_all & ~val
        for kind in ("linear", "mlp"):
            mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
            best = None
            for wd in WEIGHT_DECAYS:
                _, curve = train(kind, (X[fit] - mu) / sd, P[fit], wd, MAX_EPOCHS, (X[val] - mu) / sd, P[val])
                ep = int(np.argmin(curve)) + 1
                grid.append({"fold": fo, "head": kind, "weight decay": wd, "best epoch": ep, "inner JSD at best epoch": round(curve[ep - 1], 4)})
                if best is None or curve[ep - 1] < best[0]:
                    best = (curve[ep - 1], wd, ep)
            _, wd, ep = best
            grid.append({"fold": fo, "head": kind, "weight decay": wd, "best epoch": ep, "inner JSD at best epoch": round(best[0], 4), "chosen": True})
            mu, sd = X[tr_all].mean(0), X[tr_all].std(0) + 1e-6
            head, _ = train(kind, (X[tr_all] - mu) / sd, P[tr_all], wd, ep)
            with torch.no_grad():
                p = torch.softmax(head(torch.from_numpy((X[te] - mu) / sd)), -1).numpy()
            preds[kind].update({u: q for u, q in zip(np.array(uids)[te], p)})

    for kind, pr in preds.items():
        (args.out / kind).mkdir(exist_ok=True)
        with open(args.out / kind / "scores.jsonl", "w", encoding="utf-8") as fh:
            for u, q in pr.items():
                fh.write(json.dumps({"model": f"qwen-v_only-{FEATURE}-{kind}", "uid": u, "condition": "V", "task": "score",
                                     **{f"logit_{l}": float(np.log(max(v, 1e-12))) for l, v in zip(LABELS, q)}}) + "\n")
    tables = {"checks": pd.DataFrame(checks), "selection": pd.DataFrame(grid)}
    lines = ["# Qwen v_only hidden state (norm) + the HSEmotion head family, on the video-only votes\n",
             "Generated by `analysis/qwen_head.py`. Inner choices are not independent of the checkpoint (see the docstring). "
             "The comparison is in `results/vonly_compare/`.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
