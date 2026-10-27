"""Stage C, readouts B0-B2: predicting the audio-visual raters' disagreement from a fine-tuned model, without retraining
it. Is a readout supervised directly on human disagreement better than the model's entropy, and does the hidden state
add anything beyond the six label logits? (Design approved 27/10/2026; B3, the jointly trained head, comes after.)

Inputs: results/visual_probe/vprobe_<arm>_A0_AV_f<fold>/ from kaggle/probe_visual.ipynb with CONDITION = AV: the Stage 1
adapter of each outer fold (seed 0) reading the actors it never saw, under audio-visual input (the Stage 1 AV prompt).

Target: A^AV = H(p_AV) / log 6 from the AV raters' raw vote counts (no smoothing; as in every earlier analysis).

Readouts, all scored on the same clips:
    B0  U^AV = normalised entropy of softmax(six label logits): the model's own uncertainty (= Stage 1 U^AV)
    B1  linear regression head on the model's output: the six label logits, their softmax probabilities and U^AV
        (is re-mapping the output enough?). It contains B0, so it can only match or improve on it in expectation.
    B2  the same head on the output features + the frozen hidden state (nested: B2 - B1 is what the hidden state adds
        beyond the output); `norm` (the answer position after the final norm) is the pre-registered state,
        `last21`, `last28`, `vid21`, `vis` are descriptive
    A head on the logits alone cannot express the entropy (a non-linear function of the logits); on the real m3
    logits it reaches about 0.35 against 0.55 for B0, hence the output features and the nesting.
    references: emotion prior (mean A^AV per intended emotion) and emotion x sentence prior
Heads: Huber loss (delta = 1) on A^AV standardised with the training clips' mean and SD, + lambda ||w||^2, features
standardised; lambda in {1, 0.1, 0.01, 1e-3, 1e-4} chosen by the Huber loss on 15% of the training actors held out,
then refitted on all training actors. Splits (P2 of visual_probe_analysis.py): within each outer fold's actors, five
folds over actors, so the heads never train on actors the adapter saw.

Metrics (95% bootstrap over actors, paired for differences):
    primary      Spearman(readout, A^AV)
    secondary    AUROC for the top quarter of A^AV (threshold from the other outer folds' clips); MAE (heads only,
                 on the A^AV scale); partial Spearman given the intended emotion, and given emotion x sentence
    breakdowns   per intended emotion (n, Spearman), per outer fold, clips with and without perceived conflict (the
                 audio-only and video-only raters' top labels share no label); leave-one-emotion-out for B2 - B1
Reading (fixed before the runs): B2 > B0 = reading the hidden state beats the entropy. B3 is trained only if B2 - B1
is above 0 with its 95% CI, for Spearman and for the partial Spearman given emotion x sentence (the hidden state holds
sample-specific disagreement information beyond the six logits).

Usage (from the repository root; downloads the CREMA-D rating files):
    python analysis/ambiguity_readout.py --runs results/visual_probe --out results/ambiguity_readout
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cremad_human import counts
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import N_BOOT, Bootstrap, softmax
from rq_analysis import auroc, markdown, spearman
from visual_probe_analysis import N_OUTER, inner_holdout, load_run, norm_entropy, outer_folds, partial_spearman, prior_oof, splits

LAMBDAS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
NAME = re.compile(r"vprobe_(?P<arm>m3|s1_noI)_A0_AV_f(?P<fold>\d)$")
FEATURES = ["logits", "norm", "last21", "last28", "vid21", "vis"]
DATA: dict = {}  # filled before the worker pool forks


def fit_huber(X: np.ndarray, y: np.ndarray, lam: float, w0: np.ndarray | None = None, delta: float = 1.0) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), np.float32)]).astype(np.float32)

    def f(w):
        r = Xb @ w.astype(np.float32) - y
        small = np.abs(r) <= delta
        loss = np.where(small, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta)).mean() + lam * (w[:-1] ** 2).sum()
        grad = Xb.T @ np.where(small, r, delta * np.sign(r)) / n
        grad[:-1] += 2 * lam * w[:-1]
        return float(loss), grad.astype(np.float64)

    res = minimize(f, np.zeros(d + 1) if w0 is None else w0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


def huber(r: np.ndarray, delta: float = 1.0) -> float:
    return float(np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta)).mean())


def head_oof(X: np.ndarray, y: np.ndarray, actors: np.ndarray, folds: np.ndarray):
    pred, chosen = np.full(len(X), np.nan), []
    for j, (tr, te) in enumerate(splits(actors, folds, "P2")):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = ((X - mu) / sd).astype(np.float32)
        ym, ys = y[tr].mean(), y[tr].std() + 1e-9
        yz = ((y - ym) / ys).astype(np.float32)
        held = inner_holdout(actors[tr], seed=100 + j)
        a, b = tr[~held], tr[held]
        w, best = None, None
        for lam in LAMBDAS:  # large to small, warm-started
            w = fit_huber(Z[a], yz[a], lam, w)
            loss = huber(np.hstack([Z[b], np.ones((len(b), 1), np.float32)]) @ w - yz[b])
            if best is None or loss < best[0]:
                best = (loss, lam, w)
        w = fit_huber(Z[tr], yz[tr], best[1], best[2])
        pred[te] = (np.hstack([Z[te], np.ones((len(te), 1), np.float32)]) @ w) * ys + ym
        chosen.append(best[1])
    return pred, chosen


def output_features(logits: np.ndarray) -> np.ndarray:
    p = softmax(logits.astype(np.float64))
    return np.hstack([logits, p, norm_entropy(p)[:, None]]).astype(np.float32)


def _job(key):
    arm, feature = key
    d = DATA[arm]
    X = output_features(d["arrays"]["logits"])
    if feature != "logits":  # nested: output features + the hidden state
        X = np.hstack([X, d["arrays"][feature].astype(np.float32)])
    return key, *head_oof(X, d["y"], d["actors"], d["folds"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/visual_probe"))
    parser.add_argument("--out", type=Path, default=Path("results/ambiguity_readout"))
    parser.add_argument("--earlier-stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table().set_index("uid")
    AV, A, V = counts(df, "av"), counts(df, "a"), counts(df, "v")
    df["A_AV"] = norm_entropy(AV / np.clip(AV.sum(1, keepdims=True), 1, None))
    top = lambda c: c == c.max(1, keepdims=True)
    df["conflict"] = ~(top(A) & top(V)).any(1)
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    pos = {u: i for i, u in enumerate(df.index)}

    # ---- load runs, check them against the Stage 1 AV scores --------------------------------------------
    parts, checks = {}, []
    for path in sorted(p for p in args.runs.iterdir() if NAME.match(p.name)):
        m = NAME.match(path.name)
        uids, arrays, meta = load_run(path)
        fold = int(m["fold"])
        row = {"run": path.name, "clips": len(uids), "expected": int(((df.fold == fold) & (df.av_n > 0)).sum()),
               "condition (meta)": meta.get("condition")}
        earlier = args.earlier_stage1 / f"stage1_{m['arm']}_f{fold}_s0" / "scores.jsonl"
        if earlier.exists():
            s = pd.DataFrame([json.loads(l) for l in open(earlier, encoding="utf-8") if l.strip()])
            s = s[(s.condition == "AV") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
            s = s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
            common = [i for i, u in enumerate(uids) if u in s.index]
            ref = s.loc[[uids[i] for i in common]].to_numpy()
            row.update({"clips vs Stage 1 AV scores": len(common),
                        "median |logit diff|": round(float(np.median(np.abs(arrays["logits"][common] - ref))), 3),
                        "same top label": round(float((arrays["logits"][common].argmax(1) == ref.argmax(1)).mean()), 3)})
        checks.append(row)
        parts.setdefault(m["arm"], []).append((uids, arrays))
    assert parts, f"no vprobe_<arm>_A0_AV_f<fold> runs under {args.runs}"
    for arm, ps in parts.items():
        uids = sum((p[0] for p in ps), [])
        arrays = {k: np.concatenate([p[1][k] for p in ps]) for k in FEATURES}
        keep = [i for i, u in enumerate(uids) if u in pos and df.av_n.iloc[pos[u]] > 0]
        idx = np.array([pos[uids[i]] for i in keep])
        DATA[arm] = {"idx": idx, "arrays": {k: v[keep] for k, v in arrays.items()}, "y": df.A_AV.to_numpy()[idx],
                     "actors": df.actor.to_numpy()[idx], "folds": df.fold.to_numpy()[idx]}
    tables = {"run_checks": pd.DataFrame(checks)}

    jobs = [(arm, f) for arm in DATA for f in FEATURES]
    oof = {}
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for n_done, (key, pred, lams) in enumerate(pool.map(_job, jobs), 1):
            oof[key] = pred
            print(f"  {n_done}/{len(jobs)} {key}: lambdas {lams}", flush=True)

    rows, paired, per_emotion, per_fold = [], [], [], []
    for arm, d in DATA.items():
        i0, y = d["idx"], d["y"]
        emo, sent = df.intended.to_numpy()[i0], (df.intended + df.sentence_code).to_numpy()[i0]
        conflict = df.conflict.to_numpy()[i0]
        high = np.zeros(len(i0), bool)
        for f in range(N_OUTER):
            here = d["folds"] == f
            high[here] = y[here] >= np.quantile(df.A_AV.to_numpy()[df.fold.to_numpy() != f], 0.75)
        preds = {"B0: U^AV (entropy of the model's prediction)": norm_entropy(softmax(d["arrays"]["logits"].astype(np.float64))),
                 "B1: head on the output (logits, probabilities, U^AV)": oof[(arm, "logits")],
                 "B2: head on the output + hidden state (norm)": oof[(arm, "norm")],
                 **{f"B2 (descriptive): output + {f}": oof[(arm, f)] for f in FEATURES if f not in ("logits", "norm")},
                 "emotion prior": prior_oof(y, emo, d["actors"], d["folds"], "P2"),
                 "emotion x sentence prior": prior_oof(y, sent, d["actors"], d["folds"], "P2")}
        boot = Bootstrap(d["actors"], args.boot)

        def stats(u, mae=True):
            out = {"Spearman(readout, A^AV)": lambda i: spearman(u[i], y[i]),
                   "partial | emotion": lambda i: partial_spearman(u[i], y[i], emo[i]),
                   "partial | emotion x sentence": lambda i: partial_spearman(u[i], y[i], sent[i]),
                   "AUROC top-25% A^AV": lambda i: auroc(u[i], high[i]),
                   "Spearman, conflict clips": lambda i: spearman(u[i][conflict[i]], y[i][conflict[i]]),
                   "Spearman, no-conflict clips": lambda i: spearman(u[i][~conflict[i]], y[i][~conflict[i]])}
            if mae:
                out["MAE"] = lambda i: float(np.abs(u[i] - y[i]).mean())
            return out

        for name, u in preds.items():
            rows.append({"arm": arm, "readout": name, **{k: boot(s) for k, s in stats(u, name.startswith("B1") or name.startswith("B2")).items()}})
        b0, b1, b2 = (preds[k] for k in list(preds)[:3])
        for label, (u, v) in {"B1 - B0": (b1, b0), "B2 - B0": (b2, b0), "B2 - B1 (decides B3)": (b2, b1)}.items():
            su, sv = stats(u, False), stats(v, False)
            paired.append({"arm": arm, "comparison": label, **{k: boot(lambda i, k=k: su[k](i) - sv[k](i)) for k in su}})
        loo = {e: spearman(b2[emo != e], y[emo != e]) - spearman(b1[emo != e], y[emo != e]) for e in sorted(set(emo))}
        paired.append({"arm": arm, "comparison": "B2 - B1, leave-one-emotion-out (min, emotion left out)",
                       "Spearman(readout, A^AV)": f"{min(loo.values()):+.3f} ({min(loo, key=loo.get)})"})
        for e in sorted(set(emo)):
            sel = np.flatnonzero(emo == e)
            b = Bootstrap(d["actors"][sel], min(args.boot, 500))
            per_emotion.append({"arm": arm, "emotion": e, "n": len(sel),
                                **{k[:2]: b(lambda i, u=u: spearman(u[sel][i], y[sel][i])) for k, u in list(preds.items())[:3]}})
        for f in range(N_OUTER):
            sel = d["folds"] == f
            per_fold.append({"arm": arm, "fold": f, "n": int(sel.sum()),
                             **{k[:2]: round(spearman(u[sel], y[sel]), 3) for k, u in list(preds.items())[:3]},
                             "B2 - B1": round(spearman(b2[sel], y[sel]) - spearman(b1[sel], y[sel]), 3)})
    tables.update(readouts=pd.DataFrame(rows), paired=pd.DataFrame(paired), per_emotion=pd.DataFrame(per_emotion),
                  per_fold=pd.DataFrame(per_fold))

    n = {arm: len(d["idx"]) for arm, d in DATA.items()}
    lines = ["# Stage C, readouts B0-B2: predicting the AV raters' disagreement from the fine-tuned model\n",
             f"Generated by `analysis/ambiguity_readout.py`. Clips: {n}. Heads trained within each outer fold's actors "
             f"(five folds over actors). CIs: 95% bootstrap over actors ({args.boot} resamples; per emotion "
             f"{min(args.boot, 500)}).\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
