"""Stage 1 probe: does the model hold more information about which clips the annotators disagree on than its
output entropy shows? (Plan approved 21/10/2026.)

Inputs: the hidden states saved by kaggle/probe_hidden.ipynb (results/probe_hidden/shard_*.npz): for every clip,
the last-position residual stream after several layers and after the final norm, read under T+A+V with the RQ3
soft-label adapter (`A`) and without it (`base`, zero-shot Qwen2.5-Omni-7B), plus the 7 label logits.

Probe (fixed before seeing the data): softmax regression from one layer's state to the 7 labels, trained with
the 3 annotators' vote distribution as a soft target (as the RQ3 soft arm), loss = mean cross-entropy +
lambda * ||W||^2, features standardised with train-split statistics. Fitted on the train split; the layer and
lambda are chosen on the valid split by the cross-entropy to the votes (not by the criterion). The probe's
entropy is then compared with the human entropy on the test split.

    primary    Spearman(probe entropy, human entropy) on A's states minus the same for A's output entropy,
               paired video-level bootstrap on the test split: the 95% CI must lie above 0
    secondary  the same probe on the base model's states; a probe trained on the valid split only (A's states
               on train clips were seen in fine-tuning, so a train-fitted probe may read memorised votes;
               layer and lambda by 5-fold cross-validation over videos within valid); a ridge regression
               straight onto human entropy (lambda by valid squared error); every layer's test value,
               descriptive only

A consistency check compares the saved logits with the earlier runs (results/rq3/rq3_soft, results/omni_runs).

Usage (from the repository root; needs scipy):
    python analysis/probe_analysis.py --out results/probe_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rq3_analysis import N_BOOT, Bootstrap, load_logits, softmax
from rq_analysis import LABELS, auroc, entropy, jsd, load_humans, markdown, spearman

LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
EARLIER = {"A": "results/rq3/rq3_soft/scores.jsonl", "base": "results/omni_runs/scores.jsonl"}


def load_shards(folder: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    uids, parts = [], {}
    for p in sorted(folder.glob("shard_*.npz")):
        z = np.load(p, allow_pickle=False)
        uids += z["uid"].tolist()
        for k in z.files:
            if k != "uid":
                parts.setdefault(k, []).append(z[k].astype(np.float32))
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    _, first = np.unique(uids, return_index=True)  # a clip saved twice (resumed run): keep the first copy
    first = np.sort(first)
    return [uids[i] for i in first], {k: v[first] for k, v in arrays.items()}


def fit_softmax(X: np.ndarray, Y: np.ndarray, lam: float, w0: np.ndarray | None = None) -> np.ndarray:
    """Convex, so a warm start (w0, the solution for a larger lambda) changes the speed, not the solution."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), np.float32)]).astype(np.float32)
    Yf = Y.astype(np.float32)

    def f(w):
        W = w.reshape(d + 1, Y.shape[1]).astype(np.float32)
        Z = Xb @ W
        Z -= Z.max(1, keepdims=True)
        logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        loss = -(Yf * logp).sum(1).mean() + lam * (W[:-1] ** 2).sum()
        grad = Xb.T @ (np.exp(logp) - Yf) / n
        grad[:-1] += 2 * lam * W[:-1]
        return float(loss), grad.ravel().astype(np.float64)

    x0 = np.zeros((d + 1) * Y.shape[1]) if w0 is None else w0.ravel()
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x.reshape(d + 1, Y.shape[1])


def predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return softmax(np.hstack([X, np.ones((len(X), 1), np.float32)]) @ W)


def ce(P: np.ndarray, Y: np.ndarray) -> float:
    return float(-(Y * np.log(np.clip(P, 1e-12, None))).sum(1).mean())


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    reg = lam * len(X) * np.eye(Xb.shape[1])
    reg[-1, -1] = 0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=Path, default=Path("results/probe_hidden"))
    parser.add_argument("--out", type=Path, default=Path("results/probe_analysis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    uids, arr = load_shards(args.hidden)
    pos = {u: i for i, u in enumerate(uids)}
    h = load_humans()
    h = h[h.uid.isin(pos)].reset_index(drop=True)
    split = {m: h[h["mode"] == m].reset_index(drop=True) for m in ("train", "valid", "test")}
    rows_of = {m: np.array([pos[u] for u in df.uid]) for m, df in split.items()}
    Y = {m: np.stack(df.human_dist) for m, df in split.items()}
    test = split["test"]
    het = test.human_entropy.to_numpy()
    disagree = test.all_agree.to_numpy() == 0
    boot = Bootstrap(test.video_id.to_numpy(), N_BOOT)
    feats = sorted({k.split("_", 1)[1] for k in arr if not k.endswith("logits")},
                   key=lambda k: (k == "norm", int(k[5:]) if k.startswith("layer") else 0))
    tables = {}

    # ---- consistency with the earlier runs ------------------------------------------------------
    rows = []
    for name, path in EARLIER.items():
        z = load_logits(path)
        common = [u for u in uids if (u, "TAV") in z.index]
        diff = np.abs(arr[f"{name}_logits"][[pos[u] for u in common]] - z.loc[[(u, "TAV") for u in common]].to_numpy())
        rows.append({"model": name, "earlier run": path, "clips compared": len(common),
                     "max |logit difference|": diff.max() if len(common) else np.nan,
                     "mean |logit difference|": diff.mean() if len(common) else np.nan})
    tables["check_logits_vs_earlier_runs"] = pd.DataFrame(rows)

    def scores(P_or_e, is_entropy=False):
        e = P_or_e if is_entropy else entropy(P_or_e)
        return {"Spearman(entropy, human entropy)": lambda i: spearman(e[i], het[i]),
                "AUROC: flags disagreement": lambda i: auroc(e[i], disagree[i])}

    # ---- probes fitted on train, chosen on valid --------------------------------------------------
    grid, test_pred = [], {}
    for name in ("A", "base"):
        best = None
        for f in feats:
            X = arr[f"{name}_{f}"]
            mu, sd = X[rows_of["train"]].mean(0), X[rows_of["train"]].std(0) + 1e-6
            Xs = {m: (X[r] - mu) / sd for m, r in rows_of.items()}
            W = None
            for lam in sorted(LAMBDAS, reverse=True):  # large to small lambda, each warm-started from the last
                W = fit_softmax(Xs["train"], Y["train"], lam, W)
                Pv, Pt = predict(W, Xs["valid"]), predict(W, Xs["test"])
                rec = {"model": name, "layer": f, "lambda": f"{lam:g}", "valid CE to votes": ce(Pv, Y["valid"]),
                       "test Spearman (descriptive)": spearman(entropy(Pt), het)}
                grid.append(rec)
                print(f"  probe {name} {f} lambda {lam:g}: valid CE {rec['valid CE to votes']:.4f}", flush=True)
                if best is None or rec["valid CE to votes"] < best[0]["valid CE to votes"]:
                    best = (rec, Pt)
        test_pred[f"probe on {name} states (train-fitted)"] = best
    tables["probe_grid"] = pd.DataFrame(grid)

    # ---- probe on A's states fitted on the valid split only ---------------------------------------
    vids = split["valid"].video_id.to_numpy()
    folds = {v: k % 5 for k, v in enumerate(np.random.default_rng(0).permutation(pd.unique(vids)))}
    fold = np.array([folds[v] for v in vids])
    best = None
    for f in feats:
        X = arr[f"A_{f}"]
        mu, sd = X[rows_of["valid"]].mean(0), X[rows_of["valid"]].std(0) + 1e-6
        Xv, Xt = (X[rows_of["valid"]] - mu) / sd, (X[rows_of["test"]] - mu) / sd
        Wk = [None] * 5
        for lam in sorted(LAMBDAS, reverse=True):
            Wk = [fit_softmax(Xv[fold != k], Y["valid"][fold != k], lam, Wk[k]) for k in range(5)]
            cv = np.mean([ce(predict(Wk[k], Xv[fold == k]), Y["valid"][fold == k]) for k in range(5)])
            if best is None or cv < best[0]:
                best = (cv, f, lam, Xv, Xt)
    cv, f, lam, Xv, Xt = best
    test_pred["probe on A states (valid-fitted)"] = ({"model": "A", "layer": f, "lambda": f"{lam:g}", "valid CV CE": cv},
                                                     predict(fit_softmax(Xv, Y["valid"], lam), Xt))

    # ---- ridge regression straight onto human entropy (A's states) --------------------------------
    best = None
    for f in feats:
        X = arr[f"A_{f}"]
        mu, sd = X[rows_of["train"]].mean(0), X[rows_of["train"]].std(0) + 1e-6
        Xs = {m: (X[r] - mu) / sd for m, r in rows_of.items()}
        y_tr, y_va = split["train"].human_entropy.to_numpy(), split["valid"].human_entropy.to_numpy()
        for lam in LAMBDAS + [10.0, 100.0]:
            b = fit_ridge(Xs["train"], y_tr, lam)
            pred = lambda Z: np.hstack([Z, np.ones((len(Z), 1))]) @ b
            mse = float(((pred(Xs["valid"]) - y_va) ** 2).mean())
            if best is None or mse < best[0]:
                best = (mse, f, lam, pred(Xs["test"]))
    mse, f, lam, ridge_test = best

    # ---- main table and the primary comparison ----------------------------------------------------
    a_out = softmax(arr["A_logits"][rows_of["test"]])
    base_out = softmax(arr["base_logits"][rows_of["test"]])
    reference = scores(a_out)
    rows = [{"predictor": "A output entropy (reference)", **{k: boot(s) for k, s in reference.items()}},
            {"predictor": "base output entropy", **{k: boot(s) for k, s in scores(base_out).items()}}]
    for name, (rec, P) in test_pred.items():
        rows.append({"predictor": name, "layer": rec["layer"], "lambda": rec["lambda"],
                     **{k: boot(s) for k, s in scores(P).items()}, "JSD to humans": boot(lambda i: jsd(P[i], Y["test"][i]).mean())})
    rows.append({"predictor": "ridge onto human entropy (A states)", "layer": f, "lambda": f"{lam:g}",
                 **{k: boot(s) for k, s in scores(ridge_test, is_entropy=True).items()}})
    tables["probe_test"] = pd.DataFrame(rows)

    rows = []
    for name, (rec, P) in test_pred.items():
        s = scores(P)
        rows.append({"comparison": f"{name} minus A output entropy" + (" (primary)" if name.startswith("probe on A states (train") else ""),
                     **{k: boot(lambda i, k=k: s[k](i) - reference[k](i)) for k in s}})
    s = scores(ridge_test, is_entropy=True)
    rows.append({"comparison": "ridge onto human entropy minus A output entropy",
                 **{k: boot(lambda i, k=k: s[k](i) - reference[k](i)) for k in s}})
    tables["probe_paired_differences"] = pd.DataFrame(rows)

    n = {m: len(r) for m, r in rows_of.items()}
    lines = ["# Stage 1: linear probe for human disagreement on the hidden states (Qwen2.5-Omni-7B)\n",
             f"Generated by `analysis/probe_analysis.py`. Clips with saved states: {n}. Layers: {', '.join(feats)}. "
             f"CIs: 95% video-level bootstrap on the test split ({N_BOOT} resamples).\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
