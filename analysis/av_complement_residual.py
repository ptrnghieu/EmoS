"""Exploratory round locked 11/11/2026 before running (after av_complement_head.py and av_complement_diag.py were seen):
a linear residual readout around Qwen's own logits, the same recipe for the video-only and the audio-visual votes.

    z_out = z_base + W x + b,  W = b = 0 at the start (the output then equals Qwen's own, checked numerically)
    objective = mean soft-label CE + lambda (||W||^2 + ||b||^2)

z_base: Qwen's (s1_noI) logits of the target reading, centred over the six classes, never standardised. x, the
standardised features of the correction only:
    AV target  R0: AV logits | R1: R0 + HSE | R2: AV, V, A logits | R3: R2 + HSE        (z_base = AV logits for all)
    V target   V0: V logits  | V1: V0 + HSE                                             (z_base = V logits)
Logits: centred per clip and reading, then StandardScaler. HSE (mean and SD of the A2 face-crop embeddings, 2560-d):
StandardScaler -> PCA 32 -> StandardScaler. Every preprocessing step is fitted on the part being fitted only (the fit
part during selection, then the whole head-training part for the refit).
Solver: SciPy L-BFGS-B without bounds, float64, analytic gradient, gtol 1e-6, ftol 1e-12, maxiter 2000. Stop status,
iterations, objective at start and end and the max-abs gradient are recorded; a fit whose final max-abs gradient is
above 1e-6 is reported as convergence not confirmed. The objective at the end must not exceed the start (W = b = 0).
Selection: lambda in {1e-4, ..., 1e2} plus "no correction" (Qwen's own prediction, no optimisation), by JSD on the
inner validation actors; ties within 1e-9 go to the stronger regularisation. Splits as av_complement_head.py (held-out
actors of each s1_noI checkpoint, 3 head folds by actor, inner split inside the head's training part).

Reading: JSD main, partial Spearman (given the intended emotion) secondary, reported separately. AV: R1 - R0 main,
R3 - R2 secondary. V: V1 - V0 is the positive control of this recipe. If it shows no clear gain: "the V benefit is not
reproduced in this few-actor protocol with the PCA-32 representation" (data size, PCA and the readout family are all
candidate reasons). No further PCA sizes, heads or the GPU level are opened on the basis of this round.

Usage (from the repository root; needs scipy):
    python analysis/av_complement_residual.py --out results/av_complement_residual
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from av_complement_diag import load_checkpoint
from av_complement_head import N_BOOT, N_HEAD
from combined_head import hsemotion_pooled
from cremad_human import jsd
from hsemotion_head import INNER_VAL_SHARE
from preprocess_cremad import clip_table
from rq3_analysis import Bootstrap, fmt
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, SPLIT_SEED, norm_entropy, outer_folds, partial_spearman

LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
N_PCA, GTOL = 32, 1e-6
SETS = {"AV": {"R0": (("AV",), False), "R1": (("AV",), True), "R2": (("AV", "V", "A"), False), "R3": (("AV", "V", "A"), True)},
        "V": {"V0": (("V",), False), "V1": (("V",), True)}}


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class Prep:
    """Logits: StandardScaler. HSE: StandardScaler -> PCA 32 -> StandardScaler. Fitted on the rows given."""

    def __init__(self, L, H, idx):
        self.lm, self.ls = L[idx].mean(0), L[idx].std(0) + 1e-9
        self.H = H is not None
        if self.H:
            self.hm, self.hs = H[idx].mean(0), H[idx].std(0) + 1e-9
            Z = (H[idx] - self.hm) / self.hs
            _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
            self.zm, self.V = Z.mean(0), vt[:N_PCA].T
            C = (Z - self.zm) @ self.V
            self.cm, self.cs = C.mean(0), C.std(0) + 1e-9

    def __call__(self, L, H):
        parts = [(L - self.lm) / self.ls]
        if self.H:
            parts.append(((((H - self.hm) / self.hs) - self.zm) @ self.V - self.cm) / self.cs)
        return np.hstack(parts)


def fit_residual(z, x, p, lam):
    n, d = x.shape
    K = z.shape[1]

    def f(theta):
        W, b = theta[:K * d].reshape(K, d), theta[K * d:]
        q = softmax(z + x @ W.T + b)
        ce = -(p * np.log(np.clip(q, 1e-300, None))).sum(1).mean()
        g = (q - p) / n
        obj = ce + lam * (np.sum(W ** 2) + np.sum(b ** 2))
        grad = np.concatenate([(g.T @ x + 2 * lam * W).ravel(), g.sum(0) + 2 * lam * b])
        return obj, grad

    theta0 = np.zeros(K * d + K)
    f0, _ = f(theta0)
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"gtol": GTOL, "ftol": 1e-12, "maxiter": 2000})
    f1, g1 = f(res.x)
    ginf = float(np.abs(g1).max())
    assert f1 <= f0 + 1e-12, (f0, f1)
    W, b = res.x[:K * d].reshape(K, d), res.x[K * d:]
    info = {"status": int(res.status), "message": str(res.message), "iterations": int(res.nit), "objective start": f0,
            "objective end": f1, "max abs gradient": ginf, "converged": ginf <= GTOL}
    return W, b, info


def run_target(target, df, hse, stage1):
    rows, fits = [], []
    for k in range(N_OUTER):
        uids, d, P, hf, H, lg = load_checkpoint(k, df, hse, stage1, target)
        P = P.astype(np.float64)
        actors = d.actor.to_numpy()
        z = lg[target].loc[uids].to_numpy(np.float64)
        Ls = {n: np.hstack([lg[c].loc[uids].to_numpy(np.float64) for c in conds]) for n, (conds, _) in SETS[target].items()}
        H = H.astype(np.float64)
        q0 = softmax(z)
        pred = {n: np.zeros_like(P) for n in SETS[target]}
        for j in range(N_HEAD):
            te, tr = hf == j, hf != j
            tr_act = sorted(set(actors[tr]))
            inner = set(np.random.default_rng(SPLIT_SEED + 200 + 10 * k + j).permutation(tr_act)[:max(1, round(INNER_VAL_SHARE * len(tr_act)))])
            val = tr & np.isin(actors, list(inner))
            fit = tr & ~val
            for n, (conds, use_h) in SETS[target].items():
                L, Hn = Ls[n], (H if use_h else None)
                pr = Prep(L, Hn, fit)
                xf, xv = pr(L[fit], Hn[fit] if use_h else None), pr(L[val], Hn[val] if use_h else None)
                # check at the start: W = b = 0 reproduces Qwen's own probabilities
                assert np.abs(softmax(z[fit] + xf @ np.zeros((6, xf.shape[1])).T) - q0[fit]).max() < 1e-12
                cand = [(float(jsd(q0[val], P[val]).mean()), np.inf, None)]
                for lam in LAMBDAS:
                    W, b, info = fit_residual(z[fit], xf, P[fit], lam)
                    s = float(jsd(softmax(z[val] + xv @ W.T + b), P[val]).mean())
                    cand.append((s, lam, info))
                    fits.append({"target": target, "checkpoint": k, "head fold": j, "features": n, "stage": "selection",
                                 "lambda": lam, "inner JSD": s, **info})
                best_s = min(c[0] for c in cand)
                chosen = max(c[1] for c in cand if c[0] <= best_s + 1e-9)  # ties: stronger regularisation
                if np.isinf(chosen):
                    pred[n][te] = q0[te]
                    fits.append({"target": target, "checkpoint": k, "head fold": j, "features": n, "stage": "chosen",
                                 "lambda": chosen, "inner JSD": best_s})
                else:
                    pr = Prep(L, Hn, tr)
                    W, b, info = fit_residual(z[tr], pr(L[tr], Hn[tr] if use_h else None), P[tr], chosen)
                    pred[n][te] = softmax(z[te] + pr(L[te], Hn[te] if use_h else None) @ W.T + b)
                    fits.append({"target": target, "checkpoint": k, "head fold": j, "features": n, "stage": "chosen",
                                 "lambda": chosen, "inner JSD": best_s, **info})
        for i, u in enumerate(uids):
            rows.append({"uid": u, "checkpoint": k, "actor": int(actors[i]), "emotion": d.intended.iloc[i], "p": P[i],
                         "Qwen (no head)": q0[i], **{n: pred[n][i] for n in SETS[target]}})
    return pd.DataFrame(rows), pd.DataFrame(fits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--features", type=Path, default=Path("results/hsemotion_features/features.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/av_complement_residual"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    hse = hsemotion_pooled(args.features)

    tables, all_fits = {}, []
    for target, pairs in (("V", [("V1", "V0", "positive control"), ("V0", "Qwen (no head)", "readout"), ("V1", "Qwen (no head)", "readout")]),
                          ("AV", [("R1", "R0", "main"), ("R3", "R2", "secondary"), ("R2", "R0", "context"),
                                  ("R0", "Qwen (no head)", "readout"), ("R1", "Qwen (no head)", "readout")])):
        R, F = run_target(target, df, hse, args.stage1)
        all_fits.append(F)
        Pv = np.stack(R.p)
        hp, emo, act, ck = norm_entropy(Pv), R.emotion.to_numpy(), R.actor.to_numpy(), R.checkpoint.to_numpy()
        top = Pv == Pv.max(1, keepdims=True)
        names = ["Qwen (no head)"] + list(SETS[target])
        Q = {n: np.stack(R[n]) for n in names}
        boot = Bootstrap(act, N_BOOT)

        def met(q):
            j, hq, hit = jsd(q, Pv), norm_entropy(q), top[np.arange(len(q)), q.argmax(1)]
            return {f"JSD {target}": lambda i: j[i].mean(), f"partial {target}": lambda i: partial_spearman(hq[i], hp[i], emo[i]),
                    "Spearman(H(q), A)": lambda i: spearman(hq[i], hp[i]),
                    "top label among raters' top": lambda i: hit[i].mean(), "mean H(q)": lambda i: hq[i].mean()}

        tables[f"{target}_readings"] = pd.DataFrame([{"predictor": n, **{m: boot(f) for m, f in met(Q[n]).items()},
                                                      **{f"JSD ckpt {c}": round(float(jsd(Q[n], Pv)[ck == c].mean()), 4) for c in range(N_OUTER)}} for n in names])
        prow, vrow = [], []
        for new, ref, role in pairs:
            a, b = met(Q[new]), met(Q[ref])
            rec = {"comparison": f"{new} minus {ref}", "role": role}
            st = {}
            for m in a:
                samples = np.array([a[m](i) - b[m](i) for i in boot.samples])
                point = a[m](boot.all) - b[m](boot.all)
                lo, hi = np.nanpercentile(samples, [2.5, 97.5])
                rec[m] = fmt(point, samples)
                st[m] = (point, lo, hi)
            for c in range(N_OUTER):
                sel = np.flatnonzero(ck == c)
                rec[f"JSD ckpt {c}"] = round(float(jsd(Q[new][sel], Pv[sel]).mean() - jsd(Q[ref][sel], Pv[sel]).mean()), 4)
            prow.append(rec)
            if role in ("main", "secondary", "positive control"):
                pj, lj, hj = st[f"JSD {target}"]
                pp, lp, hpp = st[f"partial {target}"]
                sj = "clearly lower (better)" if hj < 0 else "clearly higher (worse)" if lj > 0 else "CI contains 0"
                sp = "clearly higher (better)" if lp > 0 else "clearly lower (worse)" if hpp < 0 else "CI contains 0"
                if role == "positive control":
                    text = ("the V benefit is reproduced by this recipe (JSD)" if hj < 0 else
                            "the V benefit is not reproduced in this few-actor protocol with the PCA-32 representation")
                elif hj < 0 and lp > 0:
                    text = "HSE adds benefit for both the AV distribution fit and the AV disagreement ranking in this protocol"
                elif (hj < 0 and hpp < 0) or (lj > 0 and lp > 0):
                    text = "trade-off: one metric better, the other worse"
                elif hj < 0:
                    text = "HSE adds benefit for the fit of the AV distribution in this protocol"
                elif lp > 0:
                    text = "HSE adds benefit for the AV disagreement ranking in this protocol"
                elif lj > 0 or hpp < 0:
                    text = "worse on at least one metric, better on none"
                else:
                    text = "no benefit found with these data and this readout (check the CI widths)"
                vrow.append({"comparison": f"{new} minus {ref}", "role": role, "JSD": "{:.4f} [{:.4f}, {:.4f}]".format(pj, lj, hj),
                             "JSD status": sj, "partial": "{:.4f} [{:.4f}, {:.4f}]".format(pp, lp, hpp), "partial status": sp, "reading": text})
        tables[f"{target}_paired"] = pd.DataFrame(prow)
        tables[f"{target}_reading"] = pd.DataFrame(vrow)
        R.drop(columns=["p"] + names).to_csv(args.out / f"{target}_clips.csv", index=False)

    F = pd.concat(all_fits)
    F.to_csv(args.out / "fits.csv", index=False)
    sel = F[F.stage == "selection"]
    ch = F[F.stage == "chosen"]
    conv = sel.groupby(["target", "features"]).agg(fits=("converged", "size"), converged=("converged", "mean"),
                                                   max_abs_gradient_max=("max abs gradient", "max"), iterations_max=("iterations", "max")).reset_index()
    refit = ch[np.isfinite(ch["lambda"])].groupby(["target", "features"]).agg(refits=("converged", "size"), converged=("converged", "mean")).reset_index()
    choice = ch.groupby(["target", "features"])["lambda"].agg(
        no_correction=lambda s: f"{np.isinf(s).mean():.0%}",
        chosen=lambda s: ", ".join(f"{('none' if np.isinf(v) else f'{v:g}')} x{c}" for v, c in s.value_counts().sort_index().items())).reset_index()
    tables["convergence_selection"] = conv
    tables["convergence_refit"] = refit
    tables["lambda_choice"] = choice
    lines = ["# Linear residual readout around Qwen's logits, HSE as PCA-32 (exploratory, locked before running)\n",
             "Generated by `analysis/av_complement_residual.py`; recipe and reading rules in its docstring. Heads within the "
             "held-out actors of each s1_noI checkpoint (few actors per head).\n"]
    for name, t in tables.items():
        t.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(t.astype(object).where(t.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
