"""Step 1 (criteria revised and locked 16/11/2026, before the extraction run and before any result of it).

Decision criterion: can the representation tell apart clips of the SAME intended emotion that make viewers disagree
more or less, and does this ability fall after the merger MLP? The KL decomposition below is kept as a description of
where the probes err; it is no longer the decision criterion (a probe can improve mainly by recognising the emotion
category better, and Delta_alt / I do not isolate an ambiguity-specific mechanism).

Primary quantity: within-emotion ranking of disagreement, rho_within = mean over the six intended emotions of
Spearman(H(q_i), H(p_i)) on the held-out clips (normalised entropies; q from the six-class probe, p the V votes).
Delta_within = rho_within(after) - rho_within(before) (negative = worse after the merger MLP), paired bootstrap over
actors. Supplementary: partial Spearman given the intended emotion, per-emotion Spearman, JSD, KL and its
decomposition. Human labels are used only to train the probe's target and to score, never as probe inputs.
    Delta_within clearly < 0 in the main variant, same sign in both sensitivity variants
        -> the within-emotion disagreement ranking is less decodable after the merger MLP under these readouts:
           premise for an ambiguity-specific intervention at this position
    only KL / E_ref / JSD worse, Delta_within not clearly < 0
        -> not enough to develop an ambiguity method at the merger
    the sign of Delta_within changes across variants
        -> the merger effect is not separated from aggregation and readout
    no degradation of the within-emotion ranking
        -> stop the merger direction
Entropy alone does not tell the alternative interpretations apart; the per-emotion KL parts are reported for that.

Original question (kept for the description): is the information about the alternative interpretations
less decodable after the merger MLP than before it, relative to the information about the reference class?

Input: the base-model run of kaggle/probe_visual.ipynb with MERGER_IO = 1 (vprobe_base_A0_mio): per clip, the mean and
SD over token groups of the input of the merger MLP (after ln_q, 2x2 patches concatenated, 5120-d) and of its output
(3584-d). Base model only: the Stage 1 adapters hold no vision-encoder or merger weights (checked: 224 LoRA keys, all
in the language model's attention), and processor and frame sampling are unchanged.

Probe at each position (the same recipe and splits on both sides): a multinomial linear probe of the full six-class
V vote distribution (soft-label CE + L2 on all weights and biases, L-BFGS to convergence), the reference class used
only for scoring. Splits: the three Stage 1 outer folds by actor; the L2 weight in {1e-4, ..., 1e2} chosen by JSD on an
inner actor split (15%) of the training part, then refit on the whole training part. Preprocessing fitted on the
part being fitted. Main: mean -> StandardScaler -> PCA 256 -> StandardScaler. Sensitivity (fixed before running):
mean + SD with PCA 256; mean without PCA (StandardScaler only).
Diagnostics (not decision criteria): KL(p || q) per clip decomposed into E_ref and E_alt (analysis/kl_decomposition.py;
ties averaged); Delta_alt = mean E_alt(post) - mean E_alt(pre), Delta_ref likewise (positive = worse after the merger
MLP), I = Delta_alt - Delta_ref (a difference of absolute error increases, not a share of lost information); JSD and
top-label recognition are read from the same probes. Paired bootstrap over actors, the same resamples for all
quantities. They say where a probe errs; a worse KL or E_ref alone does not support an ambiguity method.
"Clearly < 0": 95% interval entirely below 0. A negative result (any branch other than the first) is used to stop
the merger direction. Allowed statement if the first branch holds: "the within-emotion ranking of viewer
disagreement is less decodable from the aggregated representation after the merger MLP, under the readouts tried"
(the mean of a nonlinear map is not the map of the mean; the effect of ln_q is not part of this comparison).

Usage (from the repository root):
    python analysis/merger_probe.py --run results/visual_probe/vprobe_base_A0_mio --out results/merger_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from av_complement_head import N_BOOT
from av_complement_residual import fit_residual, softmax
from cremad_human import counts, jsd
from hsemotion_head import INNER_VAL_SHARE
from kl_decomposition import decompose
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, fmt
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, SPLIT_SEED, norm_entropy, outer_folds, partial_spearman

LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
N_PCA = 256
VARIANTS = {"main: mean, PCA 256": (("mean",), N_PCA), "sensitivity: mean + SD, PCA 256": (("mean", "std"), N_PCA),
            "sensitivity: mean, no PCA": (("mean",), None)}


def load(run: Path):
    arr = {}
    for f in sorted(run.glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        for k in ("uid", "pre_mean", "pre_std", "post_mean", "post_std"):
            arr.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in arr.items()}


class Prep:
    def __init__(self, X, idx, k):
        self.m, self.s = X[idx].mean(0), X[idx].std(0) + 1e-6
        Z = (X[idx] - self.m) / self.s
        self.k = k
        if k:
            self.zm = Z.mean(0)
            _, _, vt = np.linalg.svd(Z - self.zm, full_matrices=False)
            self.V = vt[:k].T
            C = (Z - self.zm) @ self.V
            self.cm, self.cs = C.mean(0), C.std(0) + 1e-6

    def __call__(self, X):
        Z = (X - self.m) / self.s
        if not self.k:
            return Z
        return (((Z - self.zm) @ self.V) - self.cm) / self.cs


def probe(X, P, actors, tr, te, seed):
    tr_act = sorted(set(actors[tr]))
    inner = set(np.random.default_rng(seed).permutation(tr_act)[:max(1, round(INNER_VAL_SHARE * len(tr_act)))])
    val = tr & np.isin(actors, list(inner))
    fit = tr & ~val
    best, conv = None, []
    for lam in LAMBDAS:
        pr = Prep(X, fit, VARIANTS_K)
        xf, xv = pr(X[fit]), pr(X[val])
        W, b, info = fit_residual(np.zeros((fit.sum(), 6)), xf, P[fit], lam)
        conv.append(info["converged"])
        s = float(jsd(softmax(xv @ W.T + b), P[val]).mean())
        if best is None or s < best[0] - 1e-9:
            best = (s, lam)
    pr = Prep(X, tr, VARIANTS_K)
    W, b, info = fit_residual(np.zeros((tr.sum(), 6)), pr(X[tr]), P[tr], best[1])
    conv.append(info["converged"])
    return softmax(pr(X[te]) @ W.T + b), best[1], all(conv)


VARIANTS_K = N_PCA


def main() -> None:
    global VARIANTS_K
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("results/visual_probe/vprobe_base_A0_mio"))
    parser.add_argument("--out", type=Path, default=Path("results/merger_probe"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.run / "meta.json").read_text())
    assert meta.get("merger_io") and meta.get("model_kind") == "base" and meta.get("condition") == "V", meta
    A = load(args.run)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    keep = np.array([u in df.index and df.loc[u, "v_n"] > 0 for u in A["uid"]])
    uids = A["uid"][keep]
    d = df.loc[uids]
    C = counts(d, "v")
    P = C / C.sum(1, keepdims=True)
    actors, folds = d.actor.to_numpy(), d.fold.to_numpy()
    boot = Bootstrap(actors, N_BOOT)
    tables, sel_rows, E = {}, [], {}
    for vname, (stats, k) in VARIANTS.items():
        VARIANTS_K = k
        for pos in ("pre", "post"):
            X = np.hstack([A[f"{pos}_{s}"][keep].astype(np.float64) for s in stats])
            Q = np.zeros_like(P)
            for f in range(N_OUTER):
                te, tr = folds == f, folds != f
                Q[te], lam, ok = probe(X, P, actors, tr, te, SPLIT_SEED + 600 + f)
                sel_rows.append({"variant": vname, "position": pos, "fold": f, "lambda": lam, "converged": ok, "dims in": X.shape[1]})
            E[(vname, pos)] = (*decompose(P, Q), jsd(Q, P), norm_entropy(Q))
    hp, emo = norm_entropy(P), d.intended.to_numpy()
    EMOS = sorted(set(emo))
    within = lambda h, idx: float(np.nanmean([spearman(h[idx][emo[idx] == e], hp[idx][emo[idx] == e]) for e in EMOS]))
    rows, drows, erows = [], [], []
    for vname in VARIANTS:
        for pos in ("pre", "post"):
            er, ea, j, h = E[(vname, pos)]
            rows.append({"variant": vname, "position": "before merger MLP" if pos == "pre" else "after merger MLP",
                         "rho_within (mean over emotions)": boot(lambda i: within(h, i)),
                         "partial Spearman": boot(lambda i: partial_spearman(h[i], hp[i], emo[i])),
                         "JSD V": boot(lambda i: j[i].mean()), "KL": boot(lambda i: (er[i] + ea[i]).mean()),
                         "E_ref": boot(lambda i: er[i].mean()), "E_alt": boot(lambda i: ea[i].mean())})
        r0, a0, j0, h0 = E[(vname, "pre")]
        r1, a1, j1, h1 = E[(vname, "post")]
        st = {}
        for name, fn in (("Delta_within", lambda i: within(h1, i) - within(h0, i)),
                         ("Delta_partial", lambda i: partial_spearman(h1[i], hp[i], emo[i]) - partial_spearman(h0[i], hp[i], emo[i])),
                         ("Delta_JSD", lambda i: (j1[i] - j0[i]).mean()), ("Delta_KL", lambda i: (r1[i] + a1[i] - r0[i] - a0[i]).mean()),
                         ("Delta_alt", lambda i: (a1[i] - a0[i]).mean()), ("Delta_ref", lambda i: (r1[i] - r0[i]).mean()),
                         ("I = Delta_alt - Delta_ref", lambda i: (a1[i] - a0[i]).mean() - (r1[i] - r0[i]).mean())):
            smp = np.array([fn(i) for i in boot.samples])
            st[name] = (fn(boot.all), *np.nanpercentile(smp, [2.5, 97.5]))
        drows.append({"variant": vname, **{n: "{:.4f} [{:.4f}, {:.4f}]".format(*v) for n, v in st.items()},
                      "_dw": st["Delta_within"], "_dk": st["Delta_KL"], "_dj": st["Delta_JSD"]})
        for e in EMOS:
            sel = np.flatnonzero(emo == e)
            be = Bootstrap(actors[sel], N_BOOT)
            erows.append({"variant": vname, "intended emotion": e, "clips": len(sel),
                          "Spearman before": round(float(spearman(h0[sel], hp[sel])), 4), "Spearman after": round(float(spearman(h1[sel], hp[sel])), 4),
                          "difference (after - before)": be(lambda i: spearman(h1[sel][i], hp[sel][i]) - spearman(h0[sel][i], hp[sel][i])),
                          "Delta_alt": be(lambda i: (a1[sel][i] - a0[sel][i]).mean()), "Delta_ref": be(lambda i: (r1[sel][i] - r0[sel][i]).mean())})
    main_v = "main: mean, PCA 256"
    dw = {r["variant"]: r["_dw"] for r in drows}
    signs = {np.sign(v[0]) for v in dw.values()}
    m = dw[main_v]
    other_worse = any(r["_dk"][1] > 0 or r["_dj"][1] > 0 for r in drows if r["variant"] == main_v)
    if len(signs) > 1:
        overall = "the sign of Delta_within changes across variants: the merger effect is not separated from aggregation and readout"
    elif m[2] < 0:
        overall = "Delta_within clearly < 0 (main), same sign in the sensitivity variants: premise for an ambiguity-specific intervention at this position"
    elif other_worse:
        overall = "only KL / JSD worse after the merger MLP, Delta_within not clearly < 0: not enough to develop an ambiguity method at the merger"
    else:
        overall = "no degradation of the within-emotion ranking: stop the merger direction"
    for r in drows:
        for k in ("_dw", "_dk", "_dj"):
            r.pop(k)
    tables["probes"] = pd.DataFrame(rows)
    tables["deltas"] = pd.DataFrame(drows)
    tables["by_emotion"] = pd.DataFrame(erows)
    tables["overall"] = pd.DataFrame([{"overall": overall, "clips": int(keep.sum())}])
    tables["selection"] = pd.DataFrame(sel_rows)
    lines = ["# Step 1: decodability of the vote structure before and after the merger MLP (base model, V)\n",
             "Generated by `analysis/merger_probe.py`; protocol and reading rules in its docstring.\n"]
    for name, tb in tables.items():
        tb.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(tb.astype(object).where(tb.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
