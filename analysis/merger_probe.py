"""Step 1 (locked 16/11/2026, before the extraction run): is the information about the alternative interpretations
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
Scoring: KL(p || q) per clip decomposed into E_ref and E_alt (analysis/kl_decomposition.py; ties averaged).
Delta_alt = mean E_alt(post) - mean E_alt(pre), Delta_ref likewise (positive = worse after the merger MLP),
I = Delta_alt - Delta_ref (a difference of absolute error increases, not a share of lost information). Paired
bootstrap over actors, the same resamples for both parts and I.

Reading: Delta_alt > 0 and I > 0, stable across the sensitivity analyses -> supports the selective-degradation
hypothesis for this measurement; both parts worse with no clear difference -> general degradation, selectivity not
shown; result changes with pooling / PCA -> the merger effect is not separated from aggregation and readout;
Delta_alt not > 0 -> no premise for a preservation loss at this position. Allowed statement: "the decodability of the
judgment structure from the aggregated representation falls after the merger MLP, under the readouts tried" (the
mean of a nonlinear map is not the map of the mean; the effect of ln_q is not part of this comparison).

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
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, SPLIT_SEED, outer_folds

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
            E[(vname, pos)] = (*decompose(P, Q), jsd(Q, P))
    rows, drows = [], []
    for vname in VARIANTS:
        for pos in ("pre", "post"):
            er, ea, j = E[(vname, pos)]
            rows.append({"variant": vname, "position": "before merger MLP" if pos == "pre" else "after merger MLP",
                         "KL": boot(lambda i: (er[i] + ea[i]).mean()), "E_ref": boot(lambda i: er[i].mean()), "E_alt": boot(lambda i: ea[i].mean()),
                         "JSD V": boot(lambda i: j[i].mean())})
        r0, a0, _ = E[(vname, "pre")]
        r1, a1, _ = E[(vname, "post")]
        st = {}
        for name, fn in (("Delta_alt", lambda i: (a1[i] - a0[i]).mean()), ("Delta_ref", lambda i: (r1[i] - r0[i]).mean()),
                         ("I = Delta_alt - Delta_ref", lambda i: (a1[i] - a0[i]).mean() - (r1[i] - r0[i]).mean())):
            s = np.array([fn(i) for i in boot.samples])
            st[name] = (fn(boot.all), *np.nanpercentile(s, [2.5, 97.5]))
        da, di = st["Delta_alt"], st["I = Delta_alt - Delta_ref"]
        dr = st["Delta_ref"]
        if da[1] > 0 and di[1] > 0:
            read = "Delta_alt > 0 and I > 0"
        elif da[1] > 0 and dr[1] > 0:
            read = "both parts worse, no clear difference (I not > 0)"
        elif da[1] > 0:
            read = "Delta_alt > 0, I not clearly > 0"
        else:
            read = "Delta_alt not > 0"
        drows.append({"variant": vname, **{n: "{:.4f} [{:.4f}, {:.4f}]".format(*v) for n, v in st.items()}, "reading": read})
    tables["probes"] = pd.DataFrame(rows)
    tables["deltas"] = pd.DataFrame(drows)
    reads = {r["reading"] for r in drows}
    overall = ("supports the selective-degradation hypothesis for this measurement (stable across the sensitivity analyses)" if reads == {"Delta_alt > 0 and I > 0"}
               else "Delta_alt not > 0 in any variant: no premise for a preservation loss at this position" if reads == {"Delta_alt not > 0"}
               else "the result depends on pooling / PCA: the merger effect is not separated from aggregation and readout" if len(reads) > 1
               else f"stable across variants: {next(iter(reads))}")
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
