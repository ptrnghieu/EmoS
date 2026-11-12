"""V distillation pilot, exploratory follow-up (agreed 12/11/2026 after the locked result): can a single global
temperature remove the JSD V gap of `distill` against `cont`, while keeping its partial V gain?

The same protocol for both arms, on their existing V predictions (results/distill): within the held-out actors of
each outer fold, the 3 head folds by actor of av_complement_head.py; for each, one global temperature per model is
chosen on the other two head folds (by mean JSD to the V votes, over a grid in [0.5, 2] that contains T = 1) and
applied to the third. Out-of-sample predictions are pooled; paired bootstrap over actors. The locked pilot table is
unchanged; this is exploratory. Main comparison: distill calibrated minus cont calibrated.

Reading (fixed before running):
    JSD gap gone or reversed, partial still higher -> grounds to prefer a transfer that controls the softness separately
    JSD still worse after calibration              -> global confidence does not explain enough; look at the
                                                      per-clip allocation of probability and the teacher's errors
    partial advantage lost                         -> no evidence that the ranking gain survives calibration
Even the first case shows only that the gap is correctable at the output, not that the teacher's temperature
caused it during training.

Usage (from the repository root):
    python analysis/distill_calibration.py --out results/distill_calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from av_complement_head import N_BOOT, N_HEAD, head_folds
from cremad_human import counts, jsd
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, fmt, softmax
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman

GRID = np.unique(np.concatenate([np.round(np.exp(np.linspace(np.log(0.5), np.log(2.0), 61)), 4), [1.0]]))
ARMS = ("cont", "distill")


def read_v(path: Path) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == "V") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    z = s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
    return z.sub(z.mean(1), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--out", type=Path, default=Path("results/distill_calibration"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    rows, temps = [], []
    for k in range(N_OUTER):
        z = {a: read_v(args.runs / f"distill_{a}_f{k}_s0" / "scores.jsonl") for a in ARMS}
        uids = [u for u in z["cont"].index if u in z["distill"].index and u in df.index and df.loc[u, "fold"] == k and df.loc[u, "v_n"] > 0]
        d = df.loc[uids]
        P = counts(d, "v")
        P = P / P.sum(1, keepdims=True)
        hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
        out = {f"{a} (T=1)": softmax(z[a].loc[uids].to_numpy(float)) for a in ARMS}
        for a in ARMS:
            Z = z[a].loc[uids].to_numpy(float)
            cal = np.zeros_like(P)
            for j in range(N_HEAD):
                te, tr = hf == j, hf != j
                scores = [jsd(softmax(Z[tr], t), P[tr]).mean() for t in GRID]
                t = float(GRID[int(np.argmin(scores))])
                cal[te] = softmax(Z[te], t)
                temps.append({"arm": a, "outer fold": k, "head fold": j, "temperature": t,
                              "fit JSD at T=1": round(float(jsd(softmax(Z[tr]), P[tr]).mean()), 4), "fit JSD at T": round(float(min(scores)), 4)})
            out[f"{a} (calibrated)"] = cal
        for i, u in enumerate(uids):
            rows.append({"uid": u, "fold": k, "actor": int(d.actor.iloc[i]), "emotion": d.intended.iloc[i], "p": P[i], **{n: q[i] for n, q in out.items()}})
    R = pd.DataFrame(rows)
    Pv = np.stack(R.p)
    hp, emo, act, fo = norm_entropy(Pv), R.emotion.to_numpy(), R.actor.to_numpy(), R.fold.to_numpy()
    names = [f"{a} ({c})" for c in ("T=1", "calibrated") for a in ARMS]
    Q = {n: np.stack(R[n]) for n in names}
    boot = Bootstrap(act, N_BOOT)

    def met(q):
        j, hq = jsd(q, Pv), norm_entropy(q)
        return {"JSD V": lambda i: j[i].mean(), "partial V": lambda i: partial_spearman(hq[i], hp[i], emo[i]),
                "Spearman(H(q), A^V)": lambda i: spearman(hq[i], hp[i]), "mean H(q)": lambda i: hq[i].mean()}

    readings = pd.DataFrame([{"predictor": n, **{m: boot(f) for m, f in met(Q[n]).items()},
                              **{f"JSD fold {k}": round(float(jsd(Q[n], Pv)[fo == k].mean()), 4) for k in range(N_OUTER)}} for n in names])
    pairs = [("distill (calibrated)", "cont (calibrated)", "main"), ("distill (T=1)", "cont (T=1)", "locked pilot (recomputed)"),
             ("distill (calibrated)", "distill (T=1)", "calibration effect"), ("cont (calibrated)", "cont (T=1)", "calibration effect")]
    prow, st = [], {}
    for a, b, role in pairs:
        ma, mb = met(Q[a]), met(Q[b])
        rec = {"comparison": f"{a} minus {b}", "role": role}
        for m in ma:
            s = np.array([ma[m](i) - mb[m](i) for i in boot.samples])
            pt = ma[m](boot.all) - mb[m](boot.all)
            lo, hi = np.nanpercentile(s, [2.5, 97.5])
            rec[m] = fmt(pt, s)
            st[(a, b, m)] = (pt, lo, hi)
        rec.update({f"JSD fold {k}": round(float(jsd(Q[a], Pv)[fo == k].mean() - jsd(Q[b], Pv)[fo == k].mean()), 4) for k in range(N_OUTER)})
        prow.append(rec)
    _, lj, hj = st[("distill (calibrated)", "cont (calibrated)", "JSD V")]
    _, lp, hpp = st[("distill (calibrated)", "cont (calibrated)", "partial V")]
    p_ok = lp > 0
    if lj > 0:
        text = "JSD still worse after calibration: global confidence does not explain enough"
    elif not p_ok:
        text = "partial advantage not shown after calibration: no evidence that the ranking gain survives"
    else:
        text = "JSD gap gone or reversed and partial still higher: grounds to control the softness separately when distilling"
    reading = pd.DataFrame([{"JSD V (distill cal - cont cal)": "{:.4f} [{:.4f}, {:.4f}]".format(*st[("distill (calibrated)", "cont (calibrated)", "JSD V")]),
                             "JSD status": "clearly lower" if hj < 0 else "clearly higher" if lj > 0 else "CI contains 0",
                             "partial V": "{:.4f} [{:.4f}, {:.4f}]".format(*st[("distill (calibrated)", "cont (calibrated)", "partial V")]),
                             "partial status": "clearly higher" if lp > 0 else "clearly lower" if hpp < 0 else "CI contains 0", "reading": text}])
    T = pd.DataFrame(temps)
    tsum = T.groupby("arm").agg(temperature_mean=("temperature", "mean"), temperature_min=("temperature", "min"),
                                temperature_max=("temperature", "max"), fits=("temperature", "size")).reset_index().round(3)
    erows = []
    for e in sorted(set(emo)):
        sel = np.flatnonzero(emo == e)
        be = Bootstrap(act[sel], N_BOOT)
        jn, jr = jsd(Q["distill (calibrated)"], Pv), jsd(Q["cont (calibrated)"], Pv)
        erows.append({"intended emotion": e, "clips": len(sel), "JSD difference (distill cal - cont cal)": be(lambda i: jn[sel][i].mean() - jr[sel][i].mean())})
    tables = {"reading": reading, "readings": readings, "paired": pd.DataFrame(prow), "temperatures": tsum,
              "by_emotion_paired": pd.DataFrame(erows)}
    T.to_csv(args.out / "temperature_fits.csv", index=False)
    lines = ["# V distillation pilot: global temperature calibration of both arms (exploratory)\n",
             "Generated by `analysis/distill_calibration.py`; protocol and reading in its docstring. The locked pilot "
             "result (results/distill_compare) is unchanged.\n"]
    for name, t in tables.items():
        t.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(t.astype(object).where(t.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
