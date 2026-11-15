"""std_kd diagnostic (15/11/2026, after the locked result; held-out part only): did the student move, and towards
the teacher?

On the held-out actors (predictions at T = 1, V reading), for model pairs against cont seed 0:
  - mean JSD between the two models' predicted distributions and the share of clips whose top label changes;
    reference: cont seed 1 vs cont seed 0 (the change from the training seed alone);
  - the standardised KL to the teacher, KL(softmax(Z(log t)) || softmax(Z(z_S))) with Z(z) = (z - mean) / std, the
    quantity std_kd trained (teacher = mean of the five cross-fitted teachers on the held-out actors, not the
    single per-clip teacher of training), paired difference with bootstrap over actors;
  - cancellation: per clip change in JSD to the V votes; share of clips better / worse by more than 0.005 and the
    mean change within each group.
The training-clip side of the check needs the V predictions of cont and std_kd on training clips (not scored by the
notebook), and the single-teacher comparison needs the HSEmotion features; both are not covered here.

Usage (from the repository root):
    python analysis/std_kd_movement.py --out results/std_kd_movement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import counts, jsd
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, softmax
from rq_analysis import markdown
from visual_probe_analysis import N_OUTER, outer_folds

PAIRS = [("std_kd s0", "cont s0"), ("rank_teacher s0", "cont s0"), ("distill s0", "cont s0"), ("cont s1", "cont s0"), ("rank_teacher s1", "cont s1")]


def read(path: Path) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == "V") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    z = s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]
    return z.sub(z.mean(1), axis=0)


def zstd(z):
    z = z - z.mean(1, keepdims=True)
    return z / np.maximum(z.std(1, keepdims=True), 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    parser.add_argument("--out", type=Path, default=Path("results/std_kd_movement"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    names = sorted({n for p in PAIRS for n in p})
    rows = []
    for k in range(N_OUTER):
        z = {n: read(args.runs / f"distill_{n.split()[0]}_f{k}_{n.split()[1]}" / "scores.jsonl") for n in names}
        t = pd.read_csv(args.teacher / f"teacher_v_f{k}.csv").query("role == 'heldout'").set_index("uid")[LABELS]
        uids = [u for u in z["cont s0"].index if all(u in v.index for v in z.values()) and u in t.index and u in df.index
                and df.loc[u, "fold"] == k and df.loc[u, "v_n"] > 0]
        d = df.loc[uids]
        P = counts(d, "v")
        P = P / P.sum(1, keepdims=True)
        tt = softmax(zstd(np.log(np.maximum(t.loc[uids].to_numpy(float), 1e-8))))
        for i, u in enumerate(uids):
            rec = {"uid": u, "fold": k, "actor": int(d.actor.iloc[i]), "p": P[i], "tstd": tt[i]}
            for n in names:
                Z = z[n].loc[uids].to_numpy(float)
                rec[n] = softmax(Z)[i]
                rec[n + " zstd"] = softmax(zstd(Z))[i]
            rows.append(rec)
    R = pd.DataFrame(rows)
    Pv, T = np.stack(R.p), np.stack(R.tstd)
    boot = Bootstrap(R.actor.to_numpy(), 1000)
    Q = {n: np.stack(R[n]) for n in names}
    QS = {n: np.stack(R[n + " zstd"]) for n in names}
    kl = lambda q: (T * (np.log(np.clip(T, 1e-12, None)) - np.log(np.clip(q, 1e-12, None)))).sum(1)
    out = []
    for a, b in PAIRS:
        move = jsd(Q[a], Q[b])
        top = (Q[a].argmax(1) != Q[b].argmax(1))
        ka, kb = kl(QS[a]), kl(QS[b])
        dv = jsd(Q[a], Pv) - jsd(Q[b], Pv)
        better, worse = dv < -0.005, dv > 0.005
        out.append({"comparison": f"{a} vs {b}", "mean JSD between the two models": round(float(move.mean()), 5),
                    "top label changed": round(float(top.mean()), 4),
                    "standardised KL to teacher: first": round(float(ka.mean()), 4), "second": round(float(kb.mean()), 4),
                    "difference (first - second)": boot(lambda i: ka[i].mean() - kb[i].mean()),
                    "JSD to votes: share better / worse (>0.005)": f"{better.mean():.3f} / {worse.mean():.3f}",
                    "mean change in better / worse clips": f"{dv[better].mean():.4f} / {dv[worse].mean():.4f}",
                    "mean change, all clips": boot(lambda i: dv[i].mean())})
    tb = pd.DataFrame(out)
    tb.to_csv(args.out / "movement.csv", index=False)
    lines = ["# std_kd: movement of the student and distance to the teacher (held-out actors)\n",
             "Generated by `analysis/std_kd_movement.py`; definitions and limits in its docstring.\n", markdown(tb.astype(object)), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
